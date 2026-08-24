"""
SEED Processor: 在 Sam3Processor 的基础上外挂 TextAdapterSEED (静态 BlockA).
在文本编码后、decoder 前应用 A 适配.

训练: 用 query 图 + BCE 更新 A
测试: 用训练好的 A 预测 query
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image

from sam3.model import box_ops
from sam3.model.data_misc import interpolate
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.text_adapter_seed import TextAdapterSEED


class Sam3ProcessorSEED(Sam3Processor):
    """带静态 A 适配器的 SAM3 Processor."""

    def __init__(self, model, adapter: TextAdapterSEED, resolution=1008,
                 device="cuda", confidence_threshold=0.5):
        super().__init__(model, resolution=resolution, device=device,
                         confidence_threshold=confidence_threshold)
        self.adapter = adapter.to(device)

    # ---- 文本编码 + 适配器注入 ----

    def _encode_text(self, prompt: str, state: Dict) -> Dict:
        """编码文本, 返回 text_outputs, 不调用 _forward_grounding."""
        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()
        return text_outputs

    def _forward_grounding_grad(self, state: Dict) -> Dict:
        """
        与父类 _forward_grounding 完全相同的逻辑,
        但不带 @torch.inference_mode() 装饰器, 允许梯度流动 (训练 A 用).
        """
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state

    # ---- 模型预测 (用于测试) ----

    @torch.no_grad()
    def predict_query(self, image: PIL.Image.Image, prompt: str, mode: str = "a") -> Dict:
        """纯文本预测. mode: a / b / ab_parallel / ab_serial."""
        state = self.set_image(image)
        self._encode_text(prompt, state)
        lang_feat = state["backbone_out"]["language_features"]
        if mode == "a":
            adapted = self.adapter.forward_A(lang_feat)
        elif mode == "b":
            adapted = self.adapter.forward_B(lang_feat)
        elif mode == "ab_serial":
            adapted = self.adapter.forward_serial(lang_feat)
        else:  # ab_parallel
            adapted = self.adapter.forward_parallel(lang_feat)
        state["backbone_out"]["language_features"] = adapted
        state = self._forward_grounding(state)
        return {
            "masks": state.get("masks", []),
            "scores": state.get("scores", []),
        }

    # ---- 消融: B 的逐 episode 适应 ----

    def inner_loop_step(self, state: Dict, gt_mask: np.ndarray,
                        class_id: int = 1, mode: str = "ab_parallel") -> torch.Tensor:
        """单步 inner loop: A 冻结, B 训练. 按 mode 组合前向, 返回 support 上的 BCE loss."""
        x = state["backbone_out"]["language_features"]
        with torch.no_grad():
            a_out = self.adapter.forward_A(x)
        if mode == "b":
            adapted = self.adapter.forward_B(x)
        elif mode == "ab_serial":
            adapted = self.adapter.block_b(a_out)
        else:  # ab_parallel
            adapted = self.adapter.forward_parallel(x)

        state["backbone_out"]["language_features"] = adapted
        state = self._forward_grounding_grad(state)

        probs = state.get("masks_logits", None)
        scores = state.get("scores", [])
        if probs is None or probs.numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        probs = probs.squeeze(1)
        if probs.ndim == 2:
            probs = probs.unsqueeze(0)

        gt_tensor = torch.from_numpy(gt_mask).to(self.device)
        valid = gt_tensor != 255
        gt_fg = (gt_tensor == class_id) & valid

        best_idx = 0
        if len(scores) > 1:
            s_list = [s.item() if hasattr(s, 'item') else s for s in scores]
            best_idx = int(np.argmax(s_list))

        pred_prob = probs[best_idx]
        return F.binary_cross_entropy(pred_prob, gt_fg.float(), reduction="mean")

    def reset_episode(self):
        """每个新 episode 开始时调用: B 归零."""
        self.adapter.reset_B()

    def get_B_params(self) -> List[torch.Tensor]:
        return list(self.adapter.block_b.parameters())

    # ---- 保存/加载 A ----

    def save_A(self, path: str):
        torch.save(self.adapter.block_a.state_dict(), path)

    def load_A(self, path: str):
        self.adapter.block_a.load_state_dict(torch.load(path, map_location=self.device))

    def get_A_params(self) -> List[torch.Tensor]:
        return list(self.adapter.block_a.parameters())
