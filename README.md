# DOSE-SAM3

**Beyond Support Adaptation: A Light Text Adapter for Few-Shot Segmentation with SAM3**

DOSE (Domain Offset Squeezed Embedding) is a lightweight text-side adapter for Few-Shot Segmentation (FSS) with SAM3. It applies a zero-initialized residual MLP to the text embeddings output by the SAM3 text encoder, enabling segmentation from class-name text prompts **without any support images or online fine-tuning**.

## Method

```
text prompt ──> SAM3 text encoder ──> [32, 1, 256] ──> BlockA ──> SAM3 decoder ──> masks
                                                   (x + α·Δ, Δ = A(x) - x)
```

- **BlockA**: `LN → Linear(256→512) → GELU → Linear(512→256) → +x` (residual), fc2 zero-initialized. ~263K params (0.03% of SAM3).
- **Zero-init**: at training start Δ=0, model behaves identically to vanilla SAM3.
- **Training**: on base-class episodes with grounding BCE loss (support unused).
- **Inference**: only the class-name text prompt is needed — no support images, no online fine-tuning, no new-class annotations.

## Results

| Setting | PASCAL-5i (mIoU) | COCO-20i (mIoU) |
|---|---|---|
| Vanilla SAM3 (text prompt) | 78.5 | 64.1 |
| DOSE-SAM3 (0-shot, no support) | **90.5** | **78.4** |
| Cross-dataset: DOSE (COCO→PASCAL) | **91.3** | — |
| Cross-dataset: DOSE (PASCAL→COCO) | — | **78.1** |

## Installation

1. Clone this repo and set up SAM3 dependencies.
2. Follow the [SAM3 official repo](https://github.com/meta-ai/sam3) for model weights and environment setup.
3. Prepare the FSS data (PASCAL-5i / COCO-20i) and place the split lists under `lists/`.

Dependencies: `torch`, `numpy`, `PIL`, `tqdm`, `iopath` (see `requirements.txt`).

## Usage

### Train + Evaluate DOSE

```bash
python FSS_sam3_seed.py \
    --checkpoint /path/to/sam3.pt \
    --pascal_root /path/to/VOC2012 \
    --coco_root /path/to/MSCOCO2014 \
    --lists_dir ./lists \
    --dataset both \
    --device cuda \
    --epochs 1 --episodes_per_epoch 200 --lr_outer 3e-4 --seed 42 \
    --test_episodes 1000 \
    --save_adapters ./adapters_A
```

### Evaluate with saved adapters

```bash
python FSS_sam3_seed.py \
    --checkpoint /path/to/sam3.pt \
    --pascal_root /path/to/VOC2012 \
    --coco_root /path/to/MSCOCO2014 \
    --lists_dir ./lists \
    --dataset both \
    --device cuda \
    --load_adapters ./adapters_A
```

### Offset sensitivity analysis (Fig. 4-3)

```bash
python exp_offset_sensitivity.py \
    --checkpoint /path/to/sam3.pt \
    --adapter_dir ./adapters_A \
    --dataset pascal \
    --test_episodes 500 --device cuda \
    --out_csv exp_out/sens_pascal.csv
python plot_offset_sensitivity.py --csv exp_out/sens_pascal.csv --out_dir figs
```

### Cross-dataset generalization (Table 4)

```bash
python exp_cross_dataset.py \
    --checkpoint /path/to/sam3.pt \
    --dataset_source pascal --dataset_target coco \
    --epochs 1 --episodes_per_epoch 200 --lr_outer 3e-4 --seed 42 \
    --test_episodes 1000 --folds 0,1,2,3 --device cuda \
    --out_csv exp_out/cross_p2c.csv
```

### Mechanism visualization

```bash
python visualize_text_embedding.py --checkpoint /path/to/sam3.pt \
    --adapter_dir ./adapters_A --out_dir ./figs
```

## Key Files

| File | Description |
|---|---|
| `sam3/model/text_adapter_seed.py` | BlockA adapter definition (core contribution) |
| `sam3/model/sam3_image_processor_seed.py` | Processor with text-adapter injection |
| `FSS_sam3_seed.py` | Main train/evaluate script |
| `exp_offset_sensitivity.py` | Offset magnitude sensitivity scan |
| `exp_cross_dataset.py` | Cross-dataset generalization experiment |
| `plot_*.py` | Figure reproduction scripts |

## License

The `sam3/` directory is from Meta AI's [SAM3](https://github.com/meta-ai/sam3) — see its original license. The DOSE adapter code is released under the [MIT License](LICENSE).
