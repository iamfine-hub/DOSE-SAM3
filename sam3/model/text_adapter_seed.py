"""
SEED 文本适配器: 在 SAM3 文本编码器输出端外挂轻量残差模块, 用于 FSS.

BlockA: 通用 FSS 偏置 (meta 训练后固定)
BlockB: 逐 episode 支持图适应 (消融用)

形状约定: 输入输出均为 [seq_len, batch, dim] = [32, B, 256]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockA(nn.Module):
    """通用 FSS 偏置: LN → 256→512→GELU → 512→256, 输出零初始化 + 残差."""

    def __init__(self, dim: int = 256, hidden: int = 512):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self._zero_init()

    def _zero_init(self):
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x + residual


class BlockB(nn.Module):
    """逐 episode 适应: LN → 线性 → GELU + 残差, 零初始化 = identity 起点."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self._zero_init()

    def _zero_init(self):
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.fc(self.ln(x))) + x


class TextAdapterSEED(nn.Module):
    """文本适配器: 静态 BlockA + 消融用 BlockB."""

    def __init__(self, dim: int = 256, hidden_a: int = 512):
        super().__init__()
        self.block_a = BlockA(dim, hidden_a)
        self.block_b = BlockB(dim)

    def reset_B(self):
        """B 恢复零初始化 (恒等), 每 episode 开始时调用."""
        self.block_b._zero_init()

    def forward_A(self, x: torch.Tensor) -> torch.Tensor:
        return self.block_a(x)

    def forward_B(self, x: torch.Tensor) -> torch.Tensor:
        return self.block_b(x)

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """并行: x + res_A + res_B."""
        return self.block_a(x) + self.block_b(x) - x

    def forward_serial(self, x: torch.Tensor) -> torch.Tensor:
        """串行: B(A(x))."""
        return self.block_b(self.block_a(x))

    @property
    def num_params(self):
        a = sum(p.numel() for p in self.block_a.parameters())
        b = sum(p.numel() for p in self.block_b.parameters())
        return {"A": a, "B": b, "total": a + b}
