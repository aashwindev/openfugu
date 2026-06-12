"""Lightweight selection head: hidden state to worker logits."""

from __future__ import annotations

import torch
import torch.nn as nn


class SelectionHead(nn.Module):
    """Linear head projecting hidden state to worker logits (~10K params for L=3, d=1024)."""

    def __init__(self, hidden_dim: int, num_workers: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_workers, bias=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (batch, hidden_dim) or (hidden_dim,)
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        return self.linear(hidden)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
