"""Singular-value scale fine-tuning for efficient backbone adaptation."""

from __future__ import annotations

import torch
import torch.nn as nn


class SVScaleAdapter(nn.Module):
    """Train only singular value scales of selected weight matrices."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        w = linear.weight.data.float()
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        self.register_buffer("u", u)
        self.register_buffer("vh", vh)
        self.log_s = nn.Parameter(torch.log(s.clamp(min=1e-6)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.exp(self.log_s)
        weight = self.u @ torch.diag(s) @ self.vh
        return torch.nn.functional.linear(x, weight, None)


def attach_sv_adapters(model: nn.Module, layer_indices: list[int] | None = None) -> list[SVScaleAdapter]:
    """Attach SV adapters to selected transformer MLP layers."""
    adapters: list[SVScaleAdapter] = []
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        return adapters

    indices = layer_indices or [len(layers) - 1]
    for idx in indices:
        if idx < 0 or idx >= len(layers):
            continue
        layer = layers[idx]
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for name in ("gate_proj", "up_proj", "down_proj"):
            mod = getattr(mlp, name, None)
            if isinstance(mod, nn.Linear):
                adapters.append(SVScaleAdapter(mod))
    return adapters
