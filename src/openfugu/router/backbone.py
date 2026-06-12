"""Qwen3-0.6B backbone with early-position hidden state extraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BackboneOutput:
    hidden_state: torch.Tensor
    logits: torch.Tensor | None = None


class RouterBackbone(nn.Module):
    """Frozen LM backbone; extract hidden state at head_input position."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        head_position: int = -1,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self.head_position = head_position

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        self.hidden_dim = self.model.config.hidden_size

    def encode(self, text: str, device: torch.device | None = None) -> BackboneOutput:
        dev = device or next(self.model.parameters()).device
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        ).to(dev)

        with torch.set_grad_enabled(any(p.requires_grad for p in self.model.parameters())):
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[-1]  # (1, seq, dim)
        pos = self.head_position if self.head_position >= 0 else hidden_states.shape[1] + self.head_position
        pos = min(max(pos, 0), hidden_states.shape[1] - 1)
        h = hidden_states[0, pos, :]

        return BackboneOutput(hidden_state=h, logits=outputs.logits)
