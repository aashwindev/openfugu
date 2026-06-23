"""Router SFT with KL divergence on soft targets."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from openfugu.router.backbone import RouterBackbone
from openfugu.router.head import SelectionHead
from openfugu.training.datasets import build_training_mix, load_jsonl


def train_sft(config_path: str = "config/router_train.yaml") -> Path:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    labels_path = Path("data/router_labels.jsonl")
    if labels_path.exists():
        import json

        samples = []
        with labels_path.open() as f:
            for line in f:
                obj = json.loads(line)
                samples.append((obj["question"], obj["soft_targets"]))
    else:
        examples = build_training_mix(cfg.get("sft", cfg))
        # Synthetic uniform targets when labels not collected
        num_workers = 3
        uniform = [1.0 / num_workers] * num_workers
        samples = [(ex.question, uniform) for ex in examples[:50]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = RouterBackbone(model_name=cfg.get("backbone", "Qwen/Qwen3-0.6B"), freeze=True).to(device)
    num_workers = len(samples[0][1])
    head = SelectionHead(backbone.hidden_dim, num_workers).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.get("sft", {}).get("learning_rate", 1e-4))
    epochs = cfg.get("sft", {}).get("epochs", 3)
    temperature = cfg.get("temperature", 1.0)

    head.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for question, targets in samples:
            out = backbone.encode(question, device=device)
            logits = head(out.hidden_state)[0]
            log_probs = F.log_softmax(logits / temperature, dim=-1)
            target = torch.tensor(targets, device=device, dtype=torch.float32)
            target = target / target.sum()
            loss = F.kl_div(log_probs, target, reduction="sum")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"SFT epoch {epoch + 1}/{epochs} loss={total_loss / max(len(samples), 1):.4f}")

    out_dir = Path(cfg.get("output_dir", "checkpoints/router"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "router_sft.pt"
    torch.save(
        {
            "backbone": cfg.get("backbone", "Qwen/Qwen3-0.6B"),
            "head": head.state_dict(),
            "num_workers": num_workers,
        },
        ckpt_path,
    )
    print(f"Saved SFT checkpoint to {ckpt_path}")
    return ckpt_path
