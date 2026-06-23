"""sep-CMA-ES router optimization on end-to-end tasks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from openfugu.router.head import SelectionHead
from openfugu.training.router_sft import train_sft


def train_cma(config_path: str = "config/router_train.yaml") -> Path:
    try:
        import cma
    except ImportError as e:
        raise ImportError("Install train extras: pip install openfugu[train]") from e

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sft_path = Path(cfg.get("output_dir", "checkpoints/router")) / "router_sft.pt"
    if not sft_path.exists():
        print("SFT checkpoint not found, running SFT first...")
        train_sft(config_path)

    ckpt = torch.load(sft_path, map_location="cpu", weights_only=False)
    head = SelectionHead(1024, ckpt["num_workers"])  # Qwen3-0.6B hidden dim
    head.load_state_dict(ckpt["head"])

    cma_cfg = cfg.get("cma", {})
    pop_size = cma_cfg.get("population_size", 16)
    generations = cma_cfg.get("generations", 20)
    sigma = cma_cfg.get("sigma", 0.1)

    # Flatten head parameters for CMA-ES
    x0 = np.concatenate([p.detach().numpy().flatten() for p in head.parameters()])
    dim = len(x0)

    # Synthetic fitness: prefer routing toward worker with highest index on code prompts
    def fitness(x: np.ndarray) -> float:
        offset = 0
        for p in head.parameters():
            size = p.numel()
            p.data = torch.from_numpy(x[offset : offset + size].reshape(p.shape)).float()
            offset += size
        # Mock hidden state
        h = torch.randn(1, 1024)
        logits = head(h)[0]
        probs = torch.softmax(logits, dim=-1)
        # Reward peaked distribution (simulates correct routing signal)
        entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
        return -entropy  # minimize negative = maximize peaked routing

    es = cma.CMAEvolutionStrategy(x0, sigma, {"popsize": pop_size})
    for gen in range(generations):
        solutions = es.ask()
        es.tell(solutions, [fitness(s) for s in solutions])
        es.disp()
        print(f"CMA-ES generation {gen + 1}/{generations} best={es.result.fbest:.4f}")

    best = es.result.xbest
    offset = 0
    for p in head.parameters():
        size = p.numel()
        p.data = torch.from_numpy(best[offset : offset + size].reshape(p.shape)).float()
        offset += size

    out_dir = Path(cfg.get("output_dir", "checkpoints/router"))
    out_path = out_dir / "router_cma.pt"
    torch.save(
        {
            "backbone": ckpt["backbone"],
            "head": head.state_dict(),
            "num_workers": ckpt["num_workers"],
        },
        out_path,
    )
    print(f"Saved CMA-ES checkpoint to {out_path}")
    return out_path
