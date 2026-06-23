#!/usr/bin/env python3
"""Collect per-worker rewards and soft targets for router SFT."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from openfugu.config import AppConfig, load_config
from openfugu.training.datasets import build_training_mix
from openfugu.training.rewards import grade_exact_match, router_soft_targets
from openfugu.workers.pool import WorkerPool


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/router_train.yaml")
    parser.add_argument("--output", default="data/router_labels.jsonl")
    parser.add_argument("--app-config", default="config/default.yaml")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--n-repeats", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)

    app_cfg = load_config(args.app_config)
    pool = WorkerPool(app_cfg.workers)
    examples = build_training_mix(train_cfg)[: args.max_samples]
    temperature = train_cfg.get("temperature", 1.0)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as out_f:
        for ex in examples:
            rewards: list[float] = []
            for worker in pool.workers:
                worker_rewards: list[float] = []
                for _ in range(args.n_repeats):
                    try:
                        response = await pool.complete(
                            worker.id,
                            [{"role": "user", "content": ex.question}],
                        )
                        r = 1.0 if grade_exact_match(response.content, ex.answer) else 0.0
                        worker_rewards.append(r)
                    except Exception:
                        worker_rewards.append(0.0)
                mean_r = sum(worker_rewards) / max(len(worker_rewards), 1)
                rewards.append(mean_r)

            soft = router_soft_targets(rewards, temperature=temperature)
            record = {
                "question": ex.question,
                "answer": ex.answer,
                "domain": ex.domain,
                "worker_rewards": rewards,
                "soft_targets": soft,
            }
            out_f.write(json.dumps(record) + "\n")
            print(f"Collected: {ex.question[:60]}... -> {soft}")

    print(f"Wrote {len(examples)} labels to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
