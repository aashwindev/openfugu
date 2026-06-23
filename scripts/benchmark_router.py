#!/usr/bin/env python3
"""Benchmark router on held-out examples."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openfugu.config import load_config
from openfugu.router.inference import FuguRouter
from openfugu.training.datasets import _fallback_math_examples
from openfugu.workers.pool import WorkerPool


def main() -> None:
    config = load_config("config/default.yaml")
    pool = WorkerPool(config.workers)
    router = FuguRouter(
        pool,
        checkpoint=config.router.checkpoint,
        use_heuristic_fallback=True,
    )

    examples = _fallback_math_examples()
    correct = 0
    for ex in examples:
        decision = router.route(ex.question)
        print(f"Q: {ex.question[:50]}... -> {decision.worker_name} ({decision.strategy})")
        correct += 1  # routing benchmark only

    print(f"Routed {correct}/{len(examples)} examples")


if __name__ == "__main__":
    main()
