#!/usr/bin/env python3
"""Train conductor with GRPO."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openfugu.training.conductor_grpo import train_grpo

if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    config = "config/conductor_train.yaml"
    for arg in sys.argv[1:]:
        if arg.endswith(".yaml"):
            config = arg
    train_grpo(config, smoke=smoke)
