#!/usr/bin/env python3
"""Train router SFT stage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openfugu.training.router_sft import train_sft

if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "config/router_train.yaml"
    train_sft(config)
