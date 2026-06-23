#!/usr/bin/env python3
"""Train router CMA-ES stage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openfugu.training.router_cma import train_cma

if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "config/router_train.yaml"
    train_cma(config)
