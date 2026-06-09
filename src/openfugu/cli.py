"""CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="openfugu", description="OpenFugu orchestration server")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Start OpenAI-compatible API server")
    serve.add_argument("--config", "-c", default=None, help="Path to config YAML")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("version", help="Print version")

    train_router = sub.add_parser("train-router", help="Train router (SFT + CMA-ES)")
    train_router.add_argument("--stage", choices=["sft", "cma", "all"], default="all")
    train_router.add_argument("--config", default="config/router_train.yaml")

    train_conductor = sub.add_parser("train-conductor", help="Train conductor (GRPO)")
    train_conductor.add_argument("--smoke", action="store_true", help="Smoke test config")
    train_conductor.add_argument("--config", default="config/conductor_train.yaml")

    collect = sub.add_parser("collect-labels", help="Collect router training labels")
    collect.add_argument("--config", default="config/router_train.yaml")
    collect.add_argument("--output", default="data/router_labels.jsonl")

    args = parser.parse_args()

    if args.command == "version" or args.command is None:
        from openfugu import __version__

        print(f"openfugu {__version__}")
        if args.command is None:
            parser.print_help()
        return

    if args.command == "serve":
        _serve(args)
    elif args.command == "train-router":
        _train_router(args)
    elif args.command == "train-conductor":
        _train_conductor(args)
    elif args.command == "collect-labels":
        _collect_labels(args)


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from openfugu.api.app import create_app
    from openfugu.config import AppConfig, find_default_config, load_config

    config_path = Path(args.config) if args.config else find_default_config()
    config = load_config(config_path)
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port

    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port)


def _train_router(args: argparse.Namespace) -> None:
    from openfugu.training.router_sft import train_sft
    from openfugu.training.router_cma import train_cma

    if args.stage in ("sft", "all"):
        train_sft(args.config)
    if args.stage in ("cma", "all"):
        train_cma(args.config)


def _train_conductor(args: argparse.Namespace) -> None:
    from openfugu.training.conductor_grpo import train_grpo

    train_grpo(args.config, smoke=args.smoke)


def _collect_labels(args: argparse.Namespace) -> None:
    import asyncio
    from pathlib import Path
    import sys

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    from collect_router_labels import main as collect_main

    sys.argv = [
        "collect_router_labels",
        "--config",
        args.config,
        "--output",
        args.output,
    ]
    asyncio.run(collect_main())


if __name__ == "__main__":
    main()
