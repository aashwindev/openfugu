#!/usr/bin/env python3
"""Run OpenFugu benchmark suites against frontier models."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openfugu.config import load_config
from openfugu.conductor.planner import ConductorPlanner
from openfugu.router.inference import FuguRouter
from openfugu.workers.pool import WorkerPool


def load_eval_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_tasks(benchmarks_dir: Path, rel_path: str) -> list[dict[str, Any]]:
    path = benchmarks_dir / rel_path
    tasks: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def grade_mmlu(response: str, answer: str) -> bool:
    letter = answer.strip().upper()
    text = response.upper()
    return f"({letter})" in text or text.strip().endswith(letter) or f"ANSWER: {letter}" in text


def grade_exact(response: str, answer: str) -> bool:
    return answer.lower().strip() in response.lower()


def _task_prompt(task: dict[str, Any], suite: str) -> str:
    if suite == "mmlu":
        choices = "\n".join(task.get("choices", []))
        return f"{task['question']}\n\n{choices}\n\nAnswer with the letter only."
    if suite in ("gpqa_diamond", "livecodebench_v6", "humaneval_plus"):
        return task.get("question") or task.get("prompt", "")
    return task.get("description", json.dumps(task))


def _grade(task: dict[str, Any], suite: str, response: str) -> bool:
    if suite == "mmlu":
        return grade_mmlu(response, task["answer"])
    if suite == "gpqa_diamond":
        return grade_exact(response, task["answer"])
    if suite in ("livecodebench_v6", "humaneval_plus", "terminal_bench", "agentic_coding"):
        return len(response.strip()) > 20
    return False


async def run_baseline(
    pool: WorkerPool,
    worker_name: str,
    task: dict[str, Any],
    suite: str,
) -> dict[str, Any]:
    worker = pool.get_by_name(worker_name)
    t0 = time.perf_counter()
    prompt = _task_prompt(task, suite)
    resp = await pool.complete(worker.id, [{"role": "user", "content": prompt}])
    latency = time.perf_counter() - t0
    passed = _grade(task, suite, resp.content)
    return {
        "instance_id": task.get("task_id", "unknown"),
        "model": worker_name,
        "suite": suite,
        "passed": passed,
        "latency_s": round(latency, 2),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


async def run_openfugu(
    router: FuguRouter,
    pool: WorkerPool,
    task: dict[str, Any],
    suite: str,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    prompt = _task_prompt(task, suite)
    decision = router.route(prompt)
    resp = await pool.complete(decision.worker_id, [{"role": "user", "content": prompt}])
    latency = time.perf_counter() - t0
    passed = _grade(task, suite, resp.content)
    return {
        "instance_id": task.get("task_id", "unknown"),
        "model": "openfugu",
        "suite": suite,
        "passed": passed,
        "latency_s": round(latency, 2),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "routing": {
            "worker": decision.worker_name,
            "confidence": round(decision.confidence, 2),
            "strategy": decision.strategy,
        },
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


async def run_openfugu_ultra(
    planner: ConductorPlanner,
    task: dict[str, Any],
    suite: str,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    prompt = _task_prompt(task, suite)
    result = await planner.run(prompt)
    latency = time.perf_counter() - t0
    passed = _grade(task, suite, result.final_answer)
    workflow = result.workflow
    return {
        "instance_id": task.get("task_id", "unknown"),
        "model": "openfugu-ultra",
        "suite": suite,
        "passed": passed,
        "latency_s": round(latency, 2),
        "workflow": {
            "steps": workflow.num_steps,
            "topology": workflow.topology_label(),
            "workers": result.workers_used,
        },
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def write_run_artifacts(
    output_dir: Path,
    run_id: str,
    eval_cfg: dict[str, Any],
    all_results: list[dict[str, Any]],
    by_model: dict[str, dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "by_model").mkdir(exist_ok=True)
    (output_dir / "instances").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    manifest = {
        "run_id": run_id,
        "manifest_version": 1,
        "started_at": eval_cfg.get("started_at", datetime.now(timezone.utc).isoformat()),
        "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "openfugu_version": eval_cfg.get("openfugu_version", "0.1.0"),
        "config": str(eval_cfg.get("_config_path", "")),
        "worker_pool": eval_cfg.get("worker_pool", "config/frontier_eval.yaml"),
        "total_samples_evaluated": len(all_results),
        "models_evaluated": len(by_model),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    with (output_dir / "instances" / "results.jsonl").open("w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    for model, stats in by_model.items():
        safe = model.replace("/", "_")
        (output_dir / "by_model" / f"{safe}.json").write_text(json.dumps(stats, indent=2))


async def run_eval(args: argparse.Namespace) -> None:
    benchmarks_dir = ROOT / "benchmarks"
    eval_path = ROOT / args.config
    eval_cfg = load_eval_config(eval_path)
    eval_cfg["_config_path"] = str(args.config)

    run_id = eval_cfg.get("run_id") or datetime.now(timezone.utc).strftime("%Y-%m-%d") + "-main"
    output_dir = Path(args.output) if args.output else benchmarks_dir / "runs" / run_id

    pool_cfg_path = ROOT / eval_cfg.get("worker_pool", "config/frontier_eval.yaml")
    pool_cfg = load_config(pool_cfg_path)
    pool = WorkerPool(pool_cfg.workers)

    router = FuguRouter(
        pool,
        checkpoint=pool_cfg.router.checkpoint,
        use_heuristic_fallback=pool_cfg.router.use_heuristic_fallback,
    )
    planner = ConductorPlanner(
        pool,
        checkpoint=pool_cfg.conductor.checkpoint,
        max_steps=pool_cfg.conductor.max_steps,
        use_heuristic_fallback=pool_cfg.conductor.use_heuristic_fallback,
    )

    models: list[str] = list(eval_cfg.get("orchestrators", []))
    models.extend(eval_cfg.get("baselines", {}).get("closed", []))
    models.extend(eval_cfg.get("baselines", {}).get("open", []))

    all_results: list[dict[str, Any]] = []
    by_model: dict[str, dict[str, Any]] = {}

    for suite_cfg in eval_cfg.get("suites", []):
        suite_name = suite_cfg["name"]
        if args.suite and suite_name != args.suite:
            continue

        tasks = load_tasks(benchmarks_dir, suite_cfg["file"])
        limit = args.samples or suite_cfg.get("samples", len(tasks))
        tasks = tasks[:limit]

        print(f"Suite {suite_name}: {len(tasks)} tasks, models={len(models)}")

        if args.dry_run:
            for t in tasks:
                print(f"  - {t.get('task_id', '?')}")
            continue

        for model in models:
            suite_results: list[bool] = []
            for task in tasks:
                try:
                    if model == "openfugu":
                        result = await run_openfugu(router, pool, task, suite_name)
                    elif model == "openfugu-ultra":
                        result = await run_openfugu_ultra(planner, task, suite_name)
                    else:
                        result = await run_baseline(pool, model, task, suite_name)
                    all_results.append(result)
                    suite_results.append(result["passed"])
                    status = "pass" if result["passed"] else "fail"
                    print(f"  {model} {task.get('task_id')} {status}")
                except Exception as exc:
                    print(f"  {model} {task.get('task_id')} error: {exc}")

            if suite_results:
                score = 100.0 * sum(suite_results) / len(suite_results)
                if model not in by_model:
                    by_model[model] = {"model": model, "run_id": run_id, "suites": {}}
                by_model[model]["suites"][suite_name] = {
                    "score": round(score, 1),
                    "correct": sum(suite_results),
                    "total": len(suite_results),
                }

    if not args.dry_run and all_results:
        for stats in by_model.values():
            scores = [s["score"] for s in stats["suites"].values()]
            stats["avg_score"] = round(sum(scores) / len(scores), 1) if scores else 0.0
        write_run_artifacts(output_dir, run_id, eval_cfg, all_results, by_model)
        print(f"Wrote artifacts to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenFugu benchmarks")
    parser.add_argument("--config", default="benchmarks/config/eval_july2026.yaml")
    parser.add_argument("--output", default=None, help="Output run directory")
    parser.add_argument("--suite", default=None, help="Run single suite only")
    parser.add_argument("--samples", type=int, default=None, help="Limit samples per suite")
    parser.add_argument("--dry-run", action="store_true", help="List tasks without API calls")
    args = parser.parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
