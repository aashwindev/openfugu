#!/usr/bin/env python3
"""Verify benchmark artifact internal consistency."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "benchmarks" / "runs" / "2026-07-08-main"
TASKS = ROOT / "benchmarks" / "tasks"
CONFIG = ROOT / "benchmarks" / "config" / "eval_july2026.yaml"

SUITES = {
    "mmlu": ("mmlu_professional.jsonl", 200),
    "gpqa_diamond": ("gpqa_diamond.jsonl", 50),
    "livecodebench_v6": ("livecodebench_v6.jsonl", 80),
    "humaneval_plus": ("humaneval_plus.jsonl", 50),
    "terminal_bench": ("terminal_bench_sample.jsonl", 25),
    "agentic_coding": ("agentic_coding.jsonl", 30),
}

SEEDS = {42, 43, 44}
errors: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def main() -> int:
    # task pool sizes
    for suite, (fname, expected) in SUITES.items():
        n = sum(1 for _ in open(TASKS / fname) if _.strip())
        if n != expected:
            err(f"task pool {suite}: {n} lines, expected {expected}")

    # instance files exist
    inst = RUN / "instances"
    for suite in SUITES:
        p = inst / f"{suite}.jsonl"
        if not p.exists():
            err(f"missing instances/{suite}.jsonl")

    # aggregate instances
    by_model_suite: dict[str, dict[str, list[bool]]] = {}
    seed_set: set[int] = set()
    for suite in SUITES:
        p = inst / f"{suite}.jsonl"
        with p.open() as f:
            for line in f:
                row = json.loads(line)
                seed_set.add(row["seed"])
                by_model_suite.setdefault(row["model"], {}).setdefault(suite, []).append(row["passed"])

    if seed_set != SEEDS:
        err(f"seeds in instances {sorted(seed_set)}, expected {sorted(SEEDS)}")

    expected_per_suite = {s: SUITES[s][1] * 12 * 3 for s in SUITES}
    for suite in SUITES:
        count = sum(len(by_model_suite[m].get(suite, [])) for m in by_model_suite)
        if count != expected_per_suite[suite]:
            err(f"instance count {suite}: {count}, expected {expected_per_suite[suite]}")

    # by_model arithmetic
    for path in sorted((RUN / "by_model").glob("*.json")):
        d = json.loads(path.read_text())
        model = d["model"]
        for suite, s in d["suites"].items():
            calc = round(100.0 * s["correct"] / s["total"], 1)
            if calc != s["score"]:
                err(f"{model} {suite}: score {s['score']} != {s['correct']}/{s['total']} = {calc}")
            inst_pass = sum(by_model_suite.get(model, {}).get(suite, []))
            if inst_pass != s["correct"]:
                err(f"{model} {suite}: by_model correct {s['correct']} != instances {inst_pass}")

    # leaderboard sort
    summary = json.loads((RUN / "summary.json").read_text())
    lb = summary["leaderboard"]
    for i, row in enumerate(lb, 1):
        if row["rank"] != i:
            err(f"leaderboard rank mismatch at {i}: {row['rank']}")
    scores = [r["avg_score"] for r in lb]
    if scores != sorted(scores, reverse=True):
        err("leaderboard not sorted by avg_score descending")

    # cost sum
    cost_sum = round(sum(r["eval_cost_usd"] for r in lb), 2)
    worker = summary.get("worker_billing_usd")
    if cost_sum != worker:
        err(f"eval_cost sum {cost_sum} != worker_billing {worker}")

    manifest = json.loads((RUN / "manifest.json").read_text())
    if manifest.get("unique_api_spend_usd") != worker:
        err("manifest unique_api_spend != summary worker_billing")

    wl = json.loads((RUN / "worker_ledger.json").read_text())
    if wl.get("check") != worker:
        err(f"worker_ledger check {wl.get('check')} != {worker}")

    if errors:
        print("FAIL")
        for e in errors:
            print(" -", e)
        return 1
    print("OK: all consistency checks passed")
    print(f"  instance records: {manifest['total_instance_records']}")
    print(f"  worker billing: ${worker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
