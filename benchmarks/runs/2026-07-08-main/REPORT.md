# OpenFugu Benchmark Report — July 8, 2026

**Run ID:** `2026-07-08-main`  
**Duration:** 8h 42m  
**Total spend:** $847.32 (unique worker API billing; see `worker_ledger.json`)  
**Config:** `benchmarks/config/eval_july2026.yaml`  
**Verify:** `python scripts/verify_benchmark_artifacts.py`

## Summary

We evaluated OpenFugu (`openfugu`, `openfugu-ultra`) against 10 frontier baselines on 6 suites — 435 tasks total, 3 seeds (42, 43, 44), 15,660 instance records. Worker pool: 10 API workers + 2 orchestrators = 12 eval systems.

**openfugu-ultra** leads on average (**88.4%**), ahead of Fable 5 (**88.1%**) and GPT-5.5 (**85.3%**). The fast **openfugu** router hits **83.7%** at 4.2s avg latency — 4.4× faster than ultra with 4.7 points behind ultra on average.

## Full leaderboard

| Rank | Model | MMLU | GPQA-D | LCB v6 | HumanEval+ | Terminal | Agentic | **Avg** |
|------|-------|------|--------|--------|------------|----------|---------|---------|
| 1 | **openfugu-ultra** | **94.0** | **92.0** | 88.8 | **92.0** | 80.0 | **83.3** | **88.4** |
| 2 | claude-fable-5 | 93.0 | 90.0 | 87.5 | 90.0 | **88.0** | 80.0 | 88.1 |
| 3 | gpt-5.5 | 93.5 | 86.0 | 88.8 | 86.0 | 84.0 | 73.3 | 85.3 |
| 4 | claude-opus-4.8 | 91.5 | 88.0 | 83.8 | 88.0 | 84.0 | 76.7 | 85.3 |
| 5 | openfugu | 91.5 | 88.0 | 86.2 | 88.0 | 72.0 | 76.7 | 83.7 |
| 6 | claude-sonnet-5 | 90.5 | 86.0 | 85.0 | 86.0 | 80.0 | 73.3 | 83.5 |
| 7 | gemini-3.1-pro | 92.5 | 90.0 | **91.2** | 84.0 | 72.0 | 70.0 | 83.3 |
| 8 | kimi-k2.7-code | 87.0 | 80.0 | 87.5 | 86.0 | 72.0 | 73.3 | 81.0 |
| 9 | glm-5.2 | 89.0 | 82.0 | 86.2 | 82.0 | 68.0 | 70.0 | 79.5 |
| 10 | deepseek-v4-pro | 88.5 | 84.0 | 87.5 | 80.0 | 64.0 | 66.7 | 78.5 |
| 11 | qwen3-32b-thinking | 85.0 | 78.0 | 82.5 | 76.0 | 64.0 | 64.4 | 75.0 |
| 12 | minimax-m3 | 83.0 | 74.0 | 80.0 | 74.0 | 68.0 | 67.8 | 74.5 |

Scores are pass@1 / accuracy (%), averaged over seeds. Each score equals `correct / total × 100` in `by_model/*.json` and aggregates from `instances/*.jsonl`.

## Results by suite

### MMLU (200 samples, professional subset)

| Model | Accuracy |
|-------|----------|
| openfugu-ultra | **94.0%** |
| gpt-5.5 | 93.5% |
| claude-fable-5 | 93.0% |
| gemini-3.1-pro | 92.5% |
| openfugu | 91.5% |

Ultra routes math-heavy items to GPT-5.5 and humanities to Gemini 3.1 Pro. Observed 12% of items used a 2-step debate topology.

### GPQA Diamond (50 samples)

| Model | Accuracy |
|-------|----------|
| openfugu-ultra | **92.0%** |
| gemini-3.1-pro | 90.0% |
| claude-fable-5 | 90.0% |
| openfugu | 88.0% |

Chemistry/biology questions predominantly routed to Gemini; physics computation to GPT-5.5.

### LiveCodeBench v6 (80 samples, Jan–Apr 2025)

| Model | pass@1 |
|-------|--------|
| gemini-3.1-pro | **91.2%** |
| gpt-5.5 | 88.8% |
| openfugu-ultra | 88.8% |
| openfugu | 86.2% |
| glm-5.2 | 86.2% |

Ultra uses planner (Opus 4.8) → implementer (GPT-5.5) chains on hard problems. 6 items used verifier step.

### HumanEval+ (50 samples)

| Model | pass@1 |
|-------|--------|
| openfugu-ultra | **92.0%** |
| claude-fable-5 | 90.0% |
| openfugu | 88.0% |
| claude-opus-4.8 | 88.0% |

### Terminal-Bench 2.1 (25 samples, Terminus-2 harness)

| Model | Resolve rate |
|-------|--------------|
| claude-fable-5 | **88.0%** |
| gpt-5.5 | 84.0% |
| claude-opus-4.8 | 84.0% |
| openfugu-ultra | 80.0% |
| openfugu | 72.0% |

Ultra alternates GPT-5.5 (build) and Opus 4.8 (debug) on 8/25 tasks — same pattern we saw probing the closed API.

### Agentic coding (30 custom tasks)

| Model | Success rate |
|-------|--------------|
| openfugu-ultra | **83.3%** |
| claude-fable-5 | 80.0% |
| openfugu | 76.7% |
| claude-opus-4.8 | 76.7% |
| kimi-k2.7-code | 73.3% |
| gpt-5.5 | 73.3% |

## Cost / latency

Per-system eval attribution (`summary.json` → `eval_cost_usd`). Unique worker API billing sums to $847.32 (`worker_ledger.json`).

| Model | Avg latency | Eval spend |
|-------|-------------|------------|
| openfugu | 4.2s | $26.32 |
| deepseek-v4-pro | 7.1s | $44.50 |
| claude-sonnet-5 | 6.8s | $42.62 |
| claude-opus-4.8 | 10.9s | $68.31 |
| gpt-5.5 | 11.8s | $73.95 |
| claude-fable-5 | 12.1s | $75.83 |
| openfugu-ultra | 18.4s | $115.33 |
| qwen3-32b-thinking | 22.5s | $141.01 |

Fast path is the cost winner for interactive workloads. Ultra costs more than any single model but beats them on composite score.

## Routing observations (openfugu-ultra)

- Simple factual: 1 step (68% of MMLU)
- Coding: 2–3 step chain (74% of LCB)
- Hard science: tree with Gemini + GPT leaves, Opus aggregator (GPQA)
- Terminal: build/debug alternation (GPT → Opus)

Full per-instance logs: `instances/` (6 suite JSONL files + `all_results.jsonl`, seeds 42/43/44)

## Reproduce

```bash
python scripts/run_benchmarks.py --config benchmarks/config/eval_july2026.yaml
python scripts/verify_benchmark_artifacts.py
```
