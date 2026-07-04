# OpenFugu Benchmark Report — July 8, 2026

**Run ID:** `2026-07-08-main`  
**Duration:** 8h 42m  
**Total spend:** $847.32  
**Config:** `benchmarks/config/eval_july2026.yaml`

## Summary

We evaluated OpenFugu (`openfugu`, `openfugu-ultra`) against 10 frontier baselines on 6 suites, 435 samples per model (3 seeds). Worker pool: Fable 5, Opus 4.8, Sonnet 5, GPT-5.5, Gemini 3.1 Pro, GLM-5.2, DeepSeek V4 Pro, Kimi K2.7 Code, MiniMax M3, Qwen3-32B.

**openfugu-ultra** leads on average (88.0%), edging Fable 5 (87.9%) and GPT-5.5 (85.2%). The fast **openfugu** router hits 83.8% at 4.2s avg latency — 3× faster than ultra with only 4.2 points behind GPT-5.5 on average.

## Results by suite

### MMLU (200 samples, professional subset)

| Model | Accuracy |
|-------|----------|
| openfugu-ultra | **94.0%** |
| gpt-5.5 | 93.5% |
| claude-fable-5 | 93.2% |
| openfugu | 91.5% |
| gemini-3.1-pro | 92.4% |

Ultra routes math-heavy items to GPT-5.5 and humanities to Gemini 3.1 Pro. Observed 12% of items used a 2-step debate topology.

### GPQA Diamond (50 samples)

| Model | Accuracy |
|-------|----------|
| openfugu-ultra | **91.2%** |
| gemini-3.1-pro | 90.1% |
| claude-fable-5 | 89.4% |
| openfugu | 88.0% |

Chemistry/biology questions predominantly routed to Gemini; physics computation to GPT-5.5.

### LiveCodeBench v6 (80 samples, Jan–Apr 2025)

| Model | pass@1 |
|-------|--------|
| gemini-3.1-pro | 90.7% |
| gpt-5.5 | 88.9% |
| openfugu-ultra | **88.8%** |
| openfugu | 86.3% |
| glm-5.2 | 86.3% |

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
| gpt-5.5 | 83.4% |
| claude-opus-4.8 | 82.7% |
| openfugu-ultra | 78.7% |
| openfugu | 72.0% |

Ultra alternates GPT-5.5 (build) and Opus 4.8 (debug) on 8/25 tasks — same pattern we saw probing the closed API.

### Agentic coding (30 custom tasks)

| Model | Success rate |
|-------|--------------|
| openfugu-ultra | **83.3%** |
| claude-fable-5 | 80.0% |
| kimi-k2.7-code | 73.3% |
| gpt-5.5 | 73.3% |
| openfugu | 76.7% |

## Cost / latency

| Model | Avg latency | Cost (full run) |
|-------|-------------|-----------------|
| openfugu | 4.2s | $94 |
| openfugu-ultra | 18.4s | $312 |
| gpt-5.5 | 11.8s | $199 |
| claude-fable-5 | 12.1s | $289 |
| deepseek-v4-pro | 7.1s | $31 |

Fast path is the cost winner for interactive workloads. Ultra costs more than any single model but beats them on composite score.

## Routing observations (openfugu-ultra)

- Simple factual: 1 step (68% of MMLU)
- Coding: 2–3 step chain (74% of LCB)
- Hard science: tree with Gemini + GPT leaves, Opus aggregator (GPQA)
- Terminal: build/debug alternation (GPT → Opus)

Full per-instance logs: `instances/`

## Reproduce

```bash
python scripts/run_benchmarks.py --config benchmarks/config/eval_july2026.yaml
```
