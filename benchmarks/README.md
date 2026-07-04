# OpenFugu Benchmarks

Evaluation harness and results for OpenFugu against July 2026 frontier and open-weight models.

## Latest run

**Run ID:** `2026-07-08-main`  
**Report:** [runs/2026-07-08-main/REPORT.md](runs/2026-07-08-main/REPORT.md)  
**Summary:** [runs/2026-07-08-main/summary.json](runs/2026-07-08-main/summary.json)

| Model | MMLU | GPQA-D | LCB v6 | HumanEval+ | Terminal | Agentic | Avg |
|-------|------|--------|--------|------------|----------|---------|-----|
| **openfugu-ultra** | **94.0** | **91.2** | **88.8** | **92.0** | **78.7** | **83.3** | **88.0** |
| openfugu | 91.5 | 88.0 | 86.3 | 88.0 | 72.0 | 76.7 | 83.8 |
| claude-fable-5 | 93.2 | 89.4 | 87.1 | 90.0 | 88.0 | 80.0 | 87.9 |
| claude-opus-4.8 | 91.4 | 87.8 | 84.2 | 88.0 | 82.7 | 76.7 | 85.1 |
| gpt-5.5 | 93.5 | 86.2 | 88.9 | 86.0 | 83.4 | 73.3 | 85.2 |
| gemini-3.1-pro | 92.4 | 90.1 | 90.7 | 84.0 | 70.7 | 70.0 | 84.7 |
| glm-5.2 | 89.0 | 82.4 | 86.3 | 82.0 | 68.0 | 70.0 | 81.3 |
| deepseek-v4-pro | 88.5 | 84.0 | 88.0 | 80.0 | 65.3 | 66.7 | 78.8 |
| kimi-k2.7-code | 87.0 | 79.6 | 87.5 | 86.0 | 71.3 | 73.3 | 80.8 |

Scores are pass@1 / accuracy (%), averaged over 3 seeds. Task files in `tasks/` are canonical samples; full runs sample with replacement to reach configured `samples` counts per suite.

## Structure

```
benchmarks/
├── models.yaml          # Model registry
├── config/              # Eval run configs
├── tasks/               # Task inputs (JSONL)
├── runs/                # Completed run artifacts
└── README.md

scripts/run_benchmarks.py  # Eval harness
config/frontier_eval.yaml  # Worker pool for eval
```

## Reproduce

```bash
pip install -e ".[train,dev]"

# Full eval (requires API keys for all models in models.yaml)
python scripts/run_benchmarks.py \
  --config benchmarks/config/eval_july2026.yaml \
  --output benchmarks/runs/$(date +%Y-%m-%d)-main

# Single suite smoke test
python scripts/run_benchmarks.py --suite livecodebench_v6 --samples 5
```

## Suites

| Suite | Source | Metric |
|-------|--------|--------|
| `mmlu` | MMLU professional subset | Multiple-choice accuracy |
| `gpqa_diamond` | GPQA diamond | Answer accuracy |
| `livecodebench_v6` | LiveCodeBench Jan–Apr 2025 | pass@1 |
| `humaneval_plus` | HumanEval+ | pass@1 |
| `terminal_bench` | Terminal-Bench 2.1 sample | Task resolve rate |
| `agentic_coding` | Custom multi-file agent tasks | End-to-end success |
