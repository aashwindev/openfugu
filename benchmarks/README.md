# OpenFugu Benchmarks

Evaluation harness and results for OpenFugu against frontier and open-weight models.

## Latest run

**Run ID:** `2026-07-08-main`  
**Report:** [runs/2026-07-08-main/REPORT.md](runs/2026-07-08-main/REPORT.md)  
**Summary:** [runs/2026-07-08-main/summary.json](runs/2026-07-08-main/summary.json)  
**Verify:** `python scripts/verify_benchmark_artifacts.py`

| Model | MMLU | GPQA-D | LCB v6 | HumanEval+ | Terminal | Agentic | Avg |
|-------|------|--------|--------|------------|----------|---------|-----|
| **openfugu-ultra** | **94.0** | **92.0** | 88.8 | **92.0** | 80.0 | **83.3** | **88.4** |
| claude-fable-5 | 93.0 | 90.0 | 87.5 | 90.0 | **88.0** | 80.0 | 88.1 |
| gpt-5.5 | 93.5 | 86.0 | 88.8 | 86.0 | 84.0 | 73.3 | 85.3 |
| claude-opus-4.8 | 91.5 | 88.0 | 83.8 | 88.0 | 84.0 | 76.7 | 85.3 |
| openfugu | 91.5 | 88.0 | 86.2 | 88.0 | 72.0 | 76.7 | 83.7 |
| claude-sonnet-5 | 90.5 | 86.0 | 85.0 | 86.0 | 80.0 | 73.3 | 83.5 |
| gemini-3.1-pro | 92.5 | 90.0 | **91.2** | 84.0 | 72.0 | 70.0 | 83.3 |
| kimi-k2.7-code | 87.0 | 80.0 | 87.5 | 86.0 | 72.0 | 73.3 | 81.0 |
| glm-5.2 | 89.0 | 82.0 | 86.2 | 82.0 | 68.0 | 70.0 | 79.5 |
| deepseek-v4-pro | 88.5 | 84.0 | 87.5 | 80.0 | 64.0 | 66.7 | 78.5 |
| qwen3-32b-thinking | 85.0 | 78.0 | 82.5 | 76.0 | 64.0 | 64.4 | 75.0 |
| minimax-m3 | 83.0 | 74.0 | 80.0 | 74.0 | 68.0 | 67.8 | 74.5 |

Scores are pass@1 / accuracy (%). Each suite has a full task pool in `tasks/` matching `config/eval_july2026.yaml` sample counts. Every model runs every task under seeds 42, 43, 44. Instance logs: `runs/2026-07-08-main/instances/` (15,660 records).

## Structure

```
benchmarks/
├── models.yaml
├── config/eval_july2026.yaml
├── tasks/                    # 435 unique tasks total across suites
├── runs/2026-07-08-main/
│   ├── summary.json
│   ├── manifest.json
│   ├── worker_ledger.json
│   ├── by_model/             # 12 models
│   └── instances/            # 6 suite JSONL files + all_results.jsonl
└── README.md

scripts/run_benchmarks.py
scripts/verify_benchmark_artifacts.py
config/frontier_eval.yaml
```

## Reproduce

```bash
pip install -e ".[train,dev]"

python scripts/run_benchmarks.py \
  --config benchmarks/config/eval_july2026.yaml \
  --output benchmarks/runs/$(date +%Y-%m-%d)-main

python scripts/verify_benchmark_artifacts.py
```

## Suites

| Suite | Tasks in pool | Metric |
|-------|---------------|--------|
| `mmlu` | 200 | Multiple-choice accuracy |
| `gpqa_diamond` | 50 | Answer accuracy |
| `livecodebench_v6` | 80 | pass@1 |
| `humaneval_plus` | 50 | pass@1 |
| `terminal_bench` | 25 | Task resolve rate |
| `agentic_coding` | 30 | End-to-end success |
