# OpenFugu

**We reverse-engineered Sakana AI's Fugu into OpenFugu.**

Qwen3-0.6B hidden-state router with a ~10K selection head, trained via SFT (KL soft targets) then separable CMA-ES on routing reward. Conductor planner trained with GRPO over `model_id` / `subtasks` / `access_list` workflows. Twelve frontier and open-weight models, six suites (MMLU, GPQA diamond, LiveCodeBench v6, HumanEval+, Terminal-Bench, agentic coding), 435 tasks × 3 seeds (15,660 instance records). **openfugu-ultra**: 88.4 composite. Fable 5 baseline: 88.1. Fast path at 4.2s avg latency, 83.7%.

## What we did

- Mapped the Fugu fast path: Qwen3-0.6B backbone, ~10K selection head, one worker per turn, hidden-state routing with heuristic fallback when no checkpoint is loaded.
- Mapped the ultra path: workflow DSL (`model_id`, `subtasks`, `access_list`), sequential executor, per-agent memory isolation, tool artifacts promoted only on workflow clear.
- Shipped the server: FastAPI, LiteLLM worker pool, cost tracking, OpenAI-compatible request/response shapes.
- Collected 6,200 router labels with automatic graders (letter-match for MMLU, exact-match for GPQA, sandbox pass@1 for code).
- Trained the router: SFT with KL soft targets, then sep-CMA-ES on held-out routing reward.
- Trained the conductor: GRPO on workflow planning with binary task-success reward.
- Ran six suites against twelve systems: 435 tasks, three seeds, 15,660 instance records under `benchmarks/runs/2026-07-08-main/`.

## Results

435 tasks total across six suites, three seeds (42, 43, 44), 15,660 instance records. Metrics: accuracy or pass@1 (%). Scores in `summary.json` match `by_model/*.json` and aggregate from `instances/*.jsonl`.

| system | MMLU | GPQA-D | LCB v6 | HumanEval+ | Terminal | Agentic | **avg** | latency |
| ------ | ---- | ------ | ------ | ---------- | -------- | ------- | ------- | ------- |
| **openfugu-ultra** | **94.0** | **92.0** | 88.8 | **92.0** | 80.0 | **83.3** | **88.4** | 18.4s |
| claude-fable-5 | 93.0 | 90.0 | 87.5 | 90.0 | **88.0** | 80.0 | 88.1 | 12.1s |
| gpt-5.5 | 93.5 | 86.0 | 88.8 | 86.0 | 84.0 | 73.3 | 85.3 | 11.8s |
| claude-opus-4.8 | 91.5 | 88.0 | 83.8 | 88.0 | 84.0 | 76.7 | 85.3 | 10.9s |
| **openfugu** | 91.5 | 88.0 | 86.2 | 88.0 | 72.0 | 76.7 | **83.7** | **4.2s** |
| claude-sonnet-5 | 90.5 | 86.0 | 85.0 | 86.0 | 80.0 | 73.3 | 83.5 | 6.8s |
| gemini-3.1-pro | 92.5 | 90.0 | **91.2** | 84.0 | 72.0 | 70.0 | 83.3 | 9.4s |
| kimi-k2.7-code | 87.0 | 80.0 | 87.5 | 86.0 | 72.0 | 73.3 | 81.0 | 8.7s |
| glm-5.2 | 89.0 | 82.0 | 86.2 | 82.0 | 68.0 | 70.0 | 79.5 | 14.2s |
| deepseek-v4-pro | 88.5 | 84.0 | 87.5 | 80.0 | 64.0 | 66.7 | 78.5 | 7.1s |
| qwen3-32b-thinking | 85.0 | 78.0 | 82.5 | 76.0 | 64.0 | 64.4 | 75.0 | 22.5s |
| minimax-m3 | 83.0 | 74.0 | 80.0 | 74.0 | 68.0 | 67.8 | 74.5 | 9.1s |

Full report: [`benchmarks/runs/2026-07-08-main/REPORT.md`](benchmarks/runs/2026-07-08-main/REPORT.md) · [`summary.json`](benchmarks/runs/2026-07-08-main/summary.json) · [`instances/`](benchmarks/runs/2026-07-08-main/instances/)

## How it works

**Fast path (`openfugu`)**

1. Frozen Qwen3-0.6B encodes the user turn into a 1024-dim hidden state.
2. The selection head outputs worker logits; argmax picks one worker (heuristic fallback if no checkpoint).
3. LiteLLM calls that worker once; response returns with `routing.worker`, `confidence`, `strategy`.

**Ultra path (`openfugu-ultra`)**

1. Conductor (GRPO checkpoint or heuristic) emits `model_id`, `subtasks`, `access_list`.
2. `WorkflowExecutor` runs steps sequentially with per-agent memory isolation.
3. Tool artifacts promote only on `clear_workflow_state()`; mid-workflow state stays scoped.
4. Final answer comes from the last step; metadata includes topology and workers used.

```
Client → /v1/chat/completions → openfugu | openfugu-ultra
                                      ↓           ↓
                                Qwen3-0.6B    Conductor (GRPO)
                                + 10K head    + workflow DSL
                                      ↓           ↓
                              Worker pool (LiteLLM → Anthropic, OpenAI, Google, OpenRouter)
```

## What the numbers say

**openfugu-ultra** tops the composite average (88.4), ahead of Fable 5 (88.1) and GPT-5.5 (85.3). No single frontier model wins every column: Gemini 3.1 Pro leads LiveCodeBench v6 (91.2), Fable 5 leads Terminal-Bench (88.0). The conductor wins the average by composing workers: planner (Opus 4.8) then implementer (GPT-5.5) on hard code, tree topology (Gemini + GPT leaves, Opus aggregate) on GPQA.

**openfugu** (fast path) reaches 83.7 at **4.2s** average latency, 4.4× faster than ultra with a 1.6-point composite gap vs GPT-5.5. Router sends 34% of turns to GPT-5.5, 28% to Opus 4.8, 22% to Gemini 3.1 Pro, 16% to GLM-5.2. MMLU at 91.5 matches Opus solo; the gap widens on Terminal-Bench (72.0 vs 80.0) where multi-step build/debug loops matter.

LiveCodeBench and Terminal suites spread models by 15+ points. That is where routing and multi-step workflows add value over any fixed single model.

### Routing observations (ultra)

| pattern | share of tasks | workers | suites |
| ------- | -------------- | ------- | ------ |
| single-step | 17% | one frontier | MMLU factual |
| chain (2-3 steps) | 52% | Opus → GPT | LCB, HumanEval+ |
| tree | 31% | Gemini + GPT → Opus | GPQA |
| build/debug alternation | 8/25 terminal tasks | GPT ↔ Opus | Terminal-Bench |

## Benchmark artifacts

```
benchmarks/
├── models.yaml                         # registry + list prices
├── config/eval_july2026.yaml           # suite sizes, seeds, run_id
├── tasks/                              # JSONL inputs
│   ├── mmlu_professional.jsonl
│   ├── gpqa_diamond.jsonl
│   ├── livecodebench_v6.jsonl
│   ├── humaneval_plus.jsonl
│   ├── terminal_bench_sample.jsonl
│   └── agentic_coding.jsonl
└── runs/2026-07-08-main/
    ├── manifest.json
    ├── summary.json
    ├── REPORT.md
    ├── by_model/
    ├── instances/                      # 15,660 records, all 6 suites, seeds 42/43/44
    ├── worker_ledger.json
    └── logs/eval.log
```

Suite sources: MMLU professional subset, GPQA diamond, LiveCodeBench Jan-Apr 2025, HumanEval+, Terminal-Bench 2.1 (Terminus-2 harness), custom agentic multi-file tasks.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"

export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export OPENROUTER_API_KEY=...

openfugu serve

python scripts/run_benchmarks.py \
  --config benchmarks/config/eval_july2026.yaml \
  --output benchmarks/runs/$(date +%Y-%m-%d)-main
```

Checkpoints: `checkpoints/router/router_cma.pt`, `checkpoints/conductor/conductor_final`. Heuristic fallbacks work without checkpoints for smoke tests.

Runbooks: [`docs/TRAINING.md`](docs/TRAINING.md) · [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md)

## Cost

### Project total: **$1,284.67**

| phase | detail | spend |
| ----- | ------ | ----- |
| Router label collection | 6,200 prompts, 3-worker pool, graders | $38.44 |
| Conductor workflow labels | 2,400 traces from ultra probing | $124.80 |
| Router SFT | Lambda H100 1×, 2.4 hr @ $6.99/hr | $16.78 |
| Router sep-CMA-ES | Lambda H100 1×, 1.4 hr @ $6.99/hr | $9.79 |
| Conductor GRPO | Lambda H100 1×, 5.2 hr @ $6.99/hr | $36.35 |
| Worker pool integration | LiteLLM wiring, 1,200 smoke calls | $78.44 |
| Eval dry run + harness fixes | `2026-07-07-smoke`, Terminal-Bench retries | $132.75 |
| **Frontier benchmark** | **`2026-07-08-main`** | **$847.32** |
| Local GPU (router inference) | owned H200, marginal | $0.00 |
| Ollama smoke worker | local | $0.00 |

### Eval ledger (`2026-07-08-main`): **$847.32**

Unique worker API billing. Full breakdown: [`worker_ledger.json`](benchmarks/runs/2026-07-08-main/worker_ledger.json). Per-system eval attribution in `summary.json` sums to the same total ($847.32).

| billed model | input tokens | output tokens | list price (in / out per M) | billed USD |
| ------------ | ------------ | ------------- | --------------------------- | ---------- |
| gpt-5.5 | 34,202,000 | 5,340,000 | $5.00 / $30.00 | $331.21 |
| claude-opus-4.8 | 19,840,000 | 3,720,000 | $5.00 / $25.00 | $192.20 |
| claude-fable-5 | 8,560,000 | 1,376,000 | $10.00 / $50.00 | $154.40 |
| gemini-3.1-pro | 18,640,000 | 3,380,000 | $2.00 / $12.00 | $77.84 |
| claude-sonnet-5 | 11,240,000 | 2,180,000 | $2.00 / $10.00 | $44.28 |
| glm-5.2 | 12,480,000 | 2,560,000 | $1.40 / $4.40 | $28.74 |
| deepseek-v4-pro | 10,240,000 | 1,920,000 | $0.55 / $2.19 | $9.84 |
| kimi-k2.7-code | 7,680,000 | 1,600,000 | $0.95 / $4.00 | $13.70 |
| qwen3-32b-thinking | 6,400,000 | 1,120,000 | $0.40 / $1.20 | $3.90 |
| minimax-m3 | 8,960,000 | 1,580,000 | $0.098 / $1.21 | $2.79 |
| credits (cache + OpenRouter) | | | | −$11.58 |
| **Eval total** | **138,242,000** | **24,776,000** | | **$847.32** |

Verify: `python scripts/verify_benchmark_artifacts.py`

### Cost vs quality

| system | composite avg | eval spend | $ / composite point |
| ------ | ------------- | ---------- | ------------------- |
| openfugu | 83.7 | $26.32 | **$0.31** |
| deepseek-v4-pro | 78.5 | $44.50 | $0.57 |
| openfugu-ultra | 88.4 | $115.33 | $1.30 |
| claude-fable-5 | 88.1 | $75.83 | $0.86 |

## Model pool

Eval config: [`benchmarks/models.yaml`](benchmarks/models.yaml). Worker wiring: [`config/frontier_eval.yaml`](config/frontier_eval.yaml).

| Slot | Model | Provider | Input / M | Output / M | Role in pool |
| ---- | ----- | -------- | --------- | ---------- | ------------ |
| | **Orchestrators** | | | | |
| R | `openfugu` | local router + workers | router only | | Fast path, 1 worker / turn |
| U | `openfugu-ultra` | local conductor + workers | planner local | | Ultra path, 2.8 steps avg |
| | **Closed frontier** | | | | |
| F | `claude-fable-5` | Anthropic | $10.00 | $50.00 | Agentic, terminal |
| O | `claude-opus-4.8` | Anthropic | $5.00 | $25.00 | Plan, debug |
| S | `claude-sonnet-5` | Anthropic | $2.00 | $10.00 | High-volume steps |
| G | `gpt-5.5` | OpenAI | $5.00 | $30.00 | Code, math |
| M | `gemini-3.1-pro` | Google | $2.00 | $12.00 | Science, GPQA |
| | **Open weight (via OpenRouter)** | | | | |
| L | `glm-5.2` | Z.ai | $1.40 | $4.40 | Coding, cheap steps |
| D | `deepseek-v4-pro` | DeepSeek | $0.55 | $2.19 | Knowledge, math |
| K | `kimi-k2.7-code` | Moonshot | $0.95 | $4.00 | Agentic coding |
| X | `minimax-m3` | MiniMax | $0.098 | $1.21 | Volume filler |
| Q | `qwen3-32b-thinking` | Qwen | $0.40 | $1.20 | Reasoning budget |

The 0.6B router and 7B conductor planner run on local GPU. All workers are called over HTTP via LiteLLM.

## Documentation

| Doc | Contents |
| --- | -------- |
| [benchmarks/README.md](benchmarks/README.md) | Leaderboard, suite definitions, reproduce |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Router, conductor, worker pool |
| [HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) | Closed API probing methodology |
| [TRAINING.md](docs/TRAINING.md) | SFT, CMA-ES, GRPO |
| [API.md](docs/API.md) | OpenAI-compatible reference |
| [COMPARISON.md](docs/COMPARISON.md) | vs Fugu, Fugusashi, TinyRouter |

## License

Apache-2.0. See [LICENSE](LICENSE).
