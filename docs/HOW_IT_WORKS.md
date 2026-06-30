# How OpenFugu Works

Notes from reverse-engineering the Fugu API and rebuilding the stack from scratch.

## What Fugu actually is

Fugu is not a single model. It's an orchestration layer:

1. You send one request to one endpoint
2. Behind the scenes it picks which frontier models to call, how they collaborate, and how to merge outputs
3. You get back a single answer

Two modes matter:

- **Fugu** — fast. One worker per turn. Latency close to a direct API call.
- **Fugu-Ultra** — slow. Multiple workers in a planned workflow. Better on hard tasks.

We rebuilt both paths.

## The fast router

### What we observed

The fast path doesn't generate routing text. It decides from an internal representation of the prompt — logits, not chain-of-thought. That's why it's fast: no autoregressive decode step just to pick a model.

### What we built

- **Backbone:** Qwen3-0.6B, frozen. Read hidden state at an early token position.
- **Selection head:** small linear layer, ~10K params. `hidden → L logits` for L workers.
- **No roles.** Older multi-agent routers assign Thinker/Worker/Verifier. Fugu's fast path skips that — always dispatches as a plain worker. Fewer decisions, lower latency.

Files: `router/backbone.py`, `router/head.py`, `router/inference.py`

### Training the router

Two stages, matching what the closed system's behavior implies:

**Stage 1 — SFT with soft targets**

Run every worker on each question several times. Score them. Don't hard-label "worker 2 wins" — use a softmax distribution over scores so the router learns gradations (worker 2 is best, but worker 1 is close).

```
p(j) = exp(reward_j / τ) / Σ exp(reward_j' / τ)
loss = KL(target_distribution || router_distribution)
```

**Stage 2 — CMA-ES on real tasks**

Single-step labels miss multi-turn coding behavior. Evolutionary search on the router head against end-to-end task completion (pass/fail) refines routing for agentic workloads.

Files: `training/router_sft.py`, `training/router_cma.py`, `scripts/collect_router_labels.py`

## The ultra conductor

### What we observed

Ultra mode returns answers that clearly came from multiple models working in sequence — planner steps, implementer steps, occasional verifier passes. The API doesn't show the plan, but latency and token usage patterns match multi-step orchestration.

### Workflow format

By probing ultra responses and matching against known multi-agent patterns, the conductor output format is three aligned lists:

```python
model_id = [2, 0]
subtasks = ["Develop an efficient algorithm...", "Implement in Python"]
access_list = [[], ["all"]]
```

- `model_id` — which worker runs each step
- `subtasks` — natural-language instructions per step (custom prompt engineering per worker)
- `access_list` — which prior step outputs this worker can see

This supports chains, trees, and debate patterns (parallel attempts → aggregator step).

Files: `conductor/parser.py`, `conductor/prompts.py`, `conductor/executor.py`

### Memory model

Two rules that matter for correctness:

**Intra-workflow isolation.** Each agent's tool-call history stays private. Agents only see each other's work through `access_list` — step outputs, not raw tool transcripts. Without this, the first agent to touch the environment steers everyone else (orchestration collapse).

**Inter-workflow memory.** Across user turns in a session, tool artifacts persist so agents don't re-discover the same files/state.

Files: `conductor/memory.py`, `harness/tool_loop.py`

### Training the conductor

GRPO — generate multiple workflow plans per question, score them as a group, update toward the better ones.

Reward:
- `0` — malformed workflow (can't parse the three lists)
- `1` — workflow runs, final answer correct
- `0.5` — workflow runs, wrong answer

Base model: Qwen2.5-7B-Instruct. Up to 5 steps per workflow.

Files: `training/conductor_grpo.py`, `training/rewards.py`

## API surface

Fugu exposes OpenAI-compatible endpoints. We match:

| Endpoint | Behavior |
|----------|----------|
| `POST /v1/chat/completions` | Main path |
| `POST /v1/responses` | Routes to ultra/conductor |
| `GET /v1/models` | Lists `openfugu`, `openfugu-ultra`, workers |

Plus `openfugu` metadata on every response — routing decision, workflow trace, cost. The closed API gives you none of this.

## What we couldn't copy

- Trained coordinator weights (not public)
- Their exact frontier worker pool
- Production serving infra

What you get instead: the full architecture, training scripts, and a running server. Bring your own workers and checkpoints.
