# Architecture

OpenFugu is a two-tier LLM orchestration server with an OpenAI-compatible API.

## System overview

```
Client (OpenAI SDK)
       │
       ▼
┌──────────────────────────────────────┐
│  FastAPI  /v1/chat/completions       │
│  Models: openfugu | openfugu-ultra   │
└──────────────┬───────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌──────────────────┐
│ Router      │  │ Conductor        │
│ 1 worker/   │  │ up to 5 steps    │
│ turn        │  │ workflow DSL     │
└──────┬──────┘  └────────┬─────────┘
       │                  │
       └────────┬─────────┘
                ▼
        ┌───────────────┐
        │ Worker Pool   │
        │ (LiteLLM)     │
        └───────────────┘
```

## Components

### API (`openfugu.api`)

- `/v1/chat/completions`, `/v1/responses`, `/v1/models`
- Dispatch: `openfugu` → router, `openfugu-ultra` → conductor, `auto` → complexity heuristic
- `openfugu` metadata on every response

### Router (`openfugu.router`) — fast path

| File | Role |
|------|------|
| `backbone.py` | Qwen3-0.6B, early-token hidden state |
| `head.py` | Linear selection head (L logits) |
| `sv_tune.py` | Singular-value scale adapters |
| `inference.py` | Route + dispatch |
| `multistep.py` | Per-turn re-routing in agentic loops |

Routing uses hidden-state logits, not generated text.

### Conductor (`openfugu.conductor`) — ultra path

| File | Role |
|------|------|
| `parser.py` | Extract `model_id`, `subtasks`, `access_list` |
| `workflow.py` | Step/result types |
| `memory.py` | Per-agent isolation + session artifacts |
| `executor.py` | Sequential worker calls |
| `planner.py` | Workflow generation (trained or heuristic) |

### Workers (`openfugu.workers`)

YAML config → LiteLLM. Cost and token tracking per request.

### Training (`openfugu.training`)

| Stage | Module |
|-------|--------|
| Router SFT | `router_sft.py` |
| Router CMA-ES | `router_cma.py` |
| Conductor GRPO | `conductor_grpo.py` |

## Ultra request flow

1. `POST /v1/chat/completions` with `model: openfugu-ultra`
2. Planner generates workflow (three Python lists)
3. Parser validates and builds step objects
4. Executor runs each step with access-list context
5. Final step output → assistant message + workflow trace

## Failure modes

| Failure | Behavior |
|---------|----------|
| Malformed workflow | Retry, then single-worker fallback |
| Worker API error | Propagate to client |
| No checkpoint | Heuristic routing/planning |
