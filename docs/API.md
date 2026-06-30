# API Reference

OpenFugu exposes an OpenAI-compatible HTTP API.

Base URL: `http://localhost:8080/v1`

## Authentication

Optional. Set `OPENFUGU_API_KEY` env or `server.api_key` in config.

```bash
curl -H "Authorization: Bearer $OPENFUGU_API_KEY" ...
```

If no key configured, auth is disabled.

---

## Models

### `GET /v1/models`

Lists orchestration models and configured workers.

```json
{
  "object": "list",
  "data": [
    {"id": "openfugu", "object": "model", "owned_by": "openfugu"},
    {"id": "openfugu-ultra", "object": "model", "owned_by": "openfugu"},
    {"id": "auto", "object": "model", "owned_by": "openfugu"},
    {"id": "gpt-4o-mini", "object": "model", "owned_by": "worker"}
  ]
}
```

### Model IDs

| Model | Path | Latency | Use case |
|-------|------|---------|----------|
| `openfugu` | Router (Fugu) | Low | Everyday chat, coding |
| `openfugu-ultra` | Conductor (Fugu-Ultra) | Higher | Hard multi-step tasks |
| `auto` | Heuristic dispatch | Varies | General use |
| `<worker-name>` | Direct to worker | Single call | Bypass orchestration |

---

## Chat Completions

### `POST /v1/chat/completions`

Standard OpenAI format:

```json
{
  "model": "openfugu",
  "messages": [
    {"role": "user", "content": "Explain quicksort"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024
}
```

### Response (with OpenFugu extensions)

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "openfugu",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 150,
    "total_tokens": 162
  },
  "openfugu": {
    "mode": "router",
    "routing_decision": {
      "mode": "router",
      "worker_id": 0,
      "worker_name": "gpt-4o-mini",
      "confidence": 0.71,
      "strategy": "heuristic",
      "latency_ms": 0.5,
      "logits": []
    },
    "workers_used": ["gpt-4o-mini"],
    "total_cost_usd": 0.000023
  }
}
```

### Ultra mode response

```json
{
  "openfugu": {
    "mode": "ultra",
    "workflow": {
      "steps": 2,
      "topology": "chain",
      "workers_used": ["claude-haiku", "gpt-4o-mini"],
      "step_results": [
        {"step": 0, "worker": "claude-haiku", "subtask": "Analyze and plan..."},
        {"step": 1, "worker": "gpt-4o-mini", "subtask": "Implement..."}
      ]
    },
    "workers_used": ["claude-haiku", "gpt-4o-mini"],
    "total_cost_usd": 0.0012
  }
}
```

---

## Responses endpoint

### `POST /v1/responses`

OpenAI-compatible alias routing to conductor:

```json
{
  "model": "openfugu-ultra",
  "input": [{"role": "user", "content": "..."}]
}
```

---

## Stats

### `GET /v1/stats`

```json
{
  "total_requests": 42,
  "total_cost_usd": 0.15,
  "by_worker": {
    "gpt-4o-mini": {"requests": 30, "tokens": 45000, "cost_usd": 0.01}
  }
}
```

---

## Health

### `GET /health`

```json
{"status": "ok", "workers": "3"}
```

---

## Python client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="optional")

resp = client.chat.completions.create(
    model="openfugu-ultra",
    messages=[{"role": "user", "content": "Build a REST API in Python"}],
)
print(resp.choices[0].message.content)

# Access OpenFugu metadata (if using raw response)
# resp.model_extra.get("openfugu")
```

---

## Not yet implemented

- Streaming (`stream: true`) — returns 501
- Tool calling through orchestration layer — partial (harness exists, API passthrough pending)
