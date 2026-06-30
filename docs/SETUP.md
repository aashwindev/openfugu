# Setup Guide

Get OpenFugu running in under 15 minutes.

## Prerequisites

- Python 3.11+
- API keys for your worker models (OpenAI, Anthropic, or local Ollama)
- Optional: NVIDIA GPU for training

## Quick install

```bash
cd openfugu
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For training:
```bash
pip install -e ".[train,dev]"
```

## Configure workers

Edit `config/default.yaml`:

```yaml
workers:
  - id: 0
    name: "gpt-4o-mini"
    provider: "openai"
    model: "gpt-4o-mini"
    capabilities: ["chat", "code", "reasoning"]
    cost_per_input_token: 0.00000015
    cost_per_output_token: 0.0000006

  - id: 1
    name: "llama-local"
    provider: "ollama"
    model: "llama3.2:3b"
    api_base: "http://localhost:11434"
    capabilities: ["chat", "factual"]
    cost_per_input_token: 0.0
    cost_per_output_token: 0.0
    prefer_local: true
```

Set API keys:
```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
# Optional server auth
export OPENFUGU_API_KEY=your-local-key
```

## Start the server

```bash
openfugu serve
# → http://0.0.0.0:8080
```

Or:
```bash
openfugu serve --config config/default.yaml --port 8080
```

## Test it

```bash
# Fast path (router)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openfugu",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'

# Ultra path (conductor)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openfugu-ultra",
    "messages": [{"role": "user", "content": "Design and implement a binary search in Python"}]
  }'
```

## Migration from Fugu

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-key",           # OPENFUGU_API_KEY or leave unset
    base_url="http://localhost:8080/v1",
)

# Was: model="fugu"
response = client.chat.completions.create(
    model="openfugu",
    messages=[{"role": "user", "content": "Hello"}],
)

# Was: model="fugu-ultra-20260615"
response = client.chat.completions.create(
    model="openfugu-ultra",
    messages=[{"role": "user", "content": "Hard multi-step task..."}],
)
```

## $50 budget path

| Item | Cost | Purpose |
|------|------|---------|
| Ollama (local) | $0 | Free worker via `llama3.2:3b` |
| OpenRouter credit | ~$20 | 2-3 API workers for label collection |
| Lambda H100 4hr | ~$20-30 | Router SFT training |
| **Total** | **~$50** | Self-hosted orchestration |

Steps:
1. Run Ollama: `ollama pull llama3.2:3b`
2. Configure 1 local + 2 OpenRouter models in `config/default.yaml`
3. Collect labels: `python scripts/collect_router_labels.py --max-samples 100`
4. Train SFT: `openfugu train-router --stage sft`
5. Serve: `openfugu serve`

## Production path

- 3+ frontier/API workers in pool
- Train router SFT + CMA-ES on your traffic
- Train conductor GRPO with 2× H100 (see [TRAINING.md](TRAINING.md))
- Set `OPENFUGU_API_KEY` for auth
- Put behind reverse proxy (nginx/caddy) with TLS

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `litellm` auth error | Check provider API keys in env |
| Ollama connection refused | `ollama serve` running on :11434 |
| Router loads slowly | First run downloads Qwen3-0.6B (~1.2GB); use heuristic fallback |
| No workers configured | Add at least one worker to `config/default.yaml` |
