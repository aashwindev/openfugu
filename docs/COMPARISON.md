# Comparison

## OpenFugu vs closed Fugu

| | Closed Fugu | OpenFugu |
|---|-------------|----------|
| Source | API only | Full stack (Apache-2.0) |
| Weights | Not released | Train your own |
| Router | Proprietary coordinator | Qwen3-0.6B + ~10K head |
| Conductor | Proprietary | Qwen2.5-7B + GRPO pipeline |
| Hosting | `api.sakana.ai` | Self-hosted |
| Cost | Per-token + subscription | Free software + your API bills |
| Routing visibility | None | Full metadata per request |
| Worker pool | Fixed | You configure |
| EU hosting | Cloud-only | Run anywhere |

Use closed Fugu if you want zero ops and their managed pool.

Use OpenFugu if you want control, transparency, or your own models.

---

## OpenFugu vs Fugusashi

| | Fugusashi | OpenFugu |
|---|-----------|----------|
| Router | CMA-ES + similarity + cost | SFT + CMA-ES on hidden states |
| Conductor | Rule-based + GRPO | Workflow DSL + GRPO |
| Focus | Federated routing, explanations | Full Fugu stack replication |

---

## OpenFugu vs TinyRouter

| | TinyRouter | OpenFugu |
|---|------------|----------|
| Scope | Router only | Router + conductor + API |
| Roles | T/W/V | Worker-only (fast path) |
| Serving | Scripts | FastAPI server |

TinyRouter is a good reference for the CMA-ES training loop. OpenFugu adds the conductor layer and production API.

---

## Performance expectations

Closed Fugu benchmarks use a frontier worker pool. Your numbers depend on what you configure:

| Pool | Router gain | Conductor gain |
|------|-------------|----------------|
| 3 local OSS models | Modest | Moderate on hard tasks |
| Mid-tier API models | Noticeable | Good on coding |
| Frontier API models | Largest routing wins | Best multi-agent synergies |

Document your worker pool when reporting benchmarks.
