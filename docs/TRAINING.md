# Training

Train your own router and conductor on your worker pool.

## Overview

| Model | Stages |
|-------|--------|
| Router | Label collection → SFT → CMA-ES |
| Conductor | GRPO |

```bash
pip install -e ".[train]"
```

---

## Router

### 1. Collect labels

```bash
python scripts/collect_router_labels.py \
  --config config/router_train.yaml \
  --output data/router_labels.jsonl \
  --max-samples 200 \
  --n-repeats 3
```

Each line: question, per-worker rewards, soft target distribution.

### 2. SFT

```bash
openfugu train-router --stage sft
```

KL divergence between router softmax and soft targets from worker scores.

Config: `config/router_train.yaml` — backbone, temperature, epochs, learning rate.

Output: `checkpoints/router/router_sft.pt`

### 3. CMA-ES

```bash
openfugu train-router --stage cma
# or
openfugu train-router --stage all
```

Evolutionary search on router head weights against end-to-end task rewards.

Output: `checkpoints/router/router_cma.pt`

Point server at checkpoint:
```yaml
router:
  checkpoint: "checkpoints/router/router_cma.pt"
  use_heuristic_fallback: false
```

---

## Conductor (GRPO)

### Smoke test

```bash
openfugu train-conductor --smoke
```

50 iterations, 100 samples. Verifies the pipeline runs.

### Full run

```bash
openfugu train-conductor
```

Config: `config/conductor_train.yaml`

| Param | Default |
|-------|---------|
| base_model | Qwen2.5-7B-Instruct |
| iterations | 200 |
| group_size | 8 |
| max_steps | 5 |

Output: `checkpoints/conductor/conductor_final/`

---

## Checkpoints

```
checkpoints/
├── router/
│   ├── router_sft.pt
│   └── router_cma.pt
└── conductor/
    └── conductor_final/
```

---

## Cost estimates

| Stage | API | GPU |
|-------|-----|-----|
| Labels (1K questions) | $20–40 | — |
| Router SFT | — | ~4hr A100 |
| Router CMA-ES | $50–100 | ~8–24hr H100 |
| Conductor GRPO full | $200+ | 2× H100 |
