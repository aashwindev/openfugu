"""GRPO training for the Conductor planner."""

from __future__ import annotations

from pathlib import Path

import yaml

from openfugu.training.datasets import build_training_mix
from openfugu.training.rewards import conductor_reward, grade_exact_match


def train_grpo(config_path: str = "config/conductor_train.yaml", *, smoke: bool = False) -> Path:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    grpo_cfg = cfg.get("smoke" if smoke else "grpo", cfg.get("grpo", {}))
    if smoke:
        grpo_cfg = {**cfg.get("grpo", {}), **cfg.get("smoke", {})}

    examples = build_training_mix(cfg)
    max_samples = grpo_cfg.get("max_samples", len(examples))
    examples = examples[:max_samples]

    iterations = grpo_cfg.get("iterations", 50 if smoke else 200)
    group_size = grpo_cfg.get("group_size", 4 if smoke else 8)
    base_model = cfg.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
    out_dir = Path(cfg.get("output_dir", "checkpoints/conductor"))
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as e:
        raise ImportError("Install train extras: pip install openfugu[train]") from e

    print(f"Loading {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lr = grpo_cfg.get("learning_rate", 5e-7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    system_prompt = (
        "You are a Conductor. Output model_id, subtasks, access_list as Python lists "
        "after your reasoning."
    )

    for iteration in range(iterations):
        batch_rewards: list[float] = []
        batch_loss = 0.0
        count = 0

        for ex in examples[: grpo_cfg.get("batch_size", 8)]:
            rewards: list[float] = []
            log_probs: list[torch.Tensor] = []

            for _ in range(group_size):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ex.question},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=grpo_cfg.get("temperature", 1.0),
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                gen_ids = outputs.sequences[0, inputs["input_ids"].shape[1] :]
                output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                # Mock final answer from last subtask line
                final_answer = output_text[-200:]
                r = conductor_reward(output_text, final_answer, ex.answer, grade_fn=grade_exact_match)
                rewards.append(r)

                with torch.no_grad():
                    lp = model(inputs["input_ids"], labels=inputs["input_ids"]).loss
                log_probs.append(-lp)

            # GRPO grouped advantage
            import numpy as np

            r_arr = np.array(rewards)
            mean_r = r_arr.mean()
            std_r = r_arr.std() + 1e-8
            advantages = (r_arr - mean_r) / std_r

            for adv, lp in zip(advantages, log_probs):
                loss = -float(adv) * lp
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                count += 1

            batch_rewards.extend(rewards)

        avg_reward = sum(batch_rewards) / max(len(batch_rewards), 1)
        print(
            f"GRPO iter {iteration + 1}/{iterations} "
            f"avg_reward={avg_reward:.3f} loss={batch_loss / max(count, 1):.4f}"
        )

        if (iteration + 1) % cfg.get("checkpoint_every", 10) == 0:
            ckpt_path = out_dir / f"conductor_step_{iteration + 1}"
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    final_path = out_dir / "conductor_final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Training complete. Final checkpoint: {final_path}")
    return final_path
