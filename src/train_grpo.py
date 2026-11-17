"""GRPO post-training with an execution-match reward (TRL GRPOTrainer).

  # timing / smoke test: fixed number of optimizer steps, no budget file needed
  python -m src.train_grpo --config configs/grpo.yaml --out results/grpo_timing --max-steps 20

  # full run: stops when the GPU-hour budget in configs/budget.yaml is used up
  python -m src.train_grpo --config configs/grpo.yaml --out results/row3_grpo

Outputs under --out:
  reward_log.jsonl   every reward call: SQL, executed, error type, correct, reward
  groups.jsonl       per reward batch: share of all-correct / all-wrong / mixed groups
  time_log.jsonl     elapsed training time at each log and checkpoint
  run_metadata.json  config, versions, timing, LoRA-weights-changed check
  checkpoint-*/      LoRA adapters
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from src import data
from src.rewards import Scorer
from src.train_utils import (CONFIGS, load_budget, load_config, lora_b_norm, make_budget_callback,
                             make_lora_config, package_versions, write_json)


def make_reward_fn(scorer: Scorer, by_id: dict, out: Path, num_generations: int):
    groups_path = out / "groups.jsonl"
    state = {"call": 0}

    def execution_reward(completions, ex_id, trainer_state=None, **kwargs):
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        examples = [by_id[i] for i in ex_id]
        step = getattr(trainer_state, "global_step", None)
        recs = scorer.score(examples, [data.extract_sql(t) for t in texts],
                            log_extra={"step": step, "call": state["call"]})
        # Rollouts of one prompt share an ex_id; group on it rather than on position.
        by_group = defaultdict(list)
        for r in recs:
            by_group[r["id"]].append(r["correct"])
        n = len(by_group)
        kinds = [("all_correct" if all(v) else "all_wrong" if not any(v) else "mixed")
                 for v in by_group.values()]
        row = {
            "step": step, "call": state["call"], "n_groups": n, "n_rollouts": len(recs),
            "group_sizes_ok": all(len(v) == num_generations for v in by_group.values()),
            "frac_all_correct": kinds.count("all_correct") / n,
            "frac_all_wrong": kinds.count("all_wrong") / n,
            "frac_mixed": kinds.count("mixed") / n,
            "acc": sum(r["correct"] for r in recs) / len(recs),
            "executable": sum(r["executed"] for r in recs) / len(recs),
            "mean_reward": sum(r["reward"] for r in recs) / len(recs),
        }
        with open(groups_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        state["call"] += 1
        return [float(r["reward"]) for r in recs]

    return execution_reward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIGS / "grpo.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, help="smoke/timing run: ignore the budget, run N optimizer steps")
    ap.add_argument("--test-budget-hours", type=float,
                    help="smoke test of the budget/checkpoint logic with a tiny budget (not a reported run)")
    ap.add_argument("--limit", type=int, help="debugging: use the first N training questions")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override trainer config entries, e.g. learning_rate=1e-5")
    args = ap.parse_args()

    import torch
    import yaml
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    cfg = load_config(args.config)
    base = load_config(CONFIGS / "lora.yaml")
    tcfg = dict(cfg["trainer"])
    for kv in args.set:
        k, v = kv.split("=", 1)
        tcfg[k] = yaml.safe_load(v)
    prompt_cfg = data.load_prompt_config()
    if tcfg["max_completion_length"] != prompt_cfg["decoding"]["max_new_tokens"]:
        raise ValueError("max_completion_length must equal the frozen max_new_tokens")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    budget_hours = None
    if args.max_steps:
        tcfg["max_steps"] = args.max_steps
    else:
        budget_hours = args.test_budget_hours or load_budget()["gpu_hours_per_run"]
        tcfg["save_strategy"] = "no"   # checkpoints follow the GPU-time schedule instead

    train = data.load_split("train")
    if args.limit:
        train = train[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(base["model"])
    kept = []
    for e in train:  # same length filter as the SFT arm
        ids = tokenizer.apply_chat_template(
            data.build_messages(e, prompt_cfg), add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        if len(ids) <= base["max_prompt_tokens"]:
            kept.append(e)
    n_skipped = len(train) - len(kept)
    train = kept
    by_id = {e["id"]: e for e in train}
    scorer = Scorer(mode=cfg["reward"], timeout_s=prompt_cfg["execution"]["timeout_s"],
                    log_path=out / "reward_log.jsonl")
    scorer.ensure_gold(train)
    dataset = Dataset.from_list(
        [{"prompt": data.build_messages(e, prompt_cfg), "ex_id": e["id"]} for e in train]
    ).shuffle(seed=base["seed"])

    model = AutoModelForCausalLM.from_pretrained(base["model"], dtype=torch.bfloat16)

    config = GRPOConfig(output_dir=str(out), seed=base["seed"], report_to="none",
                        save_total_limit=None, **tcfg)
    if getattr(config, "loss_type", None) != tcfg["loss_type"]:
        raise RuntimeError(f"TRL did not accept loss_type={tcfg['loss_type']!r}")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=make_reward_fn(scorer, by_id, out, tcfg["num_generations"]),
        args=config,
        train_dataset=dataset,
        peft_config=make_lora_config(),
        callbacks=[make_budget_callback(out, budget_hours)],
    )
    norm_before = lora_b_norm(trainer.model)
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    t0 = time.time()
    result = trainer.train()
    train_s = time.time() - t0
    scorer.close()
    norm_after = lora_b_norm(trainer.model)
    trainer.save_model(str(out / "final"))

    steps = trainer.state.global_step
    prompts_per_step = (tcfg["per_device_train_batch_size"] * tcfg["gradient_accumulation_steps"]
                        // tcfg["num_generations"])
    write_json(out / "run_metadata.json", {
        "kind": "grpo", "config_file": args.config, "reward": cfg["reward"], "trainer": tcfg,
        "lora": base, "n_train_questions": len(train), "n_skipped_too_long": n_skipped, "limit": args.limit,
        "budget_hours": budget_hours, "max_steps": args.max_steps,
        "optimizer_steps": steps, "train_seconds": round(train_s, 1),
        "seconds_per_step": round(train_s / max(1, steps), 2),
        "prompts_per_step": prompts_per_step,
        "trainable_params": n_trainable,
        "lora_B_norm_before": norm_before, "lora_B_norm_after": norm_after,
        "lora_weights_changed": norm_after > norm_before,
        "train_loss": getattr(result, "training_loss", None),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
        "versions": package_versions(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    print(f"steps={steps} train_s={train_s:.0f} s/step={train_s / max(1, steps):.1f} "
          f"lora_B_norm {norm_before:.4f} -> {norm_after:.4f}")
    if not norm_after > norm_before:
        raise SystemExit("LoRA weights did not change")


if __name__ == "__main__":
    main()
