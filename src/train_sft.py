"""LoRA SFT on gold SQL, with the loss computed on the SQL output only.

  python -m src.train_sft --out results/sft_smoke --max-steps 20      # smoke
  python -m src.train_sft --out results/row2_sft                      # full, budget-limited

Uses the same prompt, the same training questions and the same LoRA
configuration as the GRPO arm, and stops on the same GPU-hour budget.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src import data
from src.train_utils import (CONFIGS, load_budget, load_config, lora_b_norm, make_budget_callback,
                             make_lora_config, package_versions, write_json)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIGS / "sft.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    import torch
    import yaml
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    cfg = load_config(args.config)
    base = load_config(CONFIGS / "lora.yaml")
    tcfg = dict(cfg["trainer"])
    for kv in args.set:
        k, v = kv.split("=", 1)
        tcfg[k] = yaml.safe_load(v)
    prompt_cfg = data.load_prompt_config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    budget_hours = None
    if args.max_steps:
        tcfg["max_steps"] = args.max_steps
    else:
        budget_hours = load_budget()["gpu_hours_per_run"]

    train = data.load_split("train")
    if args.limit:
        train = train[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(base["model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    end_ids = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]
    assert len(end_ids) == 1, end_ids

    features, n_skipped = [], 0
    for e in train:
        prompt_ids = tokenizer.apply_chat_template(
            data.build_messages(e, prompt_cfg), add_generation_prompt=True, tokenize=True)
        if not isinstance(prompt_ids, list):
            prompt_ids = prompt_ids["input_ids"]
        if len(prompt_ids) > cfg["max_prompt_tokens"]:
            n_skipped += 1
            continue
        target_ids = tokenizer(e["gold_sql"].strip(), add_special_tokens=False)["input_ids"] + end_ids
        features.append({
            "input_ids": prompt_ids + target_ids,
            "labels": [-100] * len(prompt_ids) + target_ids,   # loss on the SQL only
        })

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(features)

        def __getitem__(self, i):
            return features[i]

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
        }

    model = AutoModelForCausalLM.from_pretrained(base["model"], torch_dtype=torch.bfloat16)
    model = get_peft_model(model, make_lora_config())
    norm_before = lora_b_norm(model)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    targs = TrainingArguments(output_dir=str(out), seed=base["seed"], report_to="none",
                              save_total_limit=None, **tcfg)
    trainer = Trainer(model=model, args=targs, train_dataset=DS(), data_collator=collate,
                      callbacks=[make_budget_callback(out, budget_hours)])
    t0 = time.time()
    result = trainer.train()
    train_s = time.time() - t0
    norm_after = lora_b_norm(model)
    model.save_pretrained(str(out / "final"))

    steps = trainer.state.global_step
    write_json(out / "run_metadata.json", {
        "kind": "sft", "config_file": args.config, "trainer": tcfg, "lora": base,
        "n_train_questions": len(features), "n_skipped_too_long": n_skipped, "limit": args.limit,
        "budget_hours": budget_hours, "max_steps": args.max_steps,
        "optimizer_steps": steps, "train_seconds": round(train_s, 1),
        "seconds_per_step": round(train_s / max(1, steps), 3),
        "trainable_params": n_trainable,
        "lora_B_norm_before": norm_before, "lora_B_norm_after": norm_after,
        "lora_weights_changed": norm_after > norm_before,
        "train_loss": getattr(result, "training_loss", None),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
        "versions": package_versions(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    print(f"steps={steps} train_s={train_s:.0f} lora_B_norm {norm_before:.4f} -> {norm_after:.4f}")
    if not norm_after > norm_before:
        raise SystemExit("LoRA weights did not change")


if __name__ == "__main__":
    main()
