"""Batched generation (vLLM) + sandboxed execution + scoring.

Writes, under --out:
  raw.jsonl          one line per question: prompt-independent inputs, raw model
                     output, extracted SQL, execution outcome, correctness
  metrics.json       execution accuracy, executable rate, mean output tokens, ...
  run_metadata.json  model, adapter, split, prompt config hash, versions, timing

Examples:
  # G0 harness self-check (no GPU): score the gold SQL as the prediction
  python -m src.evaluate --split train --gold-as-pred --out results/g0_train
  python -m src.evaluate --split val   --gold-as-pred --out results/g0_val

  # Row 1 on val
  python -m src.evaluate --split val --model Qwen/Qwen2.5-Coder-1.5B-Instruct --out results/row1_zeroshot/val

Test splits (test_a, test_b) are refused without --allow-test, which is passed
only in Stage 4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

from src import data
from src.rewards import Scorer


_LLM = {}


def generate_vllm(messages_list: list[list[dict]], model: str, adapter: str | None,
                  max_new_tokens: int, temperature: float, max_model_len: int) -> list[dict]:
    """Generate one completion per conversation. Imported lazily: vLLM is Linux/GPU only.

    The engine is created once per process and reused, so many LoRA adapters
    can be evaluated without reloading the base model.
    """
    from vllm import LLM, SamplingParams

    if "llm" not in _LLM:
        _LLM["llm"] = LLM(model=model, max_model_len=max_model_len, seed=0,
                          enable_lora=True, max_lora_rank=64, max_loras=1)
        _LLM["model"], _LLM["n"] = model, 0
    assert _LLM["model"] == model
    llm = _LLM["llm"]
    lora_request = None
    if adapter:
        from vllm.lora.request import LoRARequest
        _LLM["n"] += 1
        lora_request = LoRARequest(f"adapter{_LLM['n']}", _LLM["n"], adapter)
    params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    outputs = llm.chat(messages_list, params, lora_request=lora_request)
    return [
        {"text": o.outputs[0].text, "n_tokens": len(o.outputs[0].token_ids),
         "finish_reason": o.outputs[0].finish_reason}
        for o in outputs
    ]


def compute_metrics(records: list[dict]) -> dict:
    n = len(records)
    gold_failed = [r["id"] for r in records if not r["gold_ok"]]
    scored = [r for r in records if r["gold_ok"]]
    toks = [r["n_tokens"] for r in records if r.get("n_tokens") is not None]
    return {
        "n": n,
        # Main metric: questions whose gold SQL fails count as wrong.
        "exec_acc": sum(r["correct"] for r in records) / n,
        "executable_rate": sum(r["executed"] for r in records) / n,
        "mean_output_tokens": sum(toks) / len(toks) if toks else None,
        "n_truncated": sum(r.get("finish_reason") == "length" for r in records),
        "error_types": dict(Counter(r["error_type"] for r in records if not r["executed"])),
        # G0 view: accuracy over questions whose gold SQL executes.
        "n_gold_failed": len(gold_failed),
        "gold_failed_ids": gold_failed,
        "exec_acc_excl_gold_failed": (sum(r["correct"] for r in scored) / len(scored)) if scored else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test_a", "test_b"])
    ap.add_argument("--out", help="output directory (single evaluation)")
    ap.add_argument("--model")
    ap.add_argument("--adapter", help="path to a LoRA adapter directory")
    ap.add_argument("--checkpoints-of", help="evaluate every checkpoint-*/ of this training run "
                    "in one process; writes <run>/eval_<split>/<checkpoint>/")
    ap.add_argument("--gold-as-pred", action="store_true", help="G0: use the gold SQL as the prediction")
    ap.add_argument("--allow-test", action="store_true", help="Stage 4 only")
    ap.add_argument("--limit", type=int, help="debugging: evaluate the first N questions")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
    if not args.gold_as_pred and not args.model:
        ap.error("--model is required unless --gold-as-pred is given")
    if args.checkpoints_of:
        run = Path(args.checkpoints_of)
        ckpts = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        for c in ckpts:
            out = run / f"eval_{args.split}" / c.name
            if (out / "metrics.json").exists():
                continue
            args.adapter, args.out = str(c), str(out)
            print(f"=== {c}")
            run_one(args)
        return
    if not args.out:
        ap.error("--out is required")
    run_one(args)


def run_one(args) -> None:
    cfg = data.load_prompt_config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    examples = data.load_split(args.split, allow_test=args.allow_test)
    if args.limit:
        examples = examples[: args.limit]

    scorer = Scorer(mode="binary", timeout_s=cfg["execution"]["timeout_s"], workers=args.workers)
    gold_stats = scorer.ensure_gold(examples)

    t0 = time.time()
    if args.gold_as_pred:
        gens = [{"text": e["gold_sql"], "n_tokens": None, "finish_reason": None} for e in examples]
    else:
        messages = [data.build_messages(e, cfg) for e in examples]
        gens = generate_vllm(
            messages, args.model, args.adapter,
            cfg["decoding"]["max_new_tokens"], cfg["decoding"]["temperature"], args.max_model_len,
        )
    gen_s = time.time() - t0

    t0 = time.time()
    pred_sqls = [data.extract_sql(g["text"]) for g in gens]
    scored = scorer.score(examples, pred_sqls)
    scorer.close()
    exec_s = time.time() - t0

    records = []
    with open(out / "raw.jsonl", "w", encoding="utf-8") as f:
        for e, g, s in zip(examples, gens, scored):
            rec = {
                "id": e["id"], "dataset": e["dataset"], "db_id": e["db_id"],
                "question": e["question"], "evidence": e["evidence"], "gold_sql": e["gold_sql"],
                "raw_output": g["text"], "n_tokens": g["n_tokens"], "finish_reason": g["finish_reason"],
                **{k: s[k] for k in ("pred_sql", "executed", "error_type", "error_msg",
                                     "n_rows", "exec_s", "gold_ok", "correct")},
            }
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = compute_metrics(records)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=1)

    meta = {
        "split": args.split,
        "model": "gold-as-pred" if args.gold_as_pred else args.model,
        "adapter": args.adapter,
        "limit": args.limit,
        "prompt_config": cfg,
        "prompt_config_sha1": hashlib.sha1(
            json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest(),
        "gold_cache": gold_stats,
        "generation_seconds": round(gen_s, 2),
        "execution_seconds": round(exec_s, 2),
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if not args.gold_as_pred:
        import importlib.metadata as md
        meta["versions"] = {}
        for pkg in ("vllm", "torch", "transformers", "peft"):
            try:
                meta["versions"][pkg] = md.version(pkg)
            except md.PackageNotFoundError:
                pass
    with open(out / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

    show = {k: v for k, v in metrics.items() if k != "gold_failed_ids"}
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()
