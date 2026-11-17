"""Checkpoint and configuration selection on val, and gates G1 / G2.

Reads results/<run>/eval_val/checkpoint-*/metrics.json for every run listed
below and writes results/selection.json:
  - per run: val accuracy of every checkpoint with its GPU time, the
    best-on-val checkpoint and the end-of-budget checkpoint
  - per arm: the configuration whose best checkpoint scores highest on val
  - G1 (best SFT > zero-shot on val) and G2 (best GRPO > zero-shot on val)

Selection uses val only. Ties go to the earlier checkpoint / first-listed run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ARMS = {
    "sft": ["sft_a", "sft_b"],
    "grpo": ["grpo_a", "grpo_b"],
    "grpo_partial": ["grpo_partial"],
}


def run_summary(run: str) -> dict | None:
    d = RES / run
    evals = sorted((d / "eval_val").glob("checkpoint-*/metrics.json"),
                   key=lambda p: int(p.parent.name.split("-")[1]))
    if not evals:
        return None
    saves = {}
    tl = d / "time_log.jsonl"
    if tl.exists():
        for line in tl.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["event"] == "save":
                saves[r["step"]] = r["elapsed_s"]
    ckpts = []
    for p in evals:
        m = json.loads(p.read_text(encoding="utf-8"))
        step = int(p.parent.name.split("-")[1])
        ckpts.append({"checkpoint": p.parent.name, "step": step,
                      "gpu_hours": round(saves[step] / 3600, 3) if step in saves else None,
                      "val_acc": m["exec_acc"], "val_executable": m["executable_rate"]})
    best = max(ckpts, key=lambda c: (c["val_acc"], -c["step"]))
    return {"run": run, "checkpoints": ckpts, "best": best, "end": ckpts[-1]}


def main() -> None:
    row1 = json.loads((RES / "row1_zeroshot" / "val" / "metrics.json").read_text(encoding="utf-8"))["exec_acc"]
    out = {"row1_val_acc": row1, "arms": {}, "runs": {}}
    for arm, runs in ARMS.items():
        summaries = [s for s in (run_summary(r) for r in runs) if s]
        for s in summaries:
            out["runs"][s["run"]] = s
        if summaries:
            chosen = max(summaries, key=lambda s: s["best"]["val_acc"])
            out["arms"][arm] = {"chosen_run": chosen["run"], "best": chosen["best"], "end": chosen["end"]}
    a = out["arms"]
    out["G1_sft_beats_zeroshot"] = ("sft" in a and a["sft"]["best"]["val_acc"] > row1) if "sft" in a else None
    out["G2_grpo_beats_zeroshot"] = ("grpo" in a and a["grpo"]["best"]["val_acc"] > row1) if "grpo" in a else None
    (RES / "selection.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"row1_val_acc": row1, "arms": a,
                      "G1": out["G1_sft_beats_zeroshot"], "G2": out["G2_grpo_beats_zeroshot"]}, indent=1))
    if "--require-g2" in sys.argv and not out["G2_grpo_beats_zeroshot"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
