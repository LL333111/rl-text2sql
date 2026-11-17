"""Stage 4 analysis. Every reported number is generated here from results/.

Writes:
  results/analysis.json        main table, statistics, gates, reward summary
  results/main_table.md        the main table in Markdown
  results/review_candidates.md examples for the manual review
  figures/val_vs_gpu_time.png
  figures/reward_curve.png
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats import clustered_bootstrap_diff, mcnemar_exact  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
ROWS = [
    ("row1_zeroshot", "1. Qwen2.5-Coder-1.5B zero-shot", None),
    ("row2_sft_best", "2. + LoRA SFT (best on val)", "sft"),
    ("row2_sft_end", "2. + LoRA SFT (end of budget)", "sft"),
    ("row3_grpo_best", "3. + GRPO binary reward (best on val)", "grpo"),
    ("row3_grpo_end", "3. + GRPO binary reward (end of budget)", "grpo"),
    ("abl_partial_best", "Abl. GRPO partial-credit reward (best on val)", "grpo_partial"),
    ("abl_partial_end", "Abl. GRPO partial-credit reward (end of budget)", "grpo_partial"),
    ("ref_7b_zeroshot", "Ref. Qwen2.5-Coder-7B zero-shot", None),
]
SPLITS = [("test_a", "Spider dev"), ("test_b", "BIRD dev")]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_raw(row: str, split: str) -> list[dict] | None:
    p = RES / "final" / row / split / "raw.jsonl"
    return read_jsonl(p) if p.exists() else None


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def main() -> None:
    sel = json.loads((RES / "selection.json").read_text(encoding="utf-8"))
    out = {"val": {"row1": sel["row1_val_acc"], "G1": sel["G1_sft_beats_zeroshot"],
                   "G2": sel["G2_grpo_beats_zeroshot"], "arms": sel["arms"]}, "table": [], "stats": {}}
    ref_val = RES / "ref_7b_zeroshot" / "val" / "metrics.json"
    if ref_val.exists():
        out["val"]["ref_7b"] = json.loads(ref_val.read_text(encoding="utf-8"))["exec_acc"]

    # ---- main table -------------------------------------------------------
    lines = ["| Row | Val acc | Spider dev acc | Spider dev executable | BIRD dev acc | BIRD dev executable | "
             "Mean output tokens | Training GPU-hours |", "|---|---|---|---|---|---|---|---|"]
    for row, label, arm in ROWS:
        entry = {"row": row, "label": label}
        for split, _ in SPLITS:
            raw = load_raw(row, split)
            if raw is None:
                continue
            toks = [r["n_tokens"] for r in raw if r["n_tokens"] is not None]
            entry[split] = {"n": len(raw), "exec_acc": sum(r["correct"] for r in raw) / len(raw),
                            "executable_rate": sum(r["executed"] for r in raw) / len(raw),
                            "mean_output_tokens": sum(toks) / len(toks)}
        if "test_a" not in entry and "test_b" not in entry:
            continue
        if arm and arm in sel["arms"]:
            a = sel["arms"][arm]
            which = "best" if row.endswith("best") else "end"
            entry["val_acc"] = a[which]["val_acc"]
            entry["checkpoint"] = f'{a["chosen_run"]}/{a[which]["checkpoint"]}'
            entry["checkpoint_gpu_hours"] = a[which]["gpu_hours"]
            meta = json.loads((RES / a["chosen_run"] / "run_metadata.json").read_text(encoding="utf-8"))
            entry["train_gpu_hours"] = meta["train_seconds"] / 3600
        elif row == "row1_zeroshot":
            entry["val_acc"], entry["train_gpu_hours"] = sel["row1_val_acc"], 0.0
        elif row == "ref_7b_zeroshot":
            entry["val_acc"], entry["train_gpu_hours"] = out["val"].get("ref_7b"), 0.0
        out["table"].append(entry)
        ta, tb = entry.get("test_a", {}), entry.get("test_b", {})
        tok = [t["mean_output_tokens"] for t in (ta, tb) if t]
        gh = entry.get("checkpoint_gpu_hours")
        gh_s = f'{gh:.2f} (of {entry["train_gpu_hours"]:.2f})' if gh is not None else f'{entry.get("train_gpu_hours", 0):.2f}'
        lines.append(f'| {label} | {pct(entry.get("val_acc"))} | {pct(ta.get("exec_acc"))} | '
                     f'{pct(ta.get("executable_rate"))} | {pct(tb.get("exec_acc"))} | {pct(tb.get("executable_rate"))} | '
                     f'{sum(tok) / len(tok):.1f} | {gh_s} |')
    (RES / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---- paired statistics ------------------------------------------------
    pairs = [("row2_sft_best", "row3_grpo_best"), ("row2_sft_end", "row3_grpo_end"),
             ("row1_zeroshot", "row2_sft_best"), ("row1_zeroshot", "row3_grpo_best"),
             ("row3_grpo_best", "abl_partial_best")]
    for a, b in pairs:
        for split, _ in SPLITS:
            ra, rb = load_raw(a, split), load_raw(b, split)
            if ra is None or rb is None:
                continue
            assert [r["id"] for r in ra] == [r["id"] for r in rb]
            ca, cb = [r["correct"] for r in ra], [r["correct"] for r in rb]
            out["stats"][f"{b}_minus_{a}/{split}"] = {
                "mcnemar": mcnemar_exact(ca, cb),
                "bootstrap": clustered_bootstrap_diff(ca, cb, [r["db_id"] for r in ra]),
            }
    for k, v in out["stats"].items():
        b, m = v["bootstrap"], v["mcnemar"]
        print(f'{k}: diff {100 * b["diff"]:+.1f} pts, 95% CI [{100 * b["ci_low"]:+.1f}, {100 * b["ci_high"]:+.1f}] '
              f'({b["n_clusters"]} dbs), McNemar p={m["p_value"]:.4f} (only_a={m["only_a"]}, only_b={m["only_b"]})')

    # ---- reward / rollout-group summary ------------------------------------
    out["grpo_runs"] = {}
    for run in ("grpo_a", "grpo_b", "grpo_partial"):
        g = RES / run / "groups.jsonl"
        if not g.exists():
            continue
        rows = read_jsonl(g)
        k = max(1, len(rows) // 10)
        avg = lambda rs, f: sum(r[f] for r in rs) / len(rs)
        out["grpo_runs"][run] = {
            "reward_calls": len(rows), "rollouts": sum(r["n_rollouts"] for r in rows),
            **{f"first10pct_{f}": avg(rows[:k], f) for f in ("acc", "executable", "frac_mixed", "frac_all_correct", "frac_all_wrong")},
            **{f"last10pct_{f}": avg(rows[-k:], f) for f in ("acc", "executable", "frac_mixed", "frac_all_correct", "frac_all_wrong")},
        }

    # ---- figures ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG.mkdir(exist_ok=True)
        colors = {"sft_a": "#1f77b4", "sft_b": "#6baed6", "grpo_a": "#d62728", "grpo_b": "#ff9896", "grpo_partial": "#2ca02c"}
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for run, s in sel["runs"].items():
            pts = [(c["gpu_hours"], 100 * c["val_acc"]) for c in s["checkpoints"] if c["gpu_hours"] is not None]
            lr = json.loads((RES / run / "run_metadata.json").read_text(encoding="utf-8"))["trainer"]["learning_rate"]
            ax.plot(*zip(*pts), marker="o", ms=4, color=colors.get(run), label=f"{run} (lr {lr:g})")
        ax.axhline(100 * sel["row1_val_acc"], color="gray", ls="--", lw=1, label="1.5B zero-shot")
        if "ref_7b" in out["val"]:
            ax.axhline(100 * out["val"]["ref_7b"], color="black", ls=":", lw=1, label="7B zero-shot")
        ax.set_xlabel("Training GPU-hours (A40)"); ax.set_ylabel("Val execution accuracy (%)")
        ax.set_title("Val accuracy vs. GPU time (506 questions, 17 held-out databases)")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / "val_vs_gpu_time.png", dpi=150); plt.close(fig)

        runs = [r for r in ("grpo_a", "grpo_b", "grpo_partial") if (RES / r / "groups.jsonl").exists()]
        if runs:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            w = 10
            smooth = lambda v: [sum(v[max(0, i - w + 1): i + 1]) / len(v[max(0, i - w + 1): i + 1]) for i in range(len(v))]
            for run in runs:
                rows = read_jsonl(RES / run / "groups.jsonl")
                x = list(range(1, len(rows) + 1))
                axes[0].plot(x, smooth([r["mean_reward"] for r in rows]), color=colors.get(run), label=f"{run} reward")
                axes[0].plot(x, smooth([r["acc"] for r in rows]), color=colors.get(run), ls="--", lw=1, label=f"{run} execution-match rate")
                axes[1].plot(x, smooth([r["frac_mixed"] for r in rows]), color=colors.get(run), label=f"{run} mixed")
                axes[1].plot(x, smooth([r["frac_all_correct"] for r in rows]), color=colors.get(run), ls="--", lw=1, label=f"{run} all-correct")
                axes[1].plot(x, smooth([r["frac_all_wrong"] for r in rows]), color=colors.get(run), ls=":", lw=1, label=f"{run} all-wrong")
            axes[0].set_title(f"Training rollout reward ({w}-step moving average)")
            axes[1].set_title("Rollout-group composition (only mixed groups carry signal)")
            for a in axes:
                a.set_xlabel("Optimizer step (8 questions x 8 rollouts)"); a.grid(alpha=.3); a.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(FIG / "reward_curve.png", dpi=150); plt.close(fig)
    except ImportError:
        print("matplotlib not installed: figures skipped")

    # ---- manual review candidates -----------------------------------------
    rng = random.Random(0)
    md = ["# Manual review candidates\n", "Sampled with a fixed seed from Spider dev (test-A). "
          "SFT = row 2 best-on-val, GRPO = row 3 best-on-val.\n"]
    ra, rb = load_raw("row2_sft_best", "test_a"), load_raw("row3_grpo_best", "test_a")
    if ra and rb:
        buckets = {
            "GRPO right, SFT wrong": [(x, y) for x, y in zip(ra, rb) if y["correct"] and not x["correct"]],
            "GRPO wrong, SFT right": [(x, y) for x, y in zip(ra, rb) if x["correct"] and not y["correct"]],
            "Both wrong": [(x, y) for x, y in zip(ra, rb) if not x["correct"] and not y["correct"]],
            "Both right, different SQL (check for coincidental matches)":
                [(x, y) for x, y in zip(ra, rb) if x["correct"] and y["correct"]
                 and " ".join(y["pred_sql"].lower().split()) != " ".join(x["gold_sql"].lower().split())],
        }
        out["review_bucket_sizes"] = {k: len(v) for k, v in buckets.items()}
        for name, items in buckets.items():
            md.append(f"\n## {name} ({len(items)} total, {min(5, len(items))} sampled)\n")
            for x, y in rng.sample(items, min(5, len(items))):
                md += [f"**{x['id']}** ({x['db_id']}): {x['question']}", "", "```sql",
                       f"-- gold\n{x['gold_sql']}", f"-- SFT ({'ok' if x['correct'] else x['error_type'] or 'wrong result'})\n{x['pred_sql']}",
                       f"-- GRPO ({'ok' if y['correct'] else y['error_type'] or 'wrong result'})\n{y['pred_sql']}", "```", ""]
    (RES / "review_candidates.md").write_text("\n".join(md), encoding="utf-8")

    (RES / "analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
