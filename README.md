# RL post-training for text-to-SQL (GRPO vs. SFT)

Work in progress. The design, gates and rules are in [PLAN.md](PLAN.md); results
and the full write-up will go here and in REPORT.md at Stage 4.

## Status

| Stage | State |
|---|---|
| 0 — environment and data | Done locally (Spider). BIRD dev is downloaded on the GPU machine. |
| 1 — evaluation harness | Harness, tests and gate G0 done. Baseline rows and the timing test need a GPU. |
| 2 — SFT baseline | Not started |
| 3 — GRPO | Not started |
| 4 — final evaluation | Not started; test-A and test-B have not been read |

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
python scripts/download_data.py spider     # official Spider release (Google Drive)
python scripts/make_splits.py              # refuses to overwrite the frozen data/splits.json
pytest
python -m src.evaluate --split val --gold-as-pred --out results/g0_val   # gate G0
python scripts/sanity_check.py             # negative control
```

## Evaluation rule

A prediction is correct when `set(pred_rows) == set(gold_rows)`, the rule used
by the official BIRD evaluator. The same rule is applied to Spider, which
**differs from Spider's official test-suite evaluation**. Generated SQL runs
only against read-only copies of the databases (`mode=ro`, a SELECT-only
authorizer, a 10 s timeout).
