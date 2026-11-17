"""Hold out val by database and freeze data/splits.json.

  train  Spider train minus the val databases, minus questions whose gold SQL
         fails to execute or returns an empty result (an empty gold
         result would score any query that returns nothing as correct).
  val    ~10 databases (~500 questions) held out by database. Nothing is
         dropped from val except questions whose gold SQL fails to execute,
         which are listed individually.

The split is deterministic (seeded shuffle of the sorted database ids). The
script refuses to overwrite an existing splits.json unless --force is given.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data  # noqa: E402
from src.rewards import Scorer  # noqa: E402

SEED = 0
VAL_TARGET_QUESTIONS = 500
# Databases with more questions than this are not eligible for val, so that val
# is spread over about 10 databases instead of being dominated by one or two.
VAL_MAX_QUESTIONS_PER_DB = 80


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if data.SPLITS_PATH.exists() and not args.force:
        sys.exit(f"{data.SPLITS_PATH} already exists and is frozen; pass --force to regenerate.")

    examples = data.spider_train_all()
    per_db = Counter(e["db_id"] for e in examples)

    db_ids = sorted(db for db in per_db if per_db[db] <= VAL_MAX_QUESTIONS_PER_DB)
    random.Random(SEED).shuffle(db_ids)
    val_dbs, n_val = [], 0
    for db in db_ids:
        if n_val >= VAL_TARGET_QUESTIONS:
            break
        val_dbs.append(db)
        n_val += per_db[db]
    val_set = set(val_dbs)

    scorer = Scorer()
    print("gold cache:", scorer.ensure_gold(examples))

    train_ids, val_ids = [], []
    dropped = {"train_gold_error": [], "train_gold_empty": [], "val_gold_error": []}
    for e in examples:
        g = scorer.gold_result(e)
        is_val = e["db_id"] in val_set
        if not g["ok"]:
            dropped["val_gold_error" if is_val else "train_gold_error"].append(
                {"id": e["id"], "db_id": e["db_id"], "error": g["error_msg"]})
        elif is_val:
            val_ids.append(e["id"])
        elif len(g["rows"]) == 0:
            dropped["train_gold_empty"].append(e["id"])
        else:
            train_ids.append(e["id"])

    out = {
        "source": "spider/train_spider.json (7,000 questions)",
        "seed": SEED,
        "val_max_questions_per_db": VAL_MAX_QUESTIONS_PER_DB,
        "val_questions_per_db": {db: per_db[db] for db in sorted(val_dbs)},
        "val_databases": sorted(val_dbs),
        "train_databases": sorted(set(per_db) - val_set),
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            **{f"dropped_{k}": len(v) for k, v in dropped.items()},
        },
        "dropped": dropped,
        "train_ids": train_ids,
        "val_ids": val_ids,
    }
    with open(data.SPLITS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["counts"], indent=1))
    print("val databases:", ", ".join(out["val_databases"]))


if __name__ == "__main__":
    main()
