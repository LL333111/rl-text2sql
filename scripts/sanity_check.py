"""Harness sanity checks that complement G0.

G0 (gold-as-prediction = 100%) would also pass for a harness that marks
everything correct, so this adds a negative control: each val question is
scored against the gold SQL of a *different* question on the same database.
Spider questions come in paraphrase pairs that share one gold SQL, so the
control skips to the next question whose gold SQL text differs. The residual
accuracy is then reported together with how much of it is explained by the two
gold queries genuinely returning the same result set.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data  # noqa: E402
from src.rewards import Scorer  # noqa: E402

if __name__ == "__main__":
    val = data.load_split("val")
    train = data.load_split("train")
    s = Scorer(mode="partial")
    s.ensure_gold(val)
    naive = sum(val[i]["gold_sql"] == val[(i + 1) % len(val)]["gold_sql"] for i in range(len(val)))
    print(f"adjacent questions sharing one gold SQL (paraphrase pairs): {naive}/{len(val)}")
    norm = lambda q: " ".join(q.lower().split())
    preds, donors = [], []
    for i, e in enumerate(val):
        j = next(
            (k % len(val) for k in range(i + 1, i + len(val))
             if val[k % len(val)]["db_id"] == e["db_id"]
             and norm(val[k % len(val)]["gold_sql"]) != norm(e["gold_sql"])),
            None,
        )
        donors.append(j)
        preds.append(val[j]["gold_sql"] if j is not None else "SELECT 1 WHERE 0")
    recs = s.score(val, preds)
    same = sum(
        set(s.gold_result(val[i])["rows"]) == set(s.gold_result(val[j])["rows"])
        for i, j in enumerate(donors) if j is not None
    )
    n_correct = sum(r["correct"] for r in recs)
    print(f"negative control: {n_correct} marked correct; {same} donor queries truly return the same result set")
    assert n_correct == same, "harness marked a non-matching result as correct"
    print("negative control (different gold, same db): acc", round(n_correct / len(recs), 4),
          "| executable", round(sum(r["executed"] for r in recs) / len(recs), 4),
          "| mean partial reward", round(statistics.mean(r["reward"] for r in recs), 4))
    junk = s.score(val[:4], ["DROP TABLE x", "SELECT", "", "SELECT 1; SELECT 2"])
    print("junk predictions:", [(r["error_type"], r["reward"]) for r in junk])
    s.close()
    sp = json.load(open(data.SPLITS_PATH))
    print("dropped (gold SQL fails):", sp["dropped"]["train_gold_error"] + sp["dropped"]["val_gold_error"])
    lens = sorted(len(data.build_messages(e)[1]["content"]) for e in train + val)
    print("user-prompt chars: median", lens[len(lens) // 2], "p99", lens[int(len(lens) * .99)], "max", lens[-1])
    m = data.build_messages(val[0])
    print("---- sample prompt ----")
    print("[system]", m[0]["content"])
    print("[user]", m[1]["content"][:1000])
    print("[gold]", val[0]["gold_sql"])
