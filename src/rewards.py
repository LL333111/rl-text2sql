"""Execution rewards (binary and partial-credit) and the batched scorer.

Scoring a prediction:
  1. look up the cached gold result (gold SQL is never re-executed here);
  2. execute the predicted SQL against the read-only copy, with a timeout;
  3. compare with the frozen rule ``set(pred_rows) == set(gold_rows)``.

Rewards:
  binary   1.0 on match, else 0.0
  partial  1.0 on match;
           executes but differs: 0.3 * F1(column sets) + 0.3 * Jaccard(row sets), <= 0.6;
           execution error or timeout: 0.0
  Result columns have no stable names, so a column is identified by its
  multiset of values; rows are compared as tuples.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src import data, sql_exec

REWARD_MODES = ("binary", "partial")


# --------------------------------------------------------------------------
# Pure reward functions
# --------------------------------------------------------------------------

def column_keys(rows: list[tuple]) -> set:
    """One hashable key per result column: the multiset of its values."""
    return {frozenset(Counter(col).items()) for col in zip(*rows)}


def column_f1(pred_rows: list[tuple], gold_rows: list[tuple]) -> float:
    p, g = column_keys(pred_rows), column_keys(gold_rows)
    tp = len(p & g)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(p), tp / len(g)
    return 2 * precision * recall / (precision + recall)


def row_jaccard(pred_rows: list[tuple], gold_rows: list[tuple]) -> float:
    p, g = set(pred_rows), set(gold_rows)
    union = len(p | g)
    return len(p & g) / union if union else 0.0


def compute_reward(mode: str, executed: bool, pred_rows: list[tuple], gold_rows: list[tuple]) -> tuple[bool, float]:
    """Return (correct, reward)."""
    if not executed:
        return False, 0.0
    if sql_exec.results_match(pred_rows, gold_rows):
        return True, 1.0
    if mode == "binary":
        return False, 0.0
    if mode == "partial":
        r = 0.3 * column_f1(pred_rows, gold_rows) + 0.3 * row_jaccard(pred_rows, gold_rows)
        return False, min(r, 0.6)
    raise ValueError(f"unknown reward mode: {mode}")


def group_composition(correct: list[bool], group_size: int) -> dict:
    """Share of rollout groups that are all-correct, all-wrong and mixed.

    `correct` is flat, with consecutive runs of `group_size` belonging to one prompt.
    Only mixed groups carry a learning signal under a binary reward.
    """
    assert len(correct) % group_size == 0, (len(correct), group_size)
    counts = Counter()
    for i in range(0, len(correct), group_size):
        k = sum(correct[i:i + group_size])
        counts["all_correct" if k == group_size else "all_wrong" if k == 0 else "mixed"] += 1
    n = max(1, len(correct) // group_size)
    return {
        "n_groups": len(correct) // group_size,
        **{f"frac_{k}": counts[k] / n for k in ("all_correct", "all_wrong", "mixed")},
    }


# --------------------------------------------------------------------------
# Worker side
# --------------------------------------------------------------------------

_GOLD: dict[str, sql_exec.GoldCache] = {}


def _gold(dataset: str) -> sql_exec.GoldCache:
    if dataset not in _GOLD:
        _GOLD[dataset] = sql_exec.GoldCache(data.gold_cache_path(dataset))
    return _GOLD[dataset]


def _run_gold(job: tuple[str, str, str]) -> tuple[str, sql_exec.ExecResult]:
    dataset, db_id, sql = job
    res = sql_exec.execute(data.ro_db_path(dataset, db_id), sql, timeout_s=sql_exec.GOLD_TIMEOUT_S)
    return sql_exec.gold_key(dataset, db_id, sql), res


def score_one(job: dict) -> dict:
    """Score one prediction. `job` has dataset, db_id, gold_sql, pred_sql, mode, timeout_s."""
    gold = _gold(job["dataset"]).get(sql_exec.gold_key(job["dataset"], job["db_id"], job["gold_sql"]))
    res = sql_exec.execute(
        data.ro_db_path(job["dataset"], job["db_id"]), job["pred_sql"], timeout_s=job["timeout_s"]
    )
    if gold["ok"]:
        correct, reward = compute_reward(job["mode"], res.ok, res.rows, gold["rows"])
    else:  # gold SQL itself does not execute: nothing can be scored against it
        correct, reward = False, 0.0
    return {
        "pred_sql": job["pred_sql"],
        "executed": res.ok,
        "error_type": res.error_type,
        "error_msg": res.error_msg,
        "n_rows": len(res.rows),
        "exec_s": round(res.elapsed_s, 4),
        "gold_ok": gold["ok"],
        "correct": correct,
        "reward": reward,
    }


# --------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------

class Scorer:
    """Process-pool scorer. Scores a whole batch of rollouts in one call and
    logs every reward call to a JSONL file."""

    def __init__(self, mode: str = "binary", timeout_s: float = sql_exec.DEFAULT_TIMEOUT_S,
                 workers: int | None = None, log_path: str | os.PathLike | None = None):
        assert mode in REWARD_MODES, mode
        self.mode = mode
        self.timeout_s = timeout_s
        self.workers = workers or max(1, min(16, (os.cpu_count() or 2) - 1))
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._pool: ProcessPoolExecutor | None = None

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers)
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def ensure_gold(self, examples: list[dict]) -> dict:
        """Execute and cache gold SQL for any example not yet in the cache.

        Must be called (and the cache saved) before workers score predictions,
        because each worker loads the cache file once.
        """
        stats = {"executed": 0, "cached": 0}
        by_ds: dict[str, list[dict]] = {}
        for e in examples:
            by_ds.setdefault(e["dataset"], []).append(e)
        for dataset, exs in by_ds.items():
            data.prepare_ro_copies(dataset, [e["db_id"] for e in exs])
            cache = sql_exec.GoldCache(data.gold_cache_path(dataset))
            jobs = {}
            for e in exs:
                k = sql_exec.gold_key(dataset, e["db_id"], e["gold_sql"])
                if k in cache:
                    stats["cached"] += 1
                else:
                    jobs[k] = (dataset, e["db_id"], e["gold_sql"])
            if jobs:
                if self._pool is not None:  # workers may hold a stale cache
                    self.close()
                with ProcessPoolExecutor(max_workers=self.workers) as pool:
                    for k, res in pool.map(_run_gold, jobs.values(), chunksize=16):
                        cache.put(k, res)
                cache.save()
                stats["executed"] += len(jobs)
            _GOLD.pop(dataset, None)
        return stats

    def gold_result(self, example: dict) -> dict:
        return _gold(example["dataset"]).get(
            sql_exec.gold_key(example["dataset"], example["db_id"], example["gold_sql"])
        )

    def score(self, examples: list[dict], pred_sqls: list[str], log_extra: dict | None = None) -> list[dict]:
        """Score pred_sqls[i] against examples[i]. Returns one record per prediction."""
        assert len(examples) == len(pred_sqls)
        jobs = [
            {"dataset": e["dataset"], "db_id": e["db_id"], "gold_sql": e["gold_sql"],
             "pred_sql": p, "mode": self.mode, "timeout_s": self.timeout_s}
            for e, p in zip(examples, pred_sqls)
        ]
        chunk = max(1, len(jobs) // (self.workers * 4))
        records = list(self._get_pool().map(score_one, jobs, chunksize=chunk))
        for e, r in zip(examples, records):
            r["id"] = e["id"]
            r["db_id"] = e["db_id"]
        if self.log_path:
            ts = time.time()
            with open(self.log_path, "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps({"ts": ts, "mode": self.mode, **(log_extra or {}), **r},
                                       ensure_ascii=False) + "\n")
        return records
