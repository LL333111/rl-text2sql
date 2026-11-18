"""Dataset loading, schema serialization and prompt construction."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import yaml

from src import sql_exec

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPLITS_PATH = DATA / "splits.json"
RO_ROOT = DATA / "db_ro"
CACHE_ROOT = DATA / "cache"

_DB_DIRS = {
    "spider": DATA / "spider" / "database",
    "bird": DATA / "bird" / "dev_databases",
}
TEST_SPLITS = {"test_a", "test_b"}


# --------------------------------------------------------------------------
# Databases
# --------------------------------------------------------------------------

def source_db_path(dataset: str, db_id: str) -> Path:
    return _DB_DIRS[dataset] / db_id / f"{db_id}.sqlite"


def ro_db_path(dataset: str, db_id: str) -> Path:
    """Path of the read-only copy. All SQL execution goes through this path."""
    return RO_ROOT / dataset / f"{db_id}.sqlite"


def prepare_ro_copies(dataset: str, db_ids) -> None:
    for db_id in sorted(set(db_ids)):
        sql_exec.make_ro_copy(source_db_path(dataset, db_id), ro_db_path(dataset, db_id))


def gold_cache_path(dataset: str) -> Path:
    return CACHE_ROOT / f"gold_{dataset}.pkl"


@lru_cache(maxsize=None)
def schema_text(dataset: str, db_id: str) -> str:
    """CREATE TABLE statements read from sqlite_master, in creation order."""
    conn = sql_exec.connect_ro(ro_db_path(dataset, db_id))
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()
    stmts = []
    for (sql,) in rows:
        lines = [ln.rstrip() for ln in sql.replace("\r\n", "\n").replace("\t", " ").split("\n")]
        stmts.append("\n".join(ln for ln in lines if ln.strip()) + ";")
    return "\n\n".join(stmts)


# --------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------

def _spider_examples(filename: str, prefix: str) -> list[dict]:
    with open(DATA / "spider" / filename, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        {
            "id": f"{prefix}_{i}", "dataset": "spider", "db_id": r["db_id"],
            "question": r["question"], "evidence": "", "gold_sql": r["query"],
        }
        for i, r in enumerate(raw)
    ]


def _bird_dev_examples() -> list[dict]:
    with open(DATA / "bird" / "dev.json", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        {
            "id": f"bird_dev_{i}", "dataset": "bird", "db_id": r["db_id"],
            "question": r["question"], "evidence": r.get("evidence", "") or "",
            "gold_sql": r["SQL"],
        }
        for i, r in enumerate(raw)
    ]


def spider_train_all() -> list[dict]:
    """All 7,000 Spider train questions, before the split is applied."""
    return _spider_examples("train_spider.json", "spider_train")


def load_split(split: str, allow_test: bool = False) -> list[dict]:
    """Load 'train', 'val', 'test_a' (Spider dev) or 'test_b' (BIRD dev).

    Test splits are refused unless allow_test=True; that flag is only passed in
    Stage 4, after every choice is fixed (test-A and test-B are never read
    before then).
    """
    if split in TEST_SPLITS:
        if not allow_test:
            raise PermissionError(
                f"{split} is a test split and must not be read before Stage 4 "
                "(pass allow_test=True / --allow-test for the final evaluation only)."
            )
        return _spider_examples("dev.json", "spider_dev") if split == "test_a" else _bird_dev_examples()
    if split not in ("train", "val"):
        raise ValueError(split)
    with open(SPLITS_PATH, encoding="utf-8") as f:
        ids = set(json.load(f)[f"{split}_ids"])
    return [e for e in spider_train_all() if e["id"] in ids]


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_prompt_config(path: str | None = None) -> dict:
    with open(path or ROOT / "configs" / "prompt.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_messages(example: dict, cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_prompt_config()
    evidence = (example.get("evidence") or "").strip()
    evidence_block = cfg["evidence_template"].format(evidence=evidence) if evidence else ""
    user = cfg["user_template"].format(
        schema=schema_text(example["dataset"], example["db_id"]),
        evidence_block=evidence_block,
        question=example["question"].strip(),
    )
    return [
        {"role": "system", "content": cfg["system"].strip()},
        {"role": "user", "content": user},
    ]


_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Turn raw model output into the SQL string that gets executed.

    If the output contains a fenced code block, the first block is used;
    otherwise the whole output. Trailing semicolons are stripped.
    """
    m = _FENCE.search(text)
    sql = m.group(1) if m else text.replace("```sql", "").replace("```", "")
    return sql.strip().rstrip(";").strip()
