"""Sandboxed, read-only SQLite execution.

Safety layers:
  1. Generated SQL only ever runs against a *copy* of the database.
  2. The copy is opened with ``file:<path>?mode=ro`` (``uri=True``).
  3. An authorizer allows SELECT/READ/FUNCTION only; ATTACH, PRAGMA, any
     write and any DDL are denied before they run.
  4. ``sqlite3`` executes a single statement per call, so stacked statements
     are rejected.
  5. A progress handler interrupts queries that pass a wall-clock deadline,
     and the number of fetched rows is capped.

Comparison rule (frozen at the end of Stage 1):
  ``set(pred_rows) == set(gold_rows)``, as in the official BIRD evaluator.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import shutil
import sqlite3
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

DEFAULT_TIMEOUT_S = 10.0
GOLD_TIMEOUT_S = 120.0
MAX_ROWS = 500_000

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}


def _authorizer(action, arg1, arg2, dbname, source):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple] = field(default_factory=list)
    error_type: str | None = None  # see _classify_error
    error_msg: str | None = None
    elapsed_s: float = 0.0


def _classify_error(msg: str) -> str:
    m = msg.lower()
    if "interrupted" in m:
        return "timeout"
    if "not authorized" in m or "authorization denied" in m or "readonly" in m or "read-only" in m:
        return "not_authorized"
    if "one statement at a time" in m:
        return "multi_statement"
    if "no such table" in m:
        return "no_such_table"
    if "no such column" in m:
        return "no_such_column"
    if "no such function" in m:
        return "no_such_function"
    if "syntax error" in m or "incomplete input" in m or "unrecognized token" in m:
        return "syntax"
    if "ambiguous column" in m:
        return "ambiguous_column"
    if "misuse of aggregate" in m or "aggregate" in m:
        return "aggregate_misuse"
    return "other"


def connect_ro(db_path: str | os.PathLike) -> sqlite3.Connection:
    """Open a database read-only. Raises if the file does not exist."""
    p = Path(db_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    conn = sqlite3.connect(f"{p.as_uri()}?mode=ro", uri=True, check_same_thread=False)
    # Several Spider databases contain bytes that are not valid UTF-8.
    conn.text_factory = lambda b: b.decode("utf-8", errors="ignore")
    return conn


def execute(
    db_path: str | os.PathLike,
    sql: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int = MAX_ROWS,
) -> ExecResult:
    """Run one statement against a read-only database and return its rows."""
    t0 = time.monotonic()
    if not sql or not sql.strip():
        return ExecResult(False, error_type="empty", error_msg="empty SQL")
    deadline = t0 + timeout_s
    conn = connect_ro(db_path)
    try:
        conn.set_authorizer(_authorizer)
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
        cur = conn.execute(sql)
        rows: list[tuple] = []
        while True:
            chunk = cur.fetchmany(10_000)
            if not chunk:
                break
            rows.extend(chunk)
            if len(rows) > max_rows:
                return ExecResult(
                    False, error_type="too_many_rows",
                    error_msg=f"more than {max_rows} rows",
                    elapsed_s=time.monotonic() - t0,
                )
        return ExecResult(True, rows=rows, elapsed_s=time.monotonic() - t0)
    except (sqlite3.Error, sqlite3.Warning) as e:
        msg = str(e)
        return ExecResult(
            False, error_type=_classify_error(msg), error_msg=msg[:300],
            elapsed_s=time.monotonic() - t0,
        )
    except Exception as e:  # e.g. ValueError on embedded NUL, OverflowError
        return ExecResult(
            False, error_type="other", error_msg=f"{type(e).__name__}: {e}"[:300],
            elapsed_s=time.monotonic() - t0,
        )
    finally:
        conn.close()


def results_match(pred_rows: Iterable[tuple], gold_rows: Iterable[tuple]) -> bool:
    """The frozen comparison rule: order- and duplicate-insensitive set equality."""
    return set(pred_rows) == set(gold_rows)


# --------------------------------------------------------------------------
# Read-only copies
# --------------------------------------------------------------------------

def make_ro_copy(src: str | os.PathLike, dst: str | os.PathLike) -> Path:
    """Copy a database and strip write permission from the copy."""
    src, dst = Path(src), Path(dst)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        os.chmod(dst, stat.S_IREAD | stat.S_IWRITE)
    shutil.copyfile(src, dst)
    os.chmod(dst, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return dst


# --------------------------------------------------------------------------
# Gold result cache
# --------------------------------------------------------------------------

def gold_key(dataset: str, db_id: str, sql: str) -> str:
    return hashlib.sha1(f"{dataset}\x00{db_id}\x00{sql}".encode("utf-8")).hexdigest()


class GoldCache:
    """Gold execution results, computed once and stored on disk.

    Maps gold_key -> {"ok", "rows", "error_type", "error_msg"}.
    Gold SQL is never re-executed during training.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, "rb") as f:
                self.data = pickle.load(f)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str) -> dict:
        return self.data[key]

    def put(self, key: str, res: ExecResult) -> None:
        self.data[key] = {
            "ok": res.ok, "rows": res.rows,
            "error_type": res.error_type, "error_msg": res.error_msg,
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path)
