import hashlib
import sqlite3

import pytest

from src import sql_exec
from src.sql_exec import execute, make_ro_copy, results_match


@pytest.fixture()
def db(tmp_path):
    src = tmp_path / "src.sqlite"
    conn = sqlite3.connect(src)
    conn.executescript(
        """
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, score REAL);
        INSERT INTO t VALUES (1, 'a', 1.5), (2, 'b', 2.5), (3, 'b', 2.5);
        """
    )
    conn.commit()
    conn.close()
    return make_ro_copy(src, tmp_path / "ro" / "copy.sqlite")


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_select_returns_rows(db):
    r = execute(db, "SELECT id, name FROM t ORDER BY id")
    assert r.ok and r.rows == [(1, "a"), (2, "b"), (3, "b")]


def test_empty_result_is_ok_and_empty(db):
    r = execute(db, "SELECT * FROM t WHERE id > 100")
    assert r.ok and r.rows == []
    assert results_match(r.rows, [])
    assert not results_match(r.rows, [(1,)])


def test_duplicate_rows_are_kept_but_ignored_by_comparison(db):
    r = execute(db, "SELECT name FROM t")
    assert r.rows == [("a",), ("b",), ("b",)]
    assert results_match(r.rows, [("b",), ("a",)])  # order- and duplicate-insensitive
    assert not results_match(r.rows, [("a",)])


def test_timeout(db):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    r = execute(db, sql, timeout_s=0.3)
    assert not r.ok and r.error_type == "timeout"
    assert r.elapsed_s < 5


def test_row_cap(db):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c LIMIT 5000) SELECT x FROM c"
    r = execute(db, sql, max_rows=1000)
    assert not r.ok and r.error_type == "too_many_rows"


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE t",
        "DELETE FROM t",
        "INSERT INTO t VALUES (9, 'z', 0)",
        "UPDATE t SET name = 'x'",
        "CREATE TABLE u (a)",
        "ALTER TABLE t ADD COLUMN extra",
        "ATTACH DATABASE 'other.sqlite' AS o",
        "PRAGMA writable_schema = 1",
        "VACUUM",
        "CREATE TEMP TABLE tmp AS SELECT * FROM t",
    ],
)
def test_destructive_statements_are_rejected(db, sql):
    before = _digest(db)
    r = execute(db, sql)
    assert not r.ok
    assert r.error_type == "not_authorized", (r.error_type, r.error_msg)
    assert _digest(db) == before
    assert sorted(p.name for p in db.parent.iterdir()) == [db.name]  # no new files


def test_stacked_statements_are_rejected(db):
    before = _digest(db)
    r = execute(db, "SELECT 1; DROP TABLE t")
    assert not r.ok and r.error_type == "multi_statement"
    assert _digest(db) == before
    assert execute(db, "SELECT count(*) FROM t").rows == [(3,)]


def test_write_blocked_by_read_only_mode_even_without_authorizer(db):
    conn = sql_exec.connect_ro(db)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DELETE FROM t")
    conn.close()


def test_error_classification(db):
    assert execute(db, "SELEC 1").error_type == "syntax"
    assert execute(db, "SELECT * FROM nope").error_type == "no_such_table"
    assert execute(db, "SELECT nope FROM t").error_type == "no_such_column"
    assert execute(db, "").error_type == "empty"
    assert execute(db, "   ").error_type == "empty"


def test_trailing_semicolon_ok(db):
    assert execute(db, "SELECT count(*) FROM t;").rows == [(3,)]


def test_gold_cache_roundtrip(db, tmp_path):
    cache = sql_exec.GoldCache(tmp_path / "c" / "gold.pkl")
    key = sql_exec.gold_key("d", "db", "SELECT id FROM t")
    assert key not in cache
    cache.put(key, execute(db, "SELECT id FROM t"))
    cache.save()
    again = sql_exec.GoldCache(tmp_path / "c" / "gold.pkl")
    assert key in again and again.get(key)["rows"] == [(1,), (2,), (3,)]
    assert sql_exec.gold_key("d", "db2", "SELECT id FROM t") != key


def test_non_utf8_text_does_not_crash(tmp_path):
    src = tmp_path / "bad.sqlite"
    conn = sqlite3.connect(src)
    conn.execute("CREATE TABLE t (s TEXT)")
    conn.execute("INSERT INTO t VALUES (CAST(x'41ff42' AS TEXT))")
    conn.commit()
    conn.close()
    r = execute(src, "SELECT s FROM t")
    assert r.ok and r.rows == [("AB",)]
