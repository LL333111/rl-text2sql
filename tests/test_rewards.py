import pytest

from src.data import extract_sql
from src.rewards import column_f1, compute_reward, group_composition, row_jaccard

GOLD = [(1, "a"), (2, "b"), (3, "c")]


@pytest.mark.parametrize("mode", ["binary", "partial"])
def test_exact_match_is_one(mode):
    assert compute_reward(mode, True, list(reversed(GOLD)), GOLD) == (True, 1.0)
    assert compute_reward(mode, True, GOLD + GOLD[:1], GOLD) == (True, 1.0)  # duplicates ignored


@pytest.mark.parametrize("mode", ["binary", "partial"])
def test_execution_error_is_zero(mode):
    assert compute_reward(mode, False, [], GOLD) == (False, 0.0)


def test_binary_mismatch_is_zero():
    assert compute_reward("binary", True, GOLD[:2], GOLD) == (False, 0.0)


def test_empty_gold_matches_only_empty_pred():
    assert compute_reward("binary", True, [], []) == (True, 1.0)
    assert compute_reward("partial", True, [(1,)], []) == (False, 0.0)
    assert compute_reward("partial", True, [], GOLD) == (False, 0.0)


def test_partial_right_rows_extra_column():
    # Same rows, one extra column: columns P=2/3, R=2/2 -> F1=0.8; no row tuple matches.
    pred = [(1, "a", 9), (2, "b", 9), (3, "c", 9)]
    correct, r = compute_reward("partial", True, pred, GOLD)
    assert not correct
    assert r == pytest.approx(0.3 * 0.8 + 0.3 * 0.0)


def test_partial_subset_of_rows():
    # 2 of 3 rows: Jaccard = 2/3; the column multisets differ, so F1 = 0.
    correct, r = compute_reward("partial", True, GOLD[:2], GOLD)
    assert not correct
    assert r == pytest.approx(0.3 * (2 / 3))


def test_partial_swapped_columns_gets_column_credit_only():
    pred = [(b, a) for a, b in GOLD]
    correct, r = compute_reward("partial", True, pred, GOLD)
    assert not correct
    assert r == pytest.approx(0.3)  # F1 = 1, Jaccard = 0


def test_partial_is_capped_below_exact_match():
    # Duplicates change the column multisets but not the row set, so every
    # mismatch has F1 < 1 or Jaccard < 1; the reward never exceeds 0.6.
    for pred in ([(1, "a")], [(9, "z")], [(1,), (2,), (3,)], GOLD + [(4, "d")]):
        correct, r = compute_reward("partial", True, pred, GOLD)
        assert not correct and 0.0 <= r <= 0.6


def test_column_identity_is_a_multiset_of_values():
    assert column_f1([(1,), (1,), (2,)], [(1,), (2,), (2,)]) == 0.0
    assert column_f1([(2,), (1,), (1,)], [(1,), (2,), (1,)]) == 1.0
    assert row_jaccard([(1,), (2,)], [(2,), (3,)]) == pytest.approx(1 / 3)


def test_mixed_types_do_not_crash():
    pred = [(None, 1.5, b"x"), ("s", 2, None)]
    assert compute_reward("partial", True, pred, [(1, "a")]) == (False, 0.0)


def test_group_composition():
    flags = [True] * 4 + [False] * 4 + [True, False, False, False] + [False, True, True, True]
    g = group_composition(flags, 4)
    assert g == {"n_groups": 4, "frac_all_correct": 0.25, "frac_all_wrong": 0.25, "frac_mixed": 0.5}


def test_unknown_mode():
    with pytest.raises(ValueError):
        compute_reward("nope", True, [(1,)], [(2,)])


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("SELECT 1", "SELECT 1"),
        ("  SELECT 1;\n", "SELECT 1"),
        ("```sql\nSELECT 1;\n```", "SELECT 1"),
        ("Here you go:\n```sql\nSELECT a FROM t\n```\nDone.", "SELECT a FROM t"),
        ("```\nSELECT 1\n```", "SELECT 1"),
        ("```sql\nSELECT 1", "SELECT 1"),  # unterminated fence (hit the token limit)
        ("", ""),
    ],
)
def test_extract_sql(raw, expected):
    assert extract_sql(raw) == expected
