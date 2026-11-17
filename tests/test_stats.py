import numpy as np
import pytest

from src.stats import clustered_bootstrap_diff, mcnemar_exact


def test_mcnemar_counts_and_p():
    a = [1, 1, 1, 0, 0, 0, 1, 0]
    b = [1, 0, 0, 0, 1, 0, 1, 0]
    r = mcnemar_exact(a, b)
    assert (r["only_a"], r["only_b"], r["both"], r["neither"]) == (2, 1, 2, 3)
    assert r["p_value"] == pytest.approx(1.0)  # 2 vs 1 discordant: two-sided binomial p = 1


def test_mcnemar_strong_effect_and_no_discordance():
    assert mcnemar_exact([0] * 20, [1] * 20)["p_value"] == pytest.approx(2 * 0.5 ** 20)
    assert mcnemar_exact([1, 0, 1], [1, 0, 1])["p_value"] == 1.0


def test_bootstrap_point_estimate_and_ci_contains_it():
    rng = np.random.default_rng(1)
    a = rng.random(600) < 0.5
    b = rng.random(600) < 0.6
    clusters = np.repeat(np.arange(30), 20)
    r = clustered_bootstrap_diff(a, b, clusters, n_boot=2000)
    assert r["diff"] == pytest.approx(b.mean() - a.mean())
    assert r["ci_low"] < r["diff"] < r["ci_high"]
    assert r["n_clusters"] == 30


def test_clustering_widens_the_interval():
    # All of the improvement sits in 2 of 10 databases: resampling by database
    # must give a wider interval than treating questions as independent.
    a = np.zeros(1000, bool)
    b = np.zeros(1000, bool)
    b[:200] = True
    by_db = clustered_bootstrap_diff(a, b, np.repeat(np.arange(10), 100), n_boot=4000)
    by_q = clustered_bootstrap_diff(a, b, np.arange(1000), n_boot=4000)
    assert (by_db["ci_high"] - by_db["ci_low"]) > 3 * (by_q["ci_high"] - by_q["ci_low"])


def test_bootstrap_is_deterministic_and_handles_string_clusters():
    a, b = [1, 0, 1, 0], [1, 1, 1, 0]
    c = ["x", "x", "y", "z"]
    assert clustered_bootstrap_diff(a, b, c, n_boot=500) == clustered_bootstrap_diff(a, b, c, n_boot=500)
