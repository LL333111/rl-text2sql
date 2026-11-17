"""Paired statistics for comparing two rows on the same questions (plan 5.8)."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def mcnemar_exact(a_correct, b_correct) -> dict:
    """Exact (binomial) McNemar test on paired correctness flags."""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    assert a.shape == b.shape
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)
    return {"only_a": only_a, "only_b": only_b, "both": int((a & b).sum()),
            "neither": int((~a & ~b).sum()), "p_value": p}


def clustered_bootstrap_diff(a_correct, b_correct, clusters, n_boot: int = 10_000,
                             seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired bootstrap CI for accuracy(b) - accuracy(a), resampling whole
    clusters (databases) with replacement so that questions on one database
    stay together."""
    a, b = np.asarray(a_correct, float), np.asarray(b_correct, float)
    ids = np.unique(np.asarray(clusters), return_inverse=True)[1]
    k = ids.max() + 1
    diff_sum = np.bincount(ids, weights=b - a, minlength=k)
    size = np.bincount(ids, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, k, size=(n_boot, k))
    boots = diff_sum[draw].sum(1) / size[draw].sum(1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(b.mean() - a.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n_clusters": int(k), "n_boot": n_boot}
