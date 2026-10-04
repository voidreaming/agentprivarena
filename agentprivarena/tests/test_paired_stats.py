"""Tests for the paired-contrast statistics."""

from __future__ import annotations

import math

import pytest

from agentprivarena.base.paired_stats import (
    clustered_bootstrap_mean,
    mcnemar_exact_p,
    paired_contrast,
)


def test_mcnemar_matches_binomial_by_hand():
    # 10 discordant, all one way: 2 * (1/2)^10.
    assert mcnemar_exact_p(10, 0) == 2 * 0.5**10
    # Perfectly split discordant pairs cannot distinguish the conditions.
    assert mcnemar_exact_p(5, 5) == 1.0
    # No discordant pairs -> vacuous.
    assert mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_is_symmetric():
    assert mcnemar_exact_p(3, 11) == mcnemar_exact_p(11, 3)


def test_perfect_improvement():
    base = {f"t{i}": True for i in range(20)}
    treat = {f"t{i}": False for i in range(20)}
    r = paired_contrast(base, treat, samples=200)
    assert r.n == 20
    assert r.estimate == -1.0
    assert r.n_fixed == 20 and r.n_broken == 0
    assert r.p_value < 1e-5
    assert r.significant


def test_no_difference_is_not_significant():
    base = {f"t{i}": i % 2 == 0 for i in range(20)}
    r = paired_contrast(base, dict(base), samples=200)
    assert r.estimate == 0.0
    assert r.n_discordant == 0
    assert r.p_value == 1.0
    assert not r.significant


def test_discordant_counts_distinguish_equal_means():
    """-10pp built two ways: mostly-fixes vs fixes-cancelled-by-breaks.

    Both have the same point estimate; only the discordant split shows that one
    condition is reliable and the other is churning.
    """
    clean_base = {f"t{i}": i < 10 for i in range(100)}
    clean_treat = {f"t{i}": False for i in range(100)}
    clean = paired_contrast(clean_base, clean_treat, samples=200)

    churn_base = {f"t{i}": i < 30 for i in range(100)}
    churn_treat = {f"t{i}": 30 <= i < 50 for i in range(100)}
    churn = paired_contrast(churn_base, churn_treat, samples=200)

    assert math.isclose(clean.estimate, churn.estimate)
    assert (clean.n_fixed, clean.n_broken) == (10, 0)
    assert (churn.n_fixed, churn.n_broken) == (30, 20)
    assert churn.p_value > clean.p_value


def test_unpaired_keys_are_dropped_not_defaulted():
    """A missing treatment result must not count as a success."""
    base = {"a": True, "b": True, "c": True}
    treat = {"a": False}
    r = paired_contrast(base, treat, samples=100)
    assert r.n == 1
    assert r.n_fixed == 1


def test_no_overlap_returns_empty_contrast():
    r = paired_contrast({"a": True}, {"b": False}, samples=100)
    assert r.n == 0
    assert r.p_value == 1.0


def test_clustering_widens_the_interval():
    """The reason this module exists.

    The same 40 tasks appear under 4 models and the outcome is a property of
    the task, so the four rows per task are perfectly correlated. Resampling
    rows independently pretends there are 160 independent observations;
    resampling tasks admits there are 40.
    """
    base: dict[str, bool] = {}
    treat: dict[str, bool] = {}
    for task in range(40):
        leaks = task % 2 == 0
        for model in ("m1", "m2", "m3", "m4"):
            base[f"{model}:t{task}"] = leaks
            # Treatment fixes every other leaking task, identically per model.
            treat[f"{model}:t{task}"] = leaks and task % 4 != 0

    naive = paired_contrast(base, treat, samples=2000, seed=1)
    clustered = paired_contrast(
        base, treat, cluster_key=lambda k: k.split(":", 1)[1], samples=2000, seed=1
    )

    assert naive.n == clustered.n == 160
    assert math.isclose(naive.estimate, clustered.estimate)
    naive_width = naive.ci_high - naive.ci_low
    clustered_width = clustered.ci_high - clustered.ci_low
    assert clustered_width > naive_width


def test_bootstrap_is_deterministic_for_a_seed():
    base = {f"t{i}": i % 3 == 0 for i in range(50)}
    treat = {f"t{i}": i % 5 == 0 for i in range(50)}
    a = paired_contrast(base, treat, samples=500, seed=7)
    b = paired_contrast(base, treat, samples=500, seed=7)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_ci_brackets_the_estimate():
    base = {f"t{i}": i < 60 for i in range(100)}
    treat = {f"t{i}": i < 25 for i in range(100)}
    r = paired_contrast(base, treat, samples=1000)
    assert r.ci_low <= r.estimate <= r.ci_high


def test_describe_reports_units_and_counts():
    base = {f"t{i}": True for i in range(10)}
    treat = {f"t{i}": i >= 4 for i in range(10)}
    text = paired_contrast(base, treat, samples=200).describe()
    assert "-40.0pp" in text
    assert "disc 4/0" in text


def test_clustered_bootstrap_mean_recovers_the_mean():
    values = {f"t{i}": float(i % 4) for i in range(200)}
    r = clustered_bootstrap_mean(values, samples=500)
    assert r.n == 200
    assert abs(r.estimate - 1.5) < 1e-9
    assert r.ci_low <= r.estimate <= r.ci_high


def test_clustered_bootstrap_mean_finds_no_effect_when_centred_on_zero():
    # Symmetric around zero: the CI must contain it and p must not be small.
    values = {f"t{i}": 1.0 if i % 2 else -1.0 for i in range(200)}
    r = clustered_bootstrap_mean(values, samples=1000)
    assert r.ci_low <= 0.0 <= r.ci_high
    assert not r.significant


def test_clustered_bootstrap_mean_detects_a_shifted_mean():
    values = {f"t{i}": 1.0 for i in range(200)}
    r = clustered_bootstrap_mean(values, samples=1000)
    assert r.estimate == 1.0
    assert r.significant


def test_clustered_bootstrap_mean_p_is_floored_at_one_draw():
    """A bootstrap cannot resolve p below 1/samples; reporting 0 would overstate it."""
    values = {f"t{i}": 5.0 for i in range(50)}
    r = clustered_bootstrap_mean(values, samples=500)
    assert r.p_value == pytest.approx(1 / 500)


def test_clustered_bootstrap_mean_widens_when_clustering_repeats():
    # The same 25 tasks under 4 models: ignoring the cluster understates variance.
    values = {f"m{m}:t{t}": float(t % 5) for m in range(4) for t in range(25)}
    naive = clustered_bootstrap_mean(values, samples=1000, seed=3)
    clustered = clustered_bootstrap_mean(
        values, cluster_key=lambda k: k.split(":", 1)[1], samples=1000, seed=3
    )
    assert (clustered.ci_high - clustered.ci_low) > (naive.ci_high - naive.ci_low)
