"""Paired contrasts for per-task binary outcomes: clustered bootstrap + McNemar.

Every leak-rate comparison in this project is paired within task -- the same
task is run under both conditions -- so the unit of analysis is the pair, not
the cell. Two things follow, and both were re-derived by hand in four separate
analysis scripts before this module existed:

**Cluster by task.** When results are pooled across executors, the same task
appears once per model, and those repeats are not independent: a task that is
easy to leak on is easy for every model. Resampling pairs independently
understates the variance by roughly 1.3x here. Resample *tasks* and take all
their pairs together.

**Report the discordant counts, not just the difference.** A -4pp difference
built from 150 fixed and 84 broken is a different claim from one built from 66
fixed and 0 broken, and the mean hides it. The exact McNemar test is computed
on exactly those counts, so they come back with the result.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass


__all__ = [
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "BootstrapMean",
    "PairedContrast",
    "clustered_bootstrap_mean",
    "mcnemar_exact_p",
    "paired_contrast",
]

DEFAULT_BOOTSTRAP_SAMPLES = 10_000


@dataclass(frozen=True, slots=True)
class PairedContrast:
    """``treatment - baseline`` over the tasks both conditions cover."""

    n: int
    """Number of paired observations (not tasks, when pooling across models)."""
    estimate: float
    """Difference in rate, as a fraction. Negative means the treatment leaks less."""
    ci_low: float
    ci_high: float
    p_value: float
    """Exact McNemar, two-sided."""
    n_fixed: int
    """Pairs true under baseline, false under treatment."""
    n_broken: int
    """Pairs false under baseline, true under treatment."""

    @property
    def n_discordant(self) -> int:
        return self.n_fixed + self.n_broken

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def as_pp(self) -> tuple[float, float, float]:
        """Estimate and CI in percentage points, the unit tables are written in."""
        return (100 * self.estimate, 100 * self.ci_low, 100 * self.ci_high)

    def describe(self) -> str:
        est, lo, hi = self.as_pp()
        return (
            f"n={self.n} {est:+.1f}pp [{lo:+.1f}, {hi:+.1f}] "
            f"disc {self.n_fixed}/{self.n_broken} p={self.p_value:.2e}"
        )


def mcnemar_exact_p(n_fixed: int, n_broken: int) -> float:
    """Two-sided exact McNemar on the discordant pairs.

    Concordant pairs carry no information about the direction of the effect and
    are excluded by construction; with no discordant pairs at all the test is
    vacuous and returns 1.0.
    """
    discordant = n_fixed + n_broken
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(n_fixed, n_broken) + 1)) / (
        2**discordant
    )
    return min(1.0, 2 * tail)


def paired_contrast(
    baseline: Mapping[str, bool],
    treatment: Mapping[str, bool],
    *,
    cluster_key: Callable[[str], str] | None = None,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> PairedContrast:
    """Compare two conditions over the keys they share.

    ``cluster_key`` maps an observation key to the unit that should be
    resampled. When pooling several models over the same tasks, key the
    observations ``"{model}:{task}"`` and pass ``lambda k: k.split(":", 1)[1]``
    so the bootstrap resamples tasks and keeps each task's per-model rows
    together. Leave it ``None`` for a single model, where each observation is
    already its own cluster and this reduces to an ordinary paired bootstrap.

    Keys present in only one mapping are dropped: an unpaired observation
    cannot contribute to a paired comparison, and silently treating a missing
    treatment result as "no leak" is how a crashed run turns into a good
    number.
    """
    keys = sorted(set(baseline) & set(treatment))
    n = len(keys)
    if n == 0:
        return PairedContrast(0, 0.0, 0.0, 0.0, 1.0, 0, 0)

    pairs = [(baseline[k], treatment[k]) for k in keys]
    estimate = sum(t - b for b, t in pairs) / n
    n_fixed = sum(1 for b, t in pairs if b and not t)
    n_broken = sum(1 for b, t in pairs if t and not b)

    clusters: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        clusters[cluster_key(key) if cluster_key else key].append(i)
    cluster_ids = list(clusters)

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        idx: list[int] = []
        for _ in range(len(cluster_ids)):
            idx.extend(clusters[cluster_ids[rng.randrange(len(cluster_ids))]])
        draws.append(sum(pairs[i][1] - pairs[i][0] for i in idx) / len(idx))
    draws.sort()

    return PairedContrast(
        n=n,
        estimate=estimate,
        ci_low=draws[int(0.025 * samples)],
        ci_high=draws[int(0.975 * samples)],
        p_value=mcnemar_exact_p(n_fixed, n_broken),
        n_fixed=n_fixed,
        n_broken=n_broken,
    )


@dataclass(frozen=True, slots=True)
class BootstrapMean:
    """A mean with a clustered bootstrap CI, and the share of draws past zero."""

    n: int
    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    """Two-sided bootstrap p: twice the smaller tail past zero, floored at 1/samples.

    A bootstrap p cannot resolve below one draw, so it is floored rather than
    reported as 0 -- writing p=0 for a statistic with 10k resamples overstates
    what the procedure can show.
    """

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def describe(self, unit: str = "pp") -> str:
        scale = 100 if unit == "pp" else 1
        return (
            f"n={self.n} {scale * self.estimate:+.1f}{unit} "
            f"[{scale * self.ci_low:+.1f}, {scale * self.ci_high:+.1f}] "
            f"p={self.p_value:.2g}"
        )


def clustered_bootstrap_mean(
    values: Mapping[str, float],
    *,
    cluster_key: Callable[[str], str] | None = None,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> BootstrapMean:
    """Mean of per-observation values with a cluster bootstrap CI.

    ``paired_contrast`` covers the two-condition binary case. This covers the
    statistics built *from* several conditions at once, where the per-observation
    quantity is no longer binary and McNemar does not apply -- chiefly the
    difference-in-differences behind "the criterion's ranking reverses between
    prompt and enforcement", whose per-observation value is
    ``(ci_audit - pii_audit) - (ci_prompt - pii_prompt)`` in {-2,...,2}.

    Same clustering rationale as ``paired_contrast``: key observations
    ``"{model}:{task}"`` and cluster on the task, or the same 389 tasks recurring
    across four models are counted as independent.
    """
    keys = sorted(values)
    n = len(keys)
    if n == 0:
        return BootstrapMean(0, 0.0, 0.0, 0.0, 1.0)

    xs = [values[k] for k in keys]
    estimate = sum(xs) / n

    clusters: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        clusters[cluster_key(key) if cluster_key else key].append(i)
    cluster_ids = list(clusters)

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        idx: list[int] = []
        for _ in range(len(cluster_ids)):
            idx.extend(clusters[cluster_ids[rng.randrange(len(cluster_ids))]])
        draws.append(sum(xs[i] for i in idx) / len(idx))
    draws.sort()

    below = sum(1 for d in draws if d <= 0) / samples
    above = sum(1 for d in draws if d >= 0) / samples
    p = min(1.0, 2 * min(below, above))

    return BootstrapMean(
        n=n,
        estimate=estimate,
        ci_low=draws[int(0.025 * samples)],
        ci_high=draws[int(0.975 * samples)],
        p_value=max(p, 1.0 / samples),
    )
