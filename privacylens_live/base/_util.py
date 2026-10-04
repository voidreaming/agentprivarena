"""Small shared helpers for the PrivacyLens-Live CLI, runner, and evaluators.

Kept dependency-free (stdlib only) so both ``base/`` and ``runner/`` can import
it without creating import cycles. Consolidates helpers that were previously
duplicated across ``cli.py``, ``runner/agent_runner.py``, ``base/evaluator.py``,
and ``base/trajectory_evaluator.py``.
"""

from __future__ import annotations


class JudgeUnavailableError(RuntimeError):
    """Raised when too many tasks failed to be judged for metrics to mean much.

    A judge that is down or erroring scores zero leakage over a zero
    denominator, which reads as a perfect mitigation result rather than a broken
    instrument. Failing loudly is the only way that difference survives.
    """


# Above this share of judge failures, callers refuse to aggregate rather than
# publish metrics over a silently shrunken denominator.
MAX_JUDGE_ERROR_RATE = 5.0


def rate(num: float, denom: float) -> float:
    """Percentage ``num/denom``, or ``0.0`` when ``denom`` is 0."""
    return (num / denom * 100) if denom else 0.0


def check_judge_health(
    *,
    total: int,
    judge_errors: int,
    committed: int,
    source: str,
) -> float:
    """Return the judge-error rate, raising if the judge was too unreliable.

    ``committed == 0`` is treated as fatal on its own whenever any task failed
    to be judged: every downstream rate then has a zero denominator and reports
    as 0.0, which is exactly what a dead judge produced in the audit-policy
    pilot -- 16 cells that looked like flawless mitigation.
    """
    error_rate = rate(judge_errors, total)
    if judge_errors and (committed == 0 or error_rate > MAX_JUDGE_ERROR_RATE):
        raise JudgeUnavailableError(
            f"{source}: judge failed on {judge_errors}/{total} tasks "
            f"({error_rate:.1f}%), committed={committed}. Refusing to report "
            "metrics -- a dead judge scores 0% leakage, indistinguishable from "
            "a perfect result. Fix the judge endpoint and re-evaluate."
        )
    return error_rate


def format_duration(seconds: float) -> str:
    """Format a duration as ``Ns`` / ``NmMss`` / ``NhMm``."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def task_sort_key(name: str) -> tuple[int, str]:
    """Numeric-aware sort key: ``main2`` < ``main10`` < ``main100``.

    Non-``mainN`` names sort after all numbered tasks, ordered by name.
    """
    if name.startswith("main") and name[4:].isdigit():
        return (int(name[4:]), name)
    return (10**9, name)
