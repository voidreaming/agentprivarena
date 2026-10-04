"""Build the paper's leak-rate tables from persisted cells, on the judge vote.

This lives in the tracked tree on purpose. The scripts that produced the
paper's numbers used to sit under ``.agent_tmp/``, which is gitignored, so the
code behind every published table was unversioned, untested, and duplicated --
the paired-bootstrap routine had been re-typed in four places, and a table was
once built from a superseded results directory because each script re-globbed
for cells its own way.

The pieces are separated so the risky parts are testable in isolation:
``load_cell`` turns a directory into verdicts, ``pool`` merges cells across
executors while keeping the task identity needed for clustering, and the
statistics come from :mod:`agentprivarena.base.paired_stats`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from agentprivarena.base.judge_ensemble import DEFAULT_ENSEMBLE, vote
from agentprivarena.base.paired_stats import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    PairedContrast,
    paired_contrast,
)


__all__ = [
    "CellVotes",
    "cluster_by_task",
    "leak_rate",
    "load_cell",
    "pool",
    "pooled_contrast",
    "pooled_leak_rate",
]


@dataclass(frozen=True, slots=True)
class CellVotes:
    """One condition x one executor, reduced to per-task majority verdicts."""

    verdicts: dict[str, bool]
    """Resolved tasks only. Ties and <2-vote tasks are excluded, not defaulted."""
    n_unresolved: int
    helpfulness: float

    def __len__(self) -> int:
        return len(self.verdicts)


def load_cell(results_dir: Path, judges: Sequence[str] = DEFAULT_ENSEMBLE) -> CellVotes:
    """Reduce a results directory to majority verdicts and mean helpfulness.

    Helpfulness is averaged within task across the judges that scored it, then
    across tasks, so a judge that skipped a task cannot shift the cell mean by
    contributing a different number of observations than its peers.
    """
    if not results_dir.exists():
        return CellVotes({}, 0, 0.0)
    verdicts = vote(results_dir, judges)
    resolved = {v.name: v.leaked for v in verdicts if v.leaked is not None}
    unresolved = sum(1 for v in verdicts if v.leaked is None)

    per_task: dict[str, list[float]] = {}
    for slug in judges:
        path = results_dir / f"judgments_{slug}.json"
        if not path.exists():
            continue
        try:
            rows = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            score = row.get("helpfulness_score")
            if row.get("status") == "ok" and isinstance(score, int):
                per_task.setdefault(str(row.get("name")), []).append(float(score))
    means = [sum(v) / len(v) for v in per_task.values() if v]
    return CellVotes(
        verdicts={k: bool(v) for k, v in resolved.items()},
        n_unresolved=unresolved,
        helpfulness=sum(means) / len(means) if means else 0.0,
    )


def leak_rate(cell: CellVotes) -> float:
    """Share of resolved tasks judged to leak, as a fraction."""
    return sum(cell.verdicts.values()) / len(cell) if len(cell) else 0.0


def pool(cells: Mapping[str, CellVotes]) -> dict[str, bool]:
    """Merge per-executor cells into one keyspace of ``"{executor}:{task}"``.

    The executor prefix keeps observations distinct while leaving the task
    recoverable, which is what :func:`cluster_by_task` needs: the same task
    recurs once per executor and those repeats are correlated.
    """
    out: dict[str, bool] = {}
    for executor, cell in cells.items():
        for task, leaked in cell.verdicts.items():
            out[f"{executor}:{task}"] = leaked
    return out


def cluster_by_task(key: str) -> str:
    """Cluster key for :func:`paired_stats.paired_contrast` over pooled cells."""
    return key.split(":", 1)[1]


def pooled_contrast(
    baseline: Mapping[str, CellVotes],
    treatment: Mapping[str, CellVotes],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> PairedContrast:
    """Paired contrast over pooled cells, clustered on task."""
    return paired_contrast(
        pool(baseline),
        pool(treatment),
        cluster_key=cluster_by_task,
        samples=samples,
        seed=seed,
    )


def pooled_leak_rate(cells: Iterable[CellVotes]) -> tuple[float, int]:
    """Leak rate over all tasks of all cells in one denominator, and that n.

    Pooled rather than an average of per-cell rates: with unequal cell sizes the
    two differ, and pooling is the one that matches the paired contrasts, which
    are computed over the same pooled observations.
    """
    cells = list(cells)
    total = sum(len(c) for c in cells)
    leaks = sum(sum(c.verdicts.values()) for c in cells)
    return (leaks / total if total else 0.0, total)
