"""Pre-flight validation for result cells, so analysis cannot silently read a
superseded or degraded run.

Three incidents motivated this, all the same shape: two directories held the
same experimental condition, the stale one was still on disk, and a glob picked
it up.

* Mistral's five pilot cells existed twice -- a 2026-07-26 run on a stack whose
  MCP containers were stale (88-93% extraction, AUDIT-CI 36.4%) and a
  2026-07-29 re-run (98-100%, 15.0%). The stale copy inflated every audit row
  and shrank the headline effect from -29.2pp to -25.0pp.
* ``baseline_pii_redaction_mistral`` existed as a 26-task partial and a 100-task
  re-run.
* ``read_steer_only`` had read-time flows on 21% of tasks -- the arm whose whole
  purpose is read-time steering had nothing to steer with -- and was compared
  against healthy arms.

``run_metadata`` cannot catch any of these: the good and bad Mistral cells
record identical executor, auditor, mode and image. What separates them is
**derived** -- how much the extractor actually produced -- and the fact that the
condition is present twice at all.

Design follows the judge-health guard in ``_util.check_judge_health``: raise
rather than warn. A degraded cell that merely logs a warning is a cell that
ends up in a table.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


__all__ = [
    "CellHealth",
    "CellSignature",
    "CellValidationError",
    "describe_cell",
    "find_duplicate_conditions",
    "models_from_events",
    "problems_for",
    "require_healthy",
]

# usage_id under which the execution model and the auditor each bill their
# calls. Set by the runner, so they are stable across run eras.
EXEC_USAGE_ID = "default"
AUDIT_USAGE_ID = "agentprivarena-extraction"

# Healthy audit cells extract on 98-100% of tasks; the known-bad stale-stack
# cells sat at 88-93% and the broken read-steer arm at 21%. 0.95 separates them
# with room to spare.
MIN_EXTRACTION_RATE = 0.95
MIN_OK_RATE = 0.85

# Read-boundary steering notes per task, below which a low extraction rate means
# the audit was not running rather than finding nothing. Calibrated against the
# three archived stale-stack cells (0.16--0.18) and every healthy audited cell
# measured (1.23--2.22); the gap has no overlap, and 1.0 sits inside it.
MIN_STEERS_PER_TASK = 1.0


class CellValidationError(RuntimeError):
    """A results directory is unfit to analyse."""


@dataclass(frozen=True, slots=True)
class CellSignature:
    """The experimental condition a cell represents.

    Two directories sharing a signature are the same condition run twice, which
    is exactly the ambiguity that has to be resolved by hand rather than by
    whichever path a glob happens to yield first.
    """

    execution_model: str
    audit_model: str | None
    prompt_variant: str
    audit_mode: str | None
    audit_policy: str | None
    analyzer_enabled: bool
    recovered_from_events: bool = False
    """True when ``run_metadata`` was absent and the models came from events.

    Pre-dating the metadata field is not the same as being unverifiable: the
    events record which model billed each call, so the executor and auditor can
    be read back from the artifacts instead of trusted from a directory name.
    Recovery yields the models only -- prompt variant, audit mode and policy are
    not represented in events and stay unknown.
    """

    def label(self) -> str:
        bits = [self.execution_model, self.prompt_variant]
        if self.analyzer_enabled:
            bits.append(f"audit={self.audit_model}")
            if self.audit_mode:
                bits.append(self.audit_mode)
            if self.audit_policy:
                bits.append(self.audit_policy)
        return " / ".join(str(b) for b in bits)


@dataclass(frozen=True, slots=True)
class CellHealth:
    path: Path
    n_results: int
    n_ok: int
    signature: CellSignature | None
    extraction_rate: float
    """Share of tasks with at least one read-time information flow."""
    flows_per_task: float
    steers_per_task: float
    """Read-boundary steering notes emitted per task.

    This is the direct measurement of *did the audit fire at all*, which is what
    a stale MCP stack actually breaks. It separates the two causes of a low
    extraction rate cleanly: the three archived stale-stack cells sit at
    0.16--0.18 while every healthy audited cell measured sits at 1.23--2.22.
    """
    judges: tuple[str, ...]
    task_names: frozenset[str]
    """Which tasks the cell covers -- how scope is told apart from duplication."""

    @property
    def ok_rate(self) -> float:
        return self.n_ok / self.n_results if self.n_results else 0.0


def models_from_events(results_dir: Path) -> dict[str, set[str]]:
    """Map ``usage_id`` to the model(s) that billed under it, from one events file.

    Cells produced before ``run_metadata`` existed carry no record of which
    executor and auditor they used, which would otherwise leave a directory
    name as the only evidence -- and a directory name is precisely what the
    duplicate-condition incidents showed cannot be trusted. The events do carry
    it, because every LLM call records its ``usage_id`` alongside its model.

    Reads a single events file: the models are fixed for a run, so scanning all
    389 would cost time to re-derive a constant.
    """
    files = sorted(results_dir.glob("main*.events.json"))
    if not files:
        return {}
    try:
        payload = json.loads(files[0].read_text())
    except (OSError, json.JSONDecodeError):
        return {}

    found: dict[str, set[str]] = {}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            usage, model = node.get("usage_id"), node.get("model")
            if isinstance(usage, str) and isinstance(model, str):
                found.setdefault(usage, set()).add(model)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return found


def _result_files(results_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in results_dir.glob("main*.json")
        if ".events." not in p.name and "judgments" not in p.name
    )


def describe_cell(results_dir: Path) -> CellHealth:
    """Summarise a cell from its persisted results alone."""
    files = _result_files(results_dir)
    n_ok = 0
    with_flows = 0
    total_flows = 0
    total_steers = 0
    names: set[str] = set()
    signature: CellSignature | None = None
    for path in files:
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        names.add(str(row.get("name") or path.stem))
        if row.get("status") == "ok":
            n_ok += 1
        flows = row.get("privacy_flows") or []
        total_flows += len(flows)
        if flows:
            with_flows += 1
        stats = row.get("stats") or {}
        total_steers += int(stats.get("audit_instructions_emitted") or 0)
        if signature is None:
            meta = row.get("run_metadata")
            if isinstance(meta, dict):
                signature = CellSignature(
                    execution_model=str(meta.get("execution_model") or "?"),
                    audit_model=(
                        str(meta["audit_model"]) if meta.get("audit_model") else None
                    ),
                    prompt_variant=str(row.get("prompt_variant") or "?"),
                    audit_mode=(
                        str(meta["privacy_audit_mode"])
                        if meta.get("privacy_audit_mode")
                        else None
                    ),
                    audit_policy=(
                        str(
                            meta.get("audit_policy") or meta.get("privacy_audit_policy")
                        )
                        if (
                            meta.get("audit_policy") or meta.get("privacy_audit_policy")
                        )
                        else None
                    ),
                    analyzer_enabled=bool(meta.get("privacy_analyzer_enabled")),
                )
    n = len(files)
    if signature is None:
        # No run_metadata: recover what the artifacts do record rather than
        # falling back to trusting the directory name.
        usage = models_from_events(results_dir)
        exec_models = usage.get(EXEC_USAGE_ID, set())
        audit_models = usage.get(AUDIT_USAGE_ID, set())
        if exec_models:
            signature = CellSignature(
                execution_model=sorted(exec_models)[0],
                audit_model=sorted(audit_models)[0] if audit_models else None,
                prompt_variant="?",
                audit_mode=None,
                audit_policy=None,
                analyzer_enabled=bool(audit_models),
                recovered_from_events=True,
            )
    judges = tuple(
        sorted(
            p.name[len("judgments_") : -len(".json")]
            for p in results_dir.glob("judgments_*.json")
        )
    )
    return CellHealth(
        path=results_dir,
        n_results=n,
        n_ok=n_ok,
        signature=signature,
        extraction_rate=with_flows / n if n else 0.0,
        flows_per_task=total_flows / n if n else 0.0,
        steers_per_task=total_steers / n if n else 0.0,
        judges=judges,
        task_names=frozenset(names),
    )


def problems_for(
    health: CellHealth,
    *,
    expected_n: int | None = None,
    required_judges: Sequence[str] = (),
    min_extraction_rate: float = MIN_EXTRACTION_RATE,
    min_ok_rate: float = MIN_OK_RATE,
    min_steers_per_task: float = MIN_STEERS_PER_TASK,
    notes: list[str] | None = None,
) -> list[str]:
    """Return human-readable problems; empty means the cell is fit to analyse.

    ``notes`` collects findings that are worth surfacing but do not disqualify a
    cell. Pass a list to receive them; they are dropped otherwise.
    """
    out: list[str] = []
    notes = notes if notes is not None else []
    if health.n_results == 0:
        return [f"{health.path}: no result files"]
    if expected_n is not None and health.n_results < expected_n:
        out.append(
            f"{health.path}: {health.n_results}/{expected_n} results -- partial cell"
        )
    if health.ok_rate < min_ok_rate:
        out.append(
            f"{health.path}: only {health.n_ok}/{health.n_results} tasks ok "
            f"({100 * health.ok_rate:.0f}%)"
        )
    sig = health.signature
    if sig is None:
        out.append(
            f"{health.path}: no run_metadata and no events to recover from -- "
            "the condition rests on the directory name alone"
        )
    elif sig.analyzer_enabled:
        if health.extraction_rate < min_extraction_rate:
            # A low extraction rate has two causes that need separating, because
            # only one of them corrupts the cell. A stale MCP stack means the
            # audit barely ran; an executor that answers from search results
            # without opening records means it ran and correctly found nothing.
            # Steering volume distinguishes them and observation health does not
            # -- the archived stale-stack cells and Gemini-Flash-2.5 overlap on
            # every observation-content measure we tried, but the stale cells
            # emit 0.16--0.18 steering notes per task against 1.23--2.22 for
            # every healthy audited cell.
            #
            # This is an escape hatch only: it can allow a cell the old rule
            # refused, never refuse one it allowed. An arm that disables read
            # steering by design therefore keeps whatever verdict it had.
            if health.steers_per_task < min_steers_per_task:
                out.append(
                    f"{health.path}: analyzer was ON but only "
                    f"{100 * health.extraction_rate:.0f}% of tasks produced a "
                    f"read-time flow (expected "
                    f">={100 * min_extraction_rate:.0f}%), and the audit emitted "
                    f"only {health.steers_per_task:.2f} steering notes per task "
                    f"(expected >={min_steers_per_task:.2f}). The audit barely "
                    f"ran -- likely a stale MCP stack."
                )
            else:
                notes.append(
                    f"{health.path}: extraction rate is "
                    f"{100 * health.extraction_rate:.0f}% "
                    f"(below {100 * min_extraction_rate:.0f}%), but the audit "
                    f"was firing normally at {health.steers_per_task:.2f} "
                    f"steering notes per task, so this is executor behaviour "
                    f"rather than a broken stack: the agent reached its answer "
                    f"without opening records on some tasks, leaving the "
                    f"extractor nothing to extract."
                )
    elif health.flows_per_task > 0:
        out.append(
            f"{health.path}: analyzer was OFF but {health.flows_per_task:.2f} "
            "flows/task were recorded -- this is not a prompt-only cell"
        )
    missing = [j for j in required_judges if j not in health.judges]
    if missing:
        out.append(f"{health.path}: missing judgments for {', '.join(missing)}")
    return out


def find_duplicate_conditions(
    cells: Iterable[CellHealth],
) -> dict[CellSignature, list[Path]]:
    """Conditions run twice over the *same* tasks.

    Keyed on signature **and task set**, because signature alone conflates
    duplication with scope: a first100 cell and a full389 cell of the same
    condition share a signature but are different experiments, and flagging
    them would make the check noise. Requiring an identical task set catches
    the real case -- two directories covering the same 100 tasks under the same
    condition, one of them stale.

    A same-condition run over *fewer* tasks is a partial cell, which
    ``problems_for(expected_n=...)`` reports; it is deliberately not treated as
    a duplicate here, since a subset is indistinguishable from a narrower scope
    without knowing the intended range.

    A duplicate is never safe to resolve automatically -- picking the healthier
    or the newer one is a judgement about which run is canonical, and that
    belongs to whoever knows why the re-run happened. Archive the superseded
    copy (the convention here is an ``ARCHIVED_`` prefix, which the analysis
    globs skip) instead of leaving both in place.
    """
    seen: dict[tuple[CellSignature, frozenset[str]], list[Path]] = {}
    for cell in cells:
        if cell.signature is None:
            continue
        seen.setdefault((cell.signature, cell.task_names), []).append(cell.path)
    return {key[0]: paths for key, paths in seen.items() if len(paths) > 1}


def require_healthy(
    results_dirs: Iterable[Path],
    *,
    expected_n: int | None = None,
    required_judges: Sequence[str] = (),
    min_extraction_rate: float = MIN_EXTRACTION_RATE,
    min_ok_rate: float = MIN_OK_RATE,
) -> list[CellHealth]:
    """Validate every cell and raise on the first set of problems found."""
    cells = [describe_cell(Path(d)) for d in results_dirs]
    problems: list[str] = []
    for cell in cells:
        problems.extend(
            problems_for(
                cell,
                expected_n=expected_n,
                required_judges=required_judges,
                min_extraction_rate=min_extraction_rate,
                min_ok_rate=min_ok_rate,
            )
        )
    for sig, paths in find_duplicate_conditions(cells).items():
        joined = "\n      ".join(str(p) for p in paths)
        problems.append(
            f"duplicate condition [{sig.label()}] in {len(paths)} directories:\n"
            f"      {joined}\n"
            "      Archive the superseded copy (ARCHIVED_ prefix) before analysing."
        )
    if problems:
        raise CellValidationError(
            "refusing to analyse; "
            f"{len(problems)} problem(s) found:\n  - " + "\n  - ".join(problems)
        )
    return cells
