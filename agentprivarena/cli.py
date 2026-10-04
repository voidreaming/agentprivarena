"""CLI entry point for AgentPrivArena.

Usage:
    python -m agentprivarena setup        # Start services + bootstrap tokens
    python -m agentprivarena write-env    # Regenerate .env from config.py
    python -m agentprivarena generate     # Generate task directories
    python -m agentprivarena verify       # Statically verify seed conversion
    python -m agentprivarena run          # Run agent on tasks
    python -m agentprivarena evaluate     # Evaluate results
    python -m agentprivarena trajectory-evaluate
                                             # Evaluate trajectory coverage
    python -m agentprivarena teardown     # Stop Docker services
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path

from agentprivarena.base._util import (
    format_duration as _fmt_duration,
    task_sort_key as _task_sort_key,
)
from agentprivarena.config import Config
from agentprivarena.runner.prompt_builder import VALID_PROMPT_VARIANTS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("cli")

COMPOSE_FILE = Path(__file__).parent / "docker-compose.yml"

# Infrastructure services that must be healthy before bootstrap can probe
# them. The MCP servers are deliberately excluded — they get started in a
# second pass after bootstrap has provisioned the tokens they need.
INFRA_SERVICES = [
    "bookstack-db",
    "bookstack",
    "mattermost-db",
    "mattermost",
    "mongo",
    "rocketchat",
    "mailpit",
    "gotosocial",
    "radicale",
]

MCP_SERVICES = [
    "bookstack-mcp",
    "mattermost-mcp",
    "rocketchat-mcp",
    "mailpit-mcp",
    "gotosocial-mcp",
    "radicale-mcp",
    "google-drive-mcp",
]


def _compose(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a docker-compose subcommand."""
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), *args]
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _wait_for_healthy(services: list[str], timeout: int = 300) -> None:
    """Poll `docker compose ps` until all listed services report healthy.

    Containers without a healthcheck are considered ready as soon as they're
    in state ``running`` (the bookstack and mattermost healthchecks set
    long start_periods, so honest "healthy" reporting is what we want).
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = _compose("ps", "--format", "json", check=False)
        if result.returncode != 0:
            time.sleep(2)
            continue
        # Each line is a JSON object describing one container.
        statuses: dict[str, str] = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            statuses[obj.get("Service", "")] = obj.get("Health") or obj.get("State", "")
        missing = [s for s in services if s not in statuses]
        unhealthy = [
            s
            for s in services
            if s in statuses and statuses[s] not in ("healthy", "running")
        ]
        if not missing and not unhealthy:
            logger.info("All %d infra services healthy.", len(services))
            return
        logger.info(
            "Waiting for services... missing=%s unhealthy=%s",
            missing or "[]",
            {s: statuses[s] for s in unhealthy} or "{}",
        )
        time.sleep(5)
    raise RuntimeError(
        f"Timed out after {timeout}s waiting for services to become healthy."
    )


def cmd_setup(_args: argparse.Namespace) -> None:
    """Bring up the stack and provision tokens.

    Flow:
      1. Write .env from current config.py defaults.
      2. ``docker compose up -d`` for the infrastructure services only.
      3. Wait for healthchecks.
      4. Run :func:`agentprivarena.bootstrap.bootstrap_all` to probe each
         service and provision any missing user / token.
      5. If anything was provisioned, re-write .env so the new values are
         picked up by the MCP server containers.
      6. ``docker compose up -d`` for the MCP servers (now with valid env).

    Idempotent: if everything's already provisioned, only steps 1, 3, and 6
    do real work and the bootstrap probes return immediately.
    """
    from agentprivarena.bootstrap import bootstrap_all

    # Step 1: write .env from current config defaults.
    config = Config.from_env()
    config.write_env_file()

    # Step 2: bring up the infra services only.
    logger.info("Starting infrastructure services...")
    _compose("up", "-d", *INFRA_SERVICES)

    # Step 3: wait for healthchecks.
    _wait_for_healthy(INFRA_SERVICES)

    # Step 4: bootstrap (probe + provision).
    logger.info("Running bootstrap probes...")
    config = bootstrap_all(config, COMPOSE_FILE)

    # Step 5: re-write .env in case bootstrap updated any tokens.
    config.write_env_file()

    # Step 6: bring up the MCP servers, forcing recreate so they pick up
    # any updated env vars from the regenerated .env.
    logger.info("Starting MCP servers...")
    _compose("up", "-d", "--force-recreate", *MCP_SERVICES)

    logger.info("Setup complete. Use 'docker compose ps' to check status.")


def cmd_write_env(_args: argparse.Namespace) -> None:
    """Regenerate agentprivarena/.env from config.py defaults.

    Useful for refreshing the file after editing config.py without bringing
    the stack up or down.
    """
    config = Config.from_env()
    path = config.write_env_file()
    print(f"Wrote {path}")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate task directories from main_data.json."""
    from agentprivarena.tasks.generate import generate_all

    data_path = Path(args.data)
    output_dir = Path(args.output)
    generate_all(
        data_path,
        output_dir,
        drop_recommended=not args.include_drop_recommended,
    )


def cmd_live_readiness(args: argparse.Namespace) -> None:
    """Check generated tasks against live service read paths."""
    from agentprivarena.tasks.live_readiness import _select_task_dirs, check_tasks

    names = [name.strip() for name in args.names.split(",") if name.strip()]
    task_dirs = _select_task_dirs(
        Path(args.tasks_dir),
        names=names or None,
        task_range=args.range,
        limit=args.limit,
    )
    report = asyncio.run(check_tasks(task_dirs))

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2))
    print(
        f"Live readiness: {report['ok']}/{report['total']} ok, "
        f"{report['failed']} failed. Report: {report_out}"
    )
    raise SystemExit(0 if report["failed"] == 0 else 1)


def _judge_base_url(explicit: str | None, judge_model: str, config: Config) -> str:
    """Resolve the judge's base URL, without silently forcing the main endpoint.

    Inheriting ``config.llm_base_url`` is right when the judge runs on the same
    deployment as the agent -- that is the common case and how the DeepSeek judge
    was configured. It is wrong when the judge is a different provider: sending
    an ``anthropic/`` or ``gemini/`` model to the Azure Foundry URL yields a 404
    on every task, which surfaces as "judge failed on 100/100" rather than as a
    configuration error.

    So the fallback applies only when the judge shares the configured model's
    provider prefix. Cross-provider judges get litellm's own routing unless a
    base URL is passed explicitly.

    The prefix test cannot separate *same provider* from *same deployment*, and
    the case where those differ is real: Azure Foundry serves an OpenAI-shaped
    API, so an Azure-hosted ``openai/gpt-5.4`` and a natively-hosted
    ``openai/gpt-5`` carry the same prefix and would both inherit the Foundry
    URL. Sending an ``sk-`` key there fails every task with "invalid
    subscription key", which reaches the caller as a dead judge rather than as a
    routing error. Pass ``none`` (or ``-``/``direct``) to select the provider's
    own endpoint -- the same sentinel ``LLM_BASE_URL`` already uses, and for the
    same reason: the empty string is falsy and cannot express the choice.
    """
    if explicit:
        return "" if explicit.strip().lower() in {"none", "-", "direct"} else explicit
    judge_provider = judge_model.split("/", 1)[0] if "/" in judge_model else ""
    main_provider = config.llm_model.split("/", 1)[0] if "/" in config.llm_model else ""
    return config.llm_base_url if judge_provider == main_provider else ""


def cmd_trajectory_evaluate(args: argparse.Namespace) -> None:
    """Evaluate process metrics and seed-record coverage for trajectories."""
    from agentprivarena.base.trajectory_evaluator import (
        TrajectoryCoverageJudge,
        evaluate_trajectory_results_dir,
    )

    config = Config.from_env()
    model = args.judge_model or config.eval_model
    base_url = _judge_base_url(args.judge_base_url, model, config)
    api_version = args.judge_api_version or config.llm_api_version
    api_key = args.judge_api_key or config.extraction_llm_api_key or config.llm_api_key
    if not api_key:
        logger.error(
            "No API key found. Set LLM_API_KEY, EXTRACTION_LLM_API_KEY, "
            "or pass --judge-api-key."
        )
        raise SystemExit(1)

    names = [name.strip() for name in args.names.split(",") if name.strip()]
    judge = TrajectoryCoverageJudge(
        model=model,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
    )

    def _progress(done: int, total: int) -> None:
        logger.info("Trajectory eval progress: %d/%d", done, total)

    report, judgments = asyncio.run(
        evaluate_trajectory_results_dir(
            results_dir=Path(args.results_dir),
            tasks_dir=Path(args.tasks_dir),
            judge=judge,
            max_concurrency=args.max_concurrency,
            use_cache=not args.no_cache,
            names=names or None,
            task_range=args.range,
            progress_callback=_progress,
            behavior_judge_model=model,
        )
    )

    report_out = Path(
        args.report_out or Path(args.results_dir) / "report_trajectory.json"
    )
    judgments_out = Path(
        args.judgments_out or Path(args.results_dir) / "trajectory_judgments.json"
    )
    report_out.parent.mkdir(parents=True, exist_ok=True)
    judgments_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    judgments_out.write_text(json.dumps(judgments, indent=2, ensure_ascii=False) + "\n")
    logger.info("Trajectory report written to %s", report_out)
    logger.info("Trajectory judgments written to %s", judgments_out)


def cmd_summarize_reports(args: argparse.Namespace) -> None:
    """Summarize existing behavior and trajectory reports across runs."""
    from agentprivarena.base.report_summarizer import (
        build_report_summary,
        parse_run_spec,
        render_markdown,
    )

    raw_runs = list(args.run or [])
    raw_runs.extend(args.results_dir or [])
    if not raw_runs:
        raise SystemExit(
            "Provide at least one --run LABEL=RESULTS_DIR or --results-dir."
        )

    runs = [parse_run_spec(raw) for raw in raw_runs]
    summary = build_report_summary(runs)
    markdown = render_markdown(summary)

    if args.json_out:
        json_out = Path(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        logger.info("Report summary JSON written to %s", json_out)
    if args.markdown_out:
        markdown_out = Path(args.markdown_out)
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(markdown)
        logger.info("Report summary Markdown written to %s", markdown_out)
    if not args.json_out and not args.markdown_out:
        print(markdown)


def cmd_run(args: argparse.Namespace) -> None:
    """Run agent on specified tasks."""
    from agentprivarena.runner.agent_runner import AgentPrivArenaRunner

    if args.privacy_audit_mode != "full" and not args.enable_privacy_analyzer:
        raise SystemExit(
            "--privacy-audit-mode requires --enable-privacy-analyzer when "
            "set to an ablation mode."
        )

    if args.audit_strictness != "balanced" and not args.enable_privacy_analyzer:
        raise SystemExit(
            "--audit-strictness requires --enable-privacy-analyzer when set "
            "to a non-default level."
        )

    audit_policy = args.privacy_policy
    if audit_policy.startswith("@"):
        audit_policy = Path(audit_policy[1:]).read_text()
    if audit_policy and not args.enable_privacy_analyzer:
        raise SystemExit("--privacy-policy requires --enable-privacy-analyzer.")

    if args.audit_policy != "contextual_integrity" and not args.enable_privacy_analyzer:
        raise SystemExit("--audit-policy requires --enable-privacy-analyzer.")

    for flag, enabled in (
        ("--audit-judge-blind", args.audit_judge_blind),
        ("--audit-verify-recompose", args.audit_verify_recompose),
        (
            "--disable-privacy-sequential-tool-calls",
            args.disable_privacy_sequential_tool_calls,
        ),
    ):
        if enabled and not args.enable_privacy_analyzer:
            raise SystemExit(f"{flag} requires --enable-privacy-analyzer.")

    config = Config.from_env()
    if args.model:
        config.llm_model = args.model

    runner = AgentPrivArenaRunner(
        config,
        max_clarification_rounds=args.max_clarifications,
        prompt_variant=args.prompt_variant,
        read_policy=args.read_policy,
        disable_security_analyzer=args.disable_security_analyzer,
        enable_privacy_analyzer=args.enable_privacy_analyzer,
        privacy_audit_mode=args.privacy_audit_mode,
        audit_strictness=args.audit_strictness,
        privacy_policy=audit_policy,
        audit_policy=args.audit_policy,
        audit_judge_blind=args.audit_judge_blind,
        audit_verify_recompose=args.audit_verify_recompose,
        privacy_sequential_tool_calls=(not args.disable_privacy_sequential_tool_calls),
    )

    # Determine which tasks to run
    tasks_base = Path(args.tasks_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.names:
        task_dirs = [tasks_base / name for name in args.names.split(",")]
    elif args.range:
        start, end = map(int, args.range.split("-"))
        all_tasks = sorted(
            [
                d
                for d in tasks_base.iterdir()
                if d.is_dir() and (d / "task.json").exists()
            ],
            key=lambda p: _task_sort_key(p.name),
        )
        task_dirs = all_tasks[start:end]
    else:
        task_dirs = sorted(
            [
                d
                for d in tasks_base.iterdir()
                if d.is_dir() and (d / "task.json").exists()
            ],
            key=lambda p: _task_sort_key(p.name),
        )

    # Apply --resume / --retry-errors filter against existing result files.
    #
    # Semantics:
    #   --resume                 : skip every task that already has any
    #                              result file (ok, no_action, OR error)
    #   --retry-errors           : run only tasks whose existing result
    #                              has status=error; skip everything else
    #                              that already has a result; still run
    #                              tasks that have NO result yet
    #   --resume --retry-errors  : skip ok/no_action, re-run errors
    if args.resume or args.retry_errors:
        kept: list[Path] = []
        skipped = 0
        for task_dir in task_dirs:
            result_file = results_dir / f"{task_dir.name}.json"
            if not result_file.exists():
                # No result yet — always run.
                kept.append(task_dir)
                continue
            try:
                data = json.loads(result_file.read_text())
            except (json.JSONDecodeError, OSError):
                # Corrupt result — re-run from scratch.
                kept.append(task_dir)
                continue
            status = data.get("status", "")

            # --retry-errors takes priority: an existing error always
            # re-runs when this flag is set, regardless of --resume.
            if args.retry_errors and status == "error":
                kept.append(task_dir)
                continue

            # Otherwise, anything with an existing result is skipped
            # (--resume covers all statuses, --retry-errors-only skips
            # non-errors).
            skipped += 1

        if skipped:
            logger.info(f"Filter: skipping {skipped} task(s) with existing results")
        task_dirs = kept

    if not task_dirs:
        logger.info(
            "Nothing to run. "
            "(Use --retry-errors to re-run failed tasks, "
            "or remove --resume to re-run everything.)"
        )
        return

    logger.info(f"Running {len(task_dirs)} tasks...")
    results = asyncio.run(runner.run_tasks(task_dirs, results_dir))

    # Final summary across the results from THIS run only (not cumulative
    # across resumes — for the cumulative view see cmd_status).
    ok = sum(1 for r in results if r.status == "ok")
    no = sum(1 for r in results if r.status == "no_action")
    err = sum(1 for r in results if r.status == "error")
    logger.info(f"Done. ok={ok}  no_action={no}  error={err}")


def _progress_bar(done: int, total: int, width: int = 40) -> str:
    """ASCII progress bar: ``[####------] 12.3% (61/493)``."""
    if total <= 0:
        return "[" + "-" * width + "]   0.0% (0/0)"
    pct = min(1.0, done / total)
    filled = int(round(pct * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {pct * 100:5.1f}% ({done}/{total})"


def cmd_verify(args: argparse.Namespace) -> None:
    """Statically verify the seed conversion from main_data.json."""
    from agentprivarena.tasks.verify import (
        print_report,
        verify_conversion,
    )

    report = verify_conversion(Path(args.data), Path(args.tasks_dir))
    print_report(report)

    out = (
        Path(args.report_out)
        if args.report_out
        else Path(args.tasks_dir) / "verify_report.json"
    )
    out.write_text(json.dumps(report, indent=2))
    logger.info(f"Wrote {out}")

    if report["exit_code"] != 0:
        raise SystemExit(report["exit_code"])


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate collected results.

    Runs the LLM-judge pass: leakage + helpfulness per task, plus
    routing correctness, clarification-bucket split, per-channel /
    per-expected-type breakdowns, and refusal-message leakage.

    Judge calls are cached at ``<results_dir>/.eval_cache.json`` keyed
    by content hashes, so re-running after adding a metric or fixing
    the aggregator is free. Output artifacts are named per judge model
    (``report_<slug>.json``, ``judgments_<slug>.json``) so runs with
    different judges don't clobber each other.
    """
    from agentprivarena.base.evaluator import (
        LLMJudgeEvaluator,
        evaluate_results_dir,
        slugify_judge_model,
    )

    results_dir = Path(args.results_dir)
    tasks_dir = Path(args.tasks_dir)
    if not results_dir.exists() or not results_dir.is_dir():
        logger.error(f"Results dir not found: {results_dir}")
        raise SystemExit(1)

    config = Config.from_env()
    judge_model = args.judge_model or config.eval_model
    judge_base_url = _judge_base_url(args.judge_base_url, judge_model, config)
    judge_api_version = args.judge_api_version or config.llm_api_version
    # Prefer explicit --judge-api-key, then fall back to the
    # extraction LLM key (handy for DeepSeek), then the main key.
    judge_api_key = (
        args.judge_api_key or config.extraction_llm_api_key or config.llm_api_key
    )
    if not judge_api_key:
        logger.error(
            "No API key found for the judge. Set LLM_API_KEY, "
            "EXTRACTION_LLM_API_KEY, or pass --judge-api-key."
        )
        raise SystemExit(1)
    # Name the endpoint explicitly. A misrouted judge fails every task, and the
    # symptom (100% judge errors) looks nothing like the cause.
    logger.info(
        "Using LLM judge: model=%s base_url=%s api_version=%s",
        judge_model,
        judge_base_url or "<provider default>",
        judge_api_version,
    )
    judge = LLMJudgeEvaluator(
        model=judge_model,
        api_key=judge_api_key,
        base_url=judge_base_url,
        api_version=judge_api_version,
    )
    judge_slug = slugify_judge_model(judge_model)

    def _progress(done: int, total: int) -> None:
        pct = (done / total * 100) if total else 0.0
        logger.info("Evaluated %d/%d (%.1f%%)", done, total, pct)

    report, judgments = asyncio.run(
        evaluate_results_dir(
            results_dir=results_dir,
            tasks_dir=tasks_dir,
            judge=judge,
            max_concurrency=args.max_concurrency,
            use_cache=not args.no_cache,
            sanitize_trajectory=args.sanitize_trajectory,
            batch_size=args.judge_batch_size,
            progress_callback=_progress,
        )
    )

    print(json.dumps(report, indent=2))

    report_path = (
        Path(args.report_out)
        if args.report_out
        else results_dir / f"report_{judge_slug}.json"
    )
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Report saved to {report_path}")

    # Also dump per-task judgments so downstream analysis doesn't need
    # to re-parse result files. Named per judge so concurrent/repeated
    # runs with different judges coexist.
    judgments_path = results_dir / f"judgments_{judge_slug}.json"
    judgments_path.write_text(
        json.dumps(
            [
                {
                    "name": j.name,
                    "status": j.status,
                    "bucket": j.bucket,
                    "expected_action_type": j.expected_action_type,
                    "actual_tool": j.actual_tool,
                    "routing_correct": j.routing_correct,
                    "helpfulness_score": j.helpfulness_score,
                    "llm_leaked": j.llm_leaked,
                    "llm_per_item": j.llm_per_item,
                    "behavior_label": j.behavior_label,
                    "judge_interventions": j.judge_interventions,
                    "send_action_attempts": j.send_action_attempts,
                    "privacy_flow_count": j.privacy_flow_count,
                    "refusal_leaked": j.refusal_leaked,
                    "refusal_per_item": j.refusal_per_item,
                    "sensitive_items": j.sensitive_items,
                }
                for j in judgments
            ],
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info(f"Per-task judgments saved to {judgments_path}")


def cmd_status(args: argparse.Namespace) -> None:
    """Show progress for a results directory.

    Reads every ``<results_dir>/<name>.json`` (skipping events sidecars,
    report_*.json, judgments_*.json, _summary.json, and any file that
    isn't a result-shape dict), groups by status, and reports the
    breakdown. If ``--tasks-dir`` is also given, also reports how many
    tasks remain to be run.
    """
    if not args.results_dir:
        logger.error(
            "--results-dir is required and must not be empty. "
            "(If you're using a shell variable, make sure it's set.)"
        )
        raise SystemExit(1)
    results_dir = Path(args.results_dir)
    if not results_dir.exists() or not results_dir.is_dir():
        logger.error(f"Results dir not found or not a directory: {results_dir}")
        raise SystemExit(1)

    result_files = sorted(
        p
        for p in results_dir.glob("*.json")
        if not p.name.endswith(".events.json")
        and not p.name.endswith(".privacy.json")
        and p.name != "_summary.json"
        and not p.name.startswith("report")
        and not p.name.startswith("judgments")
    )

    by_status: dict[str, int] = {"ok": 0, "no_action": 0, "error": 0, "unknown": 0}
    error_names: list[str] = []
    no_action_names: list[str] = []
    total_tool_calls = 0
    total_clarifications = 0
    total_recovered_errors = 0
    skipped_non_result = 0

    for f in result_files:
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            by_status["unknown"] += 1
            continue
        # Defensive: skip files that aren't a result-shape dict (e.g.
        # someone pointed --results-dir at the project root, which has
        # main_data.json as a top-level list).
        if not isinstance(data, dict) or "status" not in data:
            skipped_non_result += 1
            continue
        status = data.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        if status == "error":
            error_names.append(data.get("name", f.stem))
        if status == "no_action":
            no_action_names.append(data.get("name", f.stem))
        stats = data.get("stats", {}) or {}
        total_tool_calls += stats.get("tool_call_count", 0)
        total_clarifications += stats.get("clarification_rounds", 0)
        total_recovered_errors += stats.get("errors_recovered", 0)

    completed = sum(by_status.values())
    print()
    print(f"Results dir: {results_dir}")
    print(f"  completed: {completed}")
    for s in ("ok", "no_action", "error", "unknown"):
        n = by_status.get(s, 0)
        pct = (n / completed * 100) if completed else 0
        print(f"    {s:9} {n:>4}  ({pct:5.1f}%)")
    if skipped_non_result:
        print(
            f"  (also found {skipped_non_result} non-result JSON file(s) "
            f"in this dir — skipped)"
        )

    if completed > 0:
        print()
        print("  per-task means:")
        print(f"    tool_calls          {total_tool_calls / completed:.1f}")
        print(f"    errors_recovered    {total_recovered_errors / completed:.1f}")
        print(f"    clarification_rounds {total_clarifications / completed:.2f}")

    if error_names:
        print()
        n_show = min(10, len(error_names))
        print(f"  errored ({len(error_names)}, showing first {n_show}):")
        for name in error_names[:n_show]:
            print(f"    {name}")
        if len(error_names) > n_show:
            print(f"    ... and {len(error_names) - n_show} more")

    summary_file = results_dir / "_summary.json"
    summary_data: dict | None = None
    if summary_file.exists():
        try:
            loaded = json.loads(summary_file.read_text())
            if isinstance(loaded, dict):
                summary_data = loaded
        except (json.JSONDecodeError, OSError):
            pass

    # Compute timing: prefer the summary file (batch finished), else
    # fall back to the earliest result file's mtime as the start.
    started_at: float | None = None
    finished_at: float | None = None
    if summary_data:
        started_at = summary_data.get("started_at_unix")
        finished_at = summary_data.get("finished_at_unix")
    if started_at is None and result_files:
        started_at = min(f.stat().st_mtime for f in result_files)

    if started_at is not None:
        end_ref = finished_at if finished_at is not None else time.time()
        elapsed = max(0.0, end_ref - started_at)
    else:
        elapsed = 0.0

    if args.tasks_dir:
        tasks_base = Path(args.tasks_dir)
        if tasks_base.exists():
            all_task_names = {
                d.name
                for d in tasks_base.iterdir()
                if d.is_dir() and (d / "task.json").exists()
            }
            done_names = {f.stem for f in result_files}
            remaining = all_task_names - done_names
            total_expected = len(all_task_names)

            print()
            print(f"  Progress: {_progress_bar(completed, total_expected)}")
            if elapsed > 0:
                print(f"  Elapsed:  {_fmt_duration(elapsed)}")
                if completed > 0:
                    mean_per_task = elapsed / completed
                    rate_per_min = (completed / elapsed) * 60
                    print(
                        f"  Rate:     {_fmt_duration(mean_per_task)} per task  "
                        f"({rate_per_min:.1f} tasks/min)"
                    )
                    if remaining:
                        eta_seconds = mean_per_task * len(remaining)
                        print(f"  ETA:      {_fmt_duration(eta_seconds)}")

            print()
            print(f"  vs {tasks_base}:")
            print(f"    total tasks expected: {total_expected}")
            print(f"    remaining to run:     {len(remaining)}")
            if remaining:
                print(
                    f"    (resume with: python -m agentprivarena run "
                    f"--resume --results-dir {results_dir})"
                )

    # If a _summary.json exists from a finished batch, surface it.
    if summary_data:
        print()
        print("  last batch summary:")
        for k, v in summary_data.items():
            print(f"    {k}: {v}")


def cmd_teardown(_args: argparse.Namespace) -> None:
    """Stop all Docker services."""
    logger.info("Stopping AgentPrivArena services...")
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "down"],
        check=True,
    )
    logger.info("Services stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgentPrivArena: Real-service privacy evaluation platform"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # setup
    sub.add_parser("setup", help="Start Docker services + bootstrap tokens")

    # write-env
    sub.add_parser("write-env", help="Regenerate .env from config.py")

    # generate
    gen = sub.add_parser("generate", help="Generate task directories")
    gen.add_argument("--data", default="main_data.json")
    gen.add_argument("--output", default="agentprivarena/tasks")
    gen.add_argument(
        "--include-drop-recommended",
        action="store_true",
        help="Include low-quality tasks marked drop_recommended.",
    )

    # live-readiness
    ready = sub.add_parser(
        "live-readiness",
        help="Seed tasks into live services and verify read-tool retrievability.",
    )
    ready.add_argument("--tasks-dir", default="agentprivarena/tasks")
    ready.add_argument("--names", default="", help="Comma-separated task names.")
    ready.add_argument("--range", default=None, help="Index range, e.g. 0-25.")
    ready.add_argument("--limit", type=int, default=None)
    ready.add_argument(
        "--report-out",
        default=".agent_tmp/live_readiness_report.json",
    )

    # trajectory-evaluate
    traj = sub.add_parser(
        "trajectory-evaluate",
        help="Evaluate process metrics and seed-record coverage for trajectories.",
    )
    traj.add_argument("--results-dir", required=True)
    traj.add_argument("--tasks-dir", default="agentprivarena/tasks")
    traj.add_argument("--names", default="", help="Comma-separated task names.")
    traj.add_argument("--range", default=None, help="Index range, e.g. 0-100.")
    traj.add_argument(
        "--judge-model",
        default=None,
        help="Override the coverage judge model.",
    )
    traj.add_argument(
        "--judge-base-url",
        default=None,
        help="Override the coverage judge LLM base URL.",
    )
    traj.add_argument(
        "--judge-api-key",
        default=None,
        help="Override the coverage judge LLM API key.",
    )
    traj.add_argument(
        "--judge-api-version",
        default=None,
        help="Override the coverage judge LLM API version.",
    )
    traj.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Max concurrent LLM coverage judge calls.",
    )
    traj.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore .trajectory_eval_cache.json and re-call the LLM.",
    )
    traj.add_argument(
        "--report-out",
        default=None,
        help="Path to write the aggregate trajectory report.",
    )
    traj.add_argument(
        "--judgments-out",
        default=None,
        help="Path to write per-case trajectory judgments.",
    )

    # summarize-reports
    summ = sub.add_parser(
        "summarize-reports",
        help="Build outcome/intermediate/conditional/L3 summary tables.",
    )
    summ.add_argument(
        "--run",
        action="append",
        default=[],
        help=(
            "Run spec in LABEL=RESULTS_DIR form. May be repeated. "
            "If LABEL= is omitted, the directory name is used."
        ),
    )
    summ.add_argument(
        "--results-dir",
        action="append",
        default=[],
        help=(
            "Results directory to include. May be repeated; label defaults "
            "to the directory name."
        ),
    )
    summ.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the structured four-section summary JSON.",
    )
    summ.add_argument(
        "--markdown-out",
        default=None,
        help="Optional path to write Markdown tables.",
    )

    # run
    run = sub.add_parser("run", help="Run agent on tasks")
    run.add_argument("--tasks-dir", default="agentprivarena/tasks")
    run.add_argument("--names", help="Comma-separated task names")
    run.add_argument("--range", help="Index range (e.g., 0-10)")
    run.add_argument("--model", help="Override LLM model")
    run.add_argument("--results-dir", default="results")
    run.add_argument(
        "--max-clarifications",
        type=int,
        default=3,
        help=(
            "Max times to auto-respond when the agent ends a turn on "
            "a clarification question instead of a tool call. Set to "
            "0 to disable (single-shot mode, matches the original "
            "PrivacyLens single-turn benchmark)."
        ),
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip tasks where a successful result file already exists "
            "in --results-dir. Use this to continue an interrupted batch."
        ),
    )
    run.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "Re-run tasks whose existing result file has status=error. "
            "Combine with --resume to keep already-OK tasks and only "
            "rerun the failed ones."
        ),
    )
    run.add_argument(
        "--prompt-variant",
        default="baseline",
        # Single source of truth: AgentProfile validates against the same tuple,
        # so a variant added in prompt_builder is reachable from the CLI without
        # a second edit that can be forgotten.
        choices=list(VALID_PROMPT_VARIANTS),
        help=(
            "Privacy mitigation level. "
            "'baseline' (L0): stock OpenHands SWE prompt, no privacy. "
            "'reflect_freeform': baseline prompt + a generic, unstructured "
            "'reflect on privacy before sending' suffix (no CI structure, no "
            "analyzer) -- control for a plain second look vs L2/L3 CI. "
            "'data_minimization': baseline + 'share only the minimum needed' "
            "(GDPR data-minimization). 'pii_redaction': baseline + detect/mask "
            "third-party PII before sending. 'air_gap': baseline + two-pass "
            "extract-relevant-then-compose (prompt-only AirGapAgent). These "
            "three are prompt-only baselines (no analyzer). "
            "'privacy_enhanced' (L1): privacy-conscious system prompt + "
            "persona suffix. "
            "'ci_reasoning' (L2): L1 + structured CI 5-tuple "
            "information-flow analysis and per-flow judgment in suffix. "
            "'ci_audit' (L3): stock baseline execution prompt + neutral "
            "CI metadata guidance + CI schema fields on write tools + "
            "external extraction/judge "
            "(requires --enable-privacy-analyzer). "
            "'ci_audit_flows' (L3b): external audit injects structured "
            "annotated flows into execution context. "
            "'ci_audit_plan' (L3c): L3b plus mandatory think-tool write "
            "privacy plan before write actions. "
            "'ci_audit_contextual' (L3d): external audit asks contextual "
            "pre-write share questions using expected recipient/channel and "
            "injects pass/abstract/block decisions with safe renderings. "
            "Recorded per-result as prompt_variant."
        ),
    )
    run.add_argument(
        "--read-policy",
        default="natural",
        choices=["natural", "forced_oracle"],
        help=(
            "Retrieval policy axis. 'natural' lets the agent decide which "
            "read/search tools to call. 'forced_oracle' programmatically "
            "executes read-only startup reads over all seeded coverage records "
            "before the first LLM completion. Recorded per-result as read_policy."
        ),
    )
    run.add_argument(
        "--disable-security-analyzer",
        action="store_true",
        help=(
            "Pass system_prompt_kwargs={'llm_security_analyzer': False} "
            "to the Agent, suppressing the SECURITY_RISK_ASSESSMENT "
            "block in the system prompt. This is an orthogonal axis to "
            "--prompt-variant. The tool-schema security_risk/summary "
            "fields remain (hard-coded in the SDK's make_llm_completion), "
            "but the model is no longer taught how to use them, which "
            "should reduce the 'sends only security_risk+summary' "
            "failure mode observed in the baseline. Recorded per-result "
            "as security_analyzer_disabled."
        ),
    )
    run.add_argument(
        "--enable-privacy-analyzer",
        action="store_true",
        help=(
            "Enable the contextual privacy analyzer: injects four structured "
            "CI fields (data_type, data_subject, data_sender, data_recipient) "
            "into every non-readOnly tool's inputSchema and runs post-read "
            "extraction plus write-time judging. The judge prompt is separate "
            "from the execution model prompt."
        ),
    )
    run.add_argument(
        "--privacy-audit-mode",
        default="full",
        choices=["full", "read_steer_only", "write_enforce_only"],
        help=(
            "Component-ablation mode for L3. 'full' enables read-boundary "
            "steering and write-time enforcement. 'read_steer_only' records "
            "write judgments but executes every write. 'write_enforce_only' "
            "keeps extraction and write enforcement while hiding read-time "
            "guidance from the execution model. Non-full modes require "
            "--enable-privacy-analyzer."
        ),
    )
    run.add_argument(
        "--audit-strictness",
        default="balanced",
        choices=["lenient", "balanced", "strict"],
        help=(
            "CI-Audit strictness for genuinely ambiguous flows "
            "(ambiguous -> PASS/ABSTRACT/BLOCK). 'lenient' resolves ambiguity "
            "toward pass, 'balanced' toward abstract (default; = prior L3 "
            "behavior), 'strict' toward block. Clear cases are unaffected. "
            "Non-default levels require --enable-privacy-analyzer."
        ),
    )
    run.add_argument(
        "--audit-policy",
        default="contextual_integrity",
        choices=["contextual_integrity", "pii", "data_minimization"],
        help=(
            "Decision criterion the audit applies to each information flow "
            "(the plug-in policy). 'contextual_integrity' (default) judges "
            "appropriateness from recipient/channel/purpose; 'pii' judges only "
            "whether a flow carries third-party PII; 'data_minimization' judges "
            "only whether a flow is necessary for the task. The extraction and "
            "enforcement machinery is identical across policies. Requires "
            "--enable-privacy-analyzer."
        ),
    )
    run.add_argument(
        "--privacy-policy",
        default="",
        help=(
            "Optional user-authored privacy policy directive appended to the "
            'audit prompts (customizable disclosure rules, e.g. "never share '
            "the principal's home address or photos\"). Prefix with '@' to read "
            "from a file. Empty (default) = pure contextual-integrity inference. "
            "Requires --enable-privacy-analyzer."
        ),
    )
    run.add_argument(
        "--audit-judge-blind",
        action="store_true",
        help=(
            "Withhold read-boundary dispositions and reasons from the write-time "
            "judge, and drop the prompt line calling them 'a strong signal'. "
            "Makes the two audit boundaries independent so the write gate's "
            "marginal contribution is measurable. Requires "
            "--enable-privacy-analyzer."
        ),
    )
    run.add_argument(
        "--audit-verify-recompose",
        action="store_true",
        help=(
            "After a rejection, require the judge to confirm each previously "
            "flagged item was actually removed, instead of re-running the "
            "general criterion on the rewritten message. Unconfirmed items keep "
            "the gate closed. Requires --enable-privacy-analyzer."
        ),
    )
    run.add_argument(
        "--disable-privacy-sequential-tool-calls",
        action="store_true",
        help=(
            "Let the agent execute every tool call it plans in a turn instead of "
            "one per turn. Ablates the serialization the audit condition adds: "
            "writes composed alongside their reads can no longer be steered by "
            "the read-boundary note. Requires --enable-privacy-analyzer."
        ),
    )

    # verify
    ver = sub.add_parser("verify", help="Statically verify the seed conversion")
    ver.add_argument("--data", default="main_data.json")
    ver.add_argument("--tasks-dir", default="agentprivarena/tasks")
    ver.add_argument(
        "--report-out",
        default=None,
        help="Path to write JSON report (default: <tasks-dir>/verify_report.json)",
    )

    # status
    st = sub.add_parser(
        "status",
        help="Show progress for a results dir (live-monitor a running batch)",
    )
    st.add_argument("--results-dir", required=True)
    st.add_argument(
        "--tasks-dir",
        default=None,
        help="Optional: also report how many tasks remain to be run",
    )

    # evaluate
    ev = sub.add_parser("evaluate", help="Evaluate results")
    ev.add_argument("--results-dir", default="results")
    ev.add_argument(
        "--tasks-dir",
        default="agentprivarena/tasks",
        help=(
            "Directory of task.json files — needed to pull "
            "user_instruction for the helpfulness judge."
        ),
    )
    ev.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Override the judge model. Defaults to config.eval_model "
            "(openai/gpt-5.4). Use a different model from the one that "
            "produced the actions to avoid self-judging artifacts."
        ),
    )
    ev.add_argument(
        "--judge-base-url",
        default=None,
        help="Override the judge LLM base URL (e.g. for DeepSeek).",
    )
    ev.add_argument(
        "--judge-api-key",
        default=None,
        help="Override the judge LLM API key.",
    )
    ev.add_argument(
        "--judge-api-version",
        default=None,
        help="Override the judge LLM API version (empty string to clear).",
    )
    ev.add_argument(
        "--max-concurrency",
        type=int,
        default=16,
        help=(
            "Max concurrent LLM judge calls. Bumps this for faster "
            "evaluation, but watch for provider rate limits."
        ),
    )
    ev.add_argument(
        "--judge-batch-size",
        type=int,
        default=20,
        help=(
            "Tasks per cache-flush batch. Each batch is a barrier -- it waits "
            "for its slowest task -- so a batch smaller than --max-concurrency "
            "caps effective parallelism and makes throughput straggler-bound. "
            "Set it to a few times --max-concurrency for reasoning judges, at "
            "the cost of redoing more work if the run is interrupted."
        ),
    )
    ev.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Ignore .eval_cache.json and re-call the LLM for every "
            "prompt. Useful after changing the prompts."
        ),
    )
    ev.add_argument(
        "--sanitize-trajectory",
        action="store_true",
        help=(
            "Strip long text fields (body, markdown, content) from "
            "trajectory observations before sending to the helpfulness "
            "judge. Avoids Azure content-policy filter blocks."
        ),
    )
    ev.add_argument(
        "--report-out",
        default=None,
        help=(
            "Path to write the report JSON (default: "
            "<results-dir>/report_<judge-slug>.json)."
        ),
    )

    # teardown
    sub.add_parser("teardown", help="Stop Docker services")

    args = parser.parse_args()
    cmd_map = {
        "setup": cmd_setup,
        "write-env": cmd_write_env,
        "generate": cmd_generate,
        "live-readiness": cmd_live_readiness,
        "trajectory-evaluate": cmd_trajectory_evaluate,
        "summarize-reports": cmd_summarize_reports,
        "verify": cmd_verify,
        "run": cmd_run,
        "status": cmd_status,
        "evaluate": cmd_evaluate,
        "teardown": cmd_teardown,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
