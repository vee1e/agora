"""Council orchestration.

Runs the finder over one checkout, then killer and judge for each candidate,
then the exploit writer for survivors. `precision` is
``survived / max(1, survived + killed)``. Grouped notes are report-only text.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .. import models as m
from ..agent_runtime import clip_untrusted
from ..containers import ContainerManager
from ..db import Database
from ..message_board import MAX_POST_CHARS, MessageBoard, RateLimited
from ..llm import LLMClient
from ..logging_ import EventLogger
from .agents import (
    MAX_FINDINGS,
    CouncilResult,
    finder_audit,
    is_valid_id,
    judge_review,
    killer_refute,
    write_exploit,
)
from .poc import PocResult, run_and_correct_poc
from .tools import CouncilTools

MAX_REVIEWED_DEFAULT = 25
MAX_POCS_DEFAULT = 8


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _max_reviewed() -> int:
    return _env_int("HARNESS_MAX_REVIEWED", MAX_REVIEWED_DEFAULT)


def _max_pocs() -> int:
    return _env_int("HARNESS_MAX_POCS", MAX_POCS_DEFAULT)


async def run_council(
    llm: LLMClient,
    tools: CouncilTools,
    db: Database,
    message_board: MessageBoard,
    topic_id: str,
    checkout: str,
    finder_id: str,
    killer_id: str,
    judge_id: str,
    exploit_id: str,
    logger: EventLogger | None = None,
    containers: ContainerManager | None = None,
    max_reviewed: int | None = None,
    max_pocs: int | None = None,
) -> CouncilResult:
    for subscriber in (finder_id, killer_id, judge_id, exploit_id):
        await _subscribe(message_board, subscriber, topic_id)

    candidates = await finder_audit(
        llm, tools, db, topic_id, finder_id, checkout, logger=logger
    )
    candidates = candidates[:MAX_FINDINGS]
    await _announce_all(message_board, topic_id, finder_id, candidates)

    review_limit = max_reviewed if max_reviewed is not None else _max_reviewed()
    review_limit = max(1, review_limit)
    reviewed = candidates[:review_limit]
    pending = candidates[review_limit:]
    if pending and logger is not None:
        logger.event(
            "council_review_capped",
            topic_id=topic_id,
            reviewed=len(reviewed),
            pending=len(pending),
            limit=review_limit,
        )

    survivors: list[m.Finding] = []
    killed = 0
    for finding in reviewed:
        verdict = await killer_refute(
            llm, tools, db, finding, killer_id, logger=logger
        )
        await judge_review(
            llm,
            tools,
            db,
            finding,
            judge_id,
            killer_reason=verdict.reason,
            logger=logger,
        )
        stored = await db.get_finding(finding.id)
        status = stored.status if stored is not None else finding.status
        if status is m.FindingStatus.SURVIVED:
            survivors.append(stored if stored is not None else finding)
        elif status is m.FindingStatus.KILLED:
            killed += 1

    workspace = _workspace_for(message_board, exploit_id)
    notes = _grouped_notes(survivors)
    poc_limit = max_pocs if max_pocs is not None else _max_pocs()
    poc_limit = max(1, poc_limit)
    poc_survivors = survivors[:poc_limit]
    exploited = 0
    pocs_run = 0
    pocs_succeeded = 0
    ran_in_sandbox = False
    if poc_survivors and containers is not None and _sandbox_enabled(containers):
        counts = await _run_sandbox_pocs(
            llm,
            db,
            containers,
            poc_survivors,
            checkout,
            workspace,
            exploit_id,
            message_board,
            logger,
            notes,
        )
        if counts is not None:
            exploited, pocs_run, pocs_succeeded = counts
            ran_in_sandbox = True
    pocs_written = 0
    if poc_survivors and not ran_in_sandbox:
        pocs_written = await _run_host_pocs(
            llm, tools, db, poc_survivors, workspace, exploit_id, logger
        )

    survived = len(survivors)
    precision = survived / max(1, survived + killed)

    findings: list[m.Finding] = []
    for finding in candidates:
        refreshed = await db.get_finding(finding.id)
        findings.append(refreshed if refreshed is not None else finding)

    result = CouncilResult(
        topic_id=topic_id,
        candidates=len(candidates),
        survived=survived,
        killed=killed,
        exploited=exploited,
        precision=precision,
        findings=findings,
        notes=notes,
    )
    setattr(result, "pocs_run", pocs_run)
    setattr(result, "pocs_succeeded", pocs_succeeded)
    setattr(result, "reviewed", len(reviewed))
    setattr(result, "pending", len(pending))
    setattr(result, "pocs_capped", max(0, len(survivors) - len(poc_survivors)))
    setattr(result, "pocs_written", pocs_written)
    setattr(result, "sandbox_ran", ran_in_sandbox)
    return result


def _sandbox_enabled(containers: ContainerManager | None) -> bool:
    if containers is None:
        return False
    settings = getattr(containers, "settings", None)
    return bool(getattr(settings, "docker_enabled", False))


async def _run_host_pocs(
    llm: LLMClient,
    tools: CouncilTools,
    db: Database,
    survivors: list[m.Finding],
    workspace: Path,
    exploit_id: str,
    logger: EventLogger | None,
) -> int:
    written = 0
    for finding in survivors:
        path = await write_exploit(
            llm, tools, db, finding, workspace, exploit_id, logger=logger
        )
        if path is not None:
            written += 1
    if written and logger is not None:
        logger.event(
            "council_poc_written_host_side", count=written, topic_id=None
        )
    return written


async def _run_poc_safe(
    llm: LLMClient,
    db: Database,
    containers: ContainerManager,
    container_id: str,
    finding: m.Finding,
    checkout: str,
    workspace: Path,
    exploit_id: str,
    logger: EventLogger | None,
) -> PocResult | None:
    try:
        return await run_and_correct_poc(
            llm,
            db,
            containers,
            container_id,
            finding,
            checkout,
            workspace,
            exploit_id,
            logger=logger,
        )
    except Exception as exc:
        _log_logger(
            logger,
            "council_poc_error",
            finding_id=finding.id,
            error=f"{type(exc).__name__}: {exc}",
        )
        return None


async def _run_sandbox_pocs(
    llm: LLMClient,
    db: Database,
    containers: ContainerManager,
    survivors: list[m.Finding],
    checkout: str,
    workspace: Path,
    exploit_id: str,
    message_board: MessageBoard,
    logger: EventLogger | None,
    notes: list[str],
) -> tuple[int, int, int] | None:
    container_id: str | None = None
    try:
        try:
            container = await containers.create(exploit_id)
            container_id = container.id
        except Exception as exc:
            _log_event(
                message_board,
                "council_sandbox_unavailable",
                exploit_id=exploit_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            notes.append(
                "warning: sandbox unavailable; proof-of-concept scripts were "
                "written host-side"
            )
            return None
        exploited = 0
        pocs_run = 0
        pocs_succeeded = 0
        for finding in survivors:
            pocs_run += 1
            poc_result = await _run_poc_safe(
                llm,
                db,
                containers,
                container_id,
                finding,
                checkout,
                workspace,
                exploit_id,
                logger,
            )
            if poc_result is not None and poc_result.success:
                await db.set_finding_status(finding.id, m.FindingStatus.EXPLOITED)
                exploited += 1
                pocs_succeeded += 1
            else:
                reason = (
                    poc_result.reason
                    if poc_result is not None and poc_result.reason
                    else "proof-of-concept did not succeed in the sandbox"
                )
                _log_event(
                    message_board, "council_poc_failed", finding_id=finding.id, reason=reason
                )
        return (exploited, pocs_run, pocs_succeeded)
    finally:
        if container_id is not None:
            try:
                await containers.remove(container_id)
            except Exception as exc:
                _log_event(
                    message_board,
                    "council_container_remove_failed",
                    container_id=container_id,
                    error=f"{type(exc).__name__}: {exc}",
                )


def _log_logger(logger: EventLogger | None, kind: str, **fields: Any) -> None:
    if logger is None:
        return
    try:
        logger.event(kind, **fields)
    except Exception:
        return


async def _subscribe(message_board: MessageBoard, agent_id: str, topic_id: str) -> None:
    try:
        await message_board.subscribe(agent_id, topic_id)
    except Exception as exc:  # noqa: BLE001
        _log_event(
            message_board,
            "council_subscribe_failed",
            agent_id=agent_id,
            topic_id=topic_id,
            error=f"{type(exc).__name__}: {exc}",
        )


async def _announce_all(
    message_board: MessageBoard, topic_id: str, agent_id: str, findings: list[m.Finding]
) -> None:
    if not findings:
        return
    lines = [f"Council findings ({len(findings)}):"]
    refs: list[dict[str, Any]] = []
    for finding in findings:
        path = " -> ".join(finding.call_path) if finding.call_path else "(none)"
        lines.append(
            f"Finding {finding.id}: {finding.file}:{finding.line} "
            f"sink={finding.sink}\ncall path: {path}"
        )
        if finding.detail:
            lines.append(f"detail: {finding.detail}")
        refs.append(
            {
                "finding_id": finding.id,
                "file": finding.file,
                "line": finding.line,
                "sink": finding.sink,
            }
        )
    body = "\n".join(lines)[:MAX_POST_CHARS]
    try:
        await message_board.post(agent_id, topic_id, body, refs=refs)
    except RateLimited as exc:
        _log_event(
            message_board,
            "council_announce_rate_limited",
            agent_id=agent_id,
            topic_id=topic_id,
            finding_ids=[finding.id for finding in findings],
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        _log_event(
            message_board,
            "council_announce_failed",
            agent_id=agent_id,
            topic_id=topic_id,
            finding_ids=[finding.id for finding in findings],
            error=f"{type(exc).__name__}: {exc}",
        )


async def _announce(
    message_board: MessageBoard, topic_id: str, agent_id: str, finding: m.Finding
) -> None:
    path = " -> ".join(finding.call_path) if finding.call_path else "(none)"
    body = (
        f"Finding {finding.id}: {finding.file}:{finding.line} "
        f"sink={finding.sink}\ncall path: {path}"
    )
    if finding.detail:
        body = f"{body}\ndetail: {finding.detail}"
    body = body[:MAX_POST_CHARS]
    refs = [
        {
            "finding_id": finding.id,
            "file": finding.file,
            "line": finding.line,
            "sink": finding.sink,
        }
    ]
    try:
        await message_board.post(agent_id, topic_id, body, refs=refs)
    except RateLimited as exc:
        _log_event(
            message_board,
            "council_announce_rate_limited",
            agent_id=agent_id,
            topic_id=topic_id,
            finding_id=finding.id,
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        _log_event(
            message_board,
            "council_announce_failed",
            agent_id=agent_id,
            topic_id=topic_id,
            finding_id=finding.id,
            error=f"{type(exc).__name__}: {exc}",
        )


def _log_event(message_board: MessageBoard, kind: str, **fields: Any) -> None:
    logger = getattr(message_board, "logger", None)
    if logger is None:
        return
    try:
        logger.event(kind, **fields)
    except Exception:  # noqa: BLE001
        return


def _workspace_for(message_board: MessageBoard, exploit_id: str) -> Path:
    if not is_valid_id(exploit_id):
        raise ValueError(f"invalid exploit id: {exploit_id!r}")
    return Path(message_board.settings.workspace_dir) / exploit_id


def _grouped_notes(findings: list[m.Finding]) -> list[str]:
    groups: dict[tuple[str, tuple[str, ...]], list[m.Finding]] = {}
    for finding in findings:
        path = tuple(finding.call_path)
        if not path:
            continue
        groups.setdefault((finding.file, path), []).append(finding)

    notes: list[str] = []
    for (file, path), items in groups.items():
        sinks: list[str] = []
        for item in items:
            if item.sink not in sinks:
                sinks.append(item.sink)
        if len(sinks) < 2:
            continue
        raw = (
            "Grouped note: call path "
            + " -> ".join(path)
            + f" in {file} reaches {len(sinks)} sinks ({', '.join(sinks)}). "
            "Report only; no payloads, no crash inputs."
        )
        notes.append(clip_untrusted(raw))
    return notes
