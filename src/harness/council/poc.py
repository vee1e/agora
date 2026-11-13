"""Run proof-of-concept scripts inside the sandbox and correct them on failure.

The model writes a PoC for a finding; the sandbox runs it; if it fails, the
failure output is fed back and the model rewrites it. Nothing is executed on the
host. The PoC only ever runs inside the agent's container, which has no network,
a read-only root filesystem, dropped capabilities, and a writable /work mount and a read-only bind of the staged checkout.
"""
from __future__ import annotations

import os
import shutil
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..containers import ContainerManager
    from ..db import Database
    from ..llm import LLMClient
    from ..logging_ import EventLogger
    from .. import models as m

SUCCESS_MARKER = "POC_SUCCESS"
POC_MAX_ATTEMPTS = 3
POC_MAX_TOKENS = 12_000
POC_RETRY_MAX_TOKENS = 24_000
POC_MAX_BYTES = 65_536
STAGED_DIR_NAME = "checkout"
STAGE_MAX_FILES = 400
STAGE_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv"}
_ID_PATTERN_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")

POC_SYSTEM_PROMPT = (
    "You write Python proof-of-concept scripts for security findings. A finding "
    "names a source-to-sink path in a codebase and the script must demonstrate it "
    "against a local copy, inside a sandboxed container.\n"
    "Environment facts you must rely on:\n"
    "- The codebase is staged read-only at /work/checkout inside the container.\n"
    "- Your script is saved to /work and run as `python3 /work/<name>.py` with cwd /work.\n"
    "- The container has gcc and python3, a read-only root filesystem, and NO network.\n"
    "- Write any build output under /work (for example /work/build).\n"
    "- You may compile C with gcc and run the binary you built.\n"
    "\n"
    "Rules: no network calls, no destructive payloads, no crash inputs beyond the "
    "one needed to demonstrate the bug, and nothing that touches paths outside "
    "/work. Keep the script under 64 KB.\n"
    "\n"
    "The script MUST exercise the real source that is staged at /work/checkout. "
    "Compile the actual file from /work/checkout (or import the actual Python "
    "module) and drive the real function. A script that re-implements, copies, or "
    "mirrors the vulnerable function proves nothing and is rejected. Your script "
    "must contain the path /work/checkout.\n"
    "\n"
    "Drive the target through its public API. Do not assign to internal struct "
    "fields or otherwise manufacture the precondition by hand; the trigger must "
    "arrive through the API's own inputs. Simulating resource exhaustion is "
    "allowed and is the correct way to prove an unchecked allocation or a lost "
    "realloc: use an interposer such as `-Wl,--wrap=malloc` or `LD_PRELOAD`, and "
    "say in a leading comment which resource you exhaust and why that is a "
    "reachable condition.\n"
    "\n"
    "A successful demonstration is a detected memory error or a crash. "
    "Build with AddressSanitizer when the bug is a memory error: "
    "`gcc -fsanitize=address -g -O0 /work/checkout/<file> ...`. AddressSanitizer "
    "is installed and works here. Run the target with your input and capture "
    "stdout and stderr. Stream the child process's stdout and stderr, and exit "
    "with the child's status. Do not print sanitizer strings or fake a fatal "
    "exit yourself. The harness decides success: it looks for an AddressSanitizer "
    "(or other sanitizer) report that mentions a path under /work/checkout, or a "
    "fatal crash signal (SIGSEGV, SIGABRT, and similar). A timeout, OOM kill, "
    "plain non-zero exit, or compile failure is not a successful demonstration.\n"
    "\n"
    "Return only the script body, no prose, no code fences."
)



_SANITIZER_MARKERS = (
    "ERROR: AddressSanitizer",
    "ERROR: LeakSanitizer",
    "ERROR: MemorySanitizer",
    "ERROR: ThreadSanitizer",
    "ERROR: UndefinedBehaviorSanitizer",
    "SUMMARY: AddressSanitizer",
    "SUMMARY: LeakSanitizer",
    "SUMMARY: MemorySanitizer",
    "SUMMARY: ThreadSanitizer",
    "SUMMARY: UndefinedBehaviorSanitizer",
)


# Crash signals only. SIGKILL (9), SIGHUP (1) and SIGTERM (15) are
# infrastructure or timeout leftovers, not exploit evidence.
_CRASH_SIGNALS = {4, 5, 6, 7, 8, 11}  # ILL, TRAP, ABRT, BUS, FPE, SEGV
_CHECKOUT_PATH_MARKER = f"/work/{STAGED_DIR_NAME}"


def poc_crash_evidence(
    exit_code: int,
    stdout: str,
    stderr: str,
    *,
    timed_out: bool = False,
) -> str | None:
    """Return a reason if the process crashed or a sanitizer fired."""
    if timed_out:
        return None
    blob = f"{stdout or ''}\n{stderr or ''}"
    for marker in _SANITIZER_MARKERS:
        if marker in blob and _CHECKOUT_PATH_MARKER in blob:
            return f"sanitizer reported {marker}"
    if exit_code in (-1, 124, 137):
        return None
    if exit_code < 0:
        sig = -exit_code
        if sig in _CRASH_SIGNALS:
            return f"process killed by signal {sig}"
        return None
    if exit_code >= 128:
        sig = exit_code - 128
        if sig in _CRASH_SIGNALS:
            return f"process exited with fatal status {exit_code}"
        return None
    return None


def script_forges_evidence(script: str) -> str | None:
    """Reject wrapper scripts that plant sanitizer markers or fake fatal exits."""
    for marker in _SANITIZER_MARKERS:
        if marker in script:
            return f"script contains sanitizer marker {marker!r}"
    return None


@dataclass
class PocAttempt:
    attempt: int
    script: str
    exit_code: int
    stdout: str
    stderr: str
    success: bool
    duration_ms: int = 0


@dataclass
class PocResult:
    finding_id: str
    success: bool
    attempts: list[PocAttempt] = field(default_factory=list)
    script_path: str | None = None
    reason: str | None = None


def _valid_id(value: str) -> bool:
    return bool(value) and len(value) <= 64 and all(c in _ID_PATTERN_CHARS for c in value)


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _clip(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _clear_write_bits(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    except OSError:
        return


def _restore_owner_write(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IWUSR)
    except OSError:
        return


def _empty_dir(target: Path) -> None:
    """Remove children of ``target`` without replacing the directory inode."""
    if not target.exists():
        return
    _restore_owner_write(target)
    for child in target.iterdir():
        try:
            if child.is_symlink() or child.is_file():
                _restore_owner_write(child)
                child.unlink()
            elif child.is_dir():
                for dirpath, dirnames, filenames in os.walk(
                    child, topdown=False, followlinks=False
                ):
                    current = Path(dirpath)
                    _restore_owner_write(current)
                    for name in filenames:
                        path = current / name
                        _restore_owner_write(path)
                        path.unlink(missing_ok=True)
                    for name in dirnames:
                        sub = current / name
                        _restore_owner_write(sub)
                        try:
                            sub.rmdir()
                        except OSError:
                            shutil.rmtree(sub, ignore_errors=True)
                try:
                    child.rmdir()
                except OSError:
                    shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


def stage_checkout(checkout: str | Path, workspace: Path) -> Path:
    """Copy the checkout into the workspace so the container can see it at /work/checkout."""
    source = Path(checkout).resolve()
    target = workspace / STAGED_DIR_NAME
    target.mkdir(parents=True, exist_ok=True)
    _empty_dir(target)
    copied = 0
    for dirpath, dirnames, filenames in os.walk(source, followlinks=False):
        current = Path(dirpath)
        # Do not walk through symlinked directories.
        dirnames[:] = [
            name
            for name in sorted(dirnames)
            if name not in STAGE_SKIP_DIRS and not (current / name).is_symlink()
        ]
        rel_dir = current.relative_to(source)
        if any(part in STAGE_SKIP_DIRS for part in rel_dir.parts):
            dirnames[:] = []
            continue
        dest_dir = target if rel_dir == Path('.') else target / rel_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            if copied >= STAGE_MAX_FILES:
                break
            src = current / name
            if src.is_symlink() or not src.is_file():
                continue
            try:
                if src.stat().st_size > 2_000_000:
                    continue
                dest = dest_dir / name
                shutil.copyfile(src, dest, follow_symlinks=False)
                _clear_write_bits(dest)
                copied += 1
            except OSError:
                continue
        if copied >= STAGE_MAX_FILES:
            break
    return target


def _finding_brief(finding: "m.Finding") -> str:
    return (
        f"finding id: {finding.id}\n"
        f"file: {finding.file}\n"
        f"line: {finding.line}\n"
        f"sink: {finding.sink}\n"
        f"call path: {' -> '.join(finding.call_path) if finding.call_path else '(unconfirmed)'}\n"
        f"detail: {_clip(finding.detail or '', 1500)}"
    )


def _build_messages(
    finding: "m.Finding", prior: PocAttempt | None
) -> list[dict[str, Any]]:
    if prior is None:
        user = (
            "Write a Python proof-of-concept for this finding.\n\n" + _finding_brief(finding)
        )
    else:
        user = (
            "Your previous proof-of-concept did not succeed. Fix it and return the "
            "corrected script body.\n\n"
            + _finding_brief(finding)
            + f"\n\nprevious attempt exit code: {prior.exit_code}"
            + f"\nprevious stdout:\n{_clip(prior.stdout)}"
            + f"\nprevious stderr:\n{_clip(prior.stderr)}"
            + f"\n\nprevious script:\n{_clip(prior.script)}"
        )
    return [
        {"role": "system", "content": POC_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


async def run_and_correct_poc(
    llm: "LLMClient",
    db: "Database",
    containers: "ContainerManager",
    container_id: str,
    finding: "m.Finding",
    checkout: str | Path,
    workspace: Path,
    agent_id: str,
    max_attempts: int = POC_MAX_ATTEMPTS,
    logger: "EventLogger | None" = None,
) -> PocResult:
    if not _valid_id(finding.id):
        return PocResult(finding_id=finding.id, success=False, reason="invalid finding id")
    workspace.mkdir(parents=True, exist_ok=True)
    workspace.chmod(0o777)
    stage_checkout(checkout, workspace)
    script_rel = f"poc_{finding.id}.py"
    script_path = workspace / script_rel
    result = PocResult(finding_id=finding.id, success=False)
    prior: PocAttempt | None = None

    for attempt in range(1, max_attempts + 1):
        messages = _build_messages(finding, prior)
        response = await llm.complete(
            messages,
            temperature=0.0,
            agent_id=agent_id,
            max_tokens=POC_MAX_TOKENS,
        )
        script = _strip_fences(getattr(response, "text", "") or "")
        if not script.strip() and getattr(response, "finish_reason", None) == "length":
            messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "Output the script now. Do not analyze. No reasoning, no "
                        "prose, no code fences. Start with `import` or `#!/usr/bin/env`."
                    ),
                }
            ]
            response = await llm.complete(
                messages,
                temperature=0.0,
                agent_id=agent_id,
                max_tokens=POC_RETRY_MAX_TOKENS,
            )
            script = _strip_fences(getattr(response, "text", "") or "")
        encoded = script.encode("utf-8", errors="replace")[:POC_MAX_BYTES]
        script = encoded.decode("utf-8", errors="replace")
        if not script.strip():
            record = PocAttempt(attempt, script, -1, "", "empty script from model", False)
            result.attempts.append(record)
            await db.add_poc_run(
                finding.id, agent_id, attempt, -1, False, "", "", "empty script", container_id
            )
            prior = record
            continue

        script_path.write_bytes(encoded)
        script_path.chmod(0o644)

        if f"/work/{STAGED_DIR_NAME}" not in script:
            record = PocAttempt(
                attempt=attempt,
                script=script,
                exit_code=-1,
                stdout="",
                stderr=(
                    f"rejected: the script does not use the staged source under "
                    f"/work/{STAGED_DIR_NAME}. Compile or import the real file from "
                    f"/work/{STAGED_DIR_NAME}; do not mirror the function."
                ),
                success=False,
            )
            result.attempts.append(record)
            await db.add_poc_run(
                finding.id,
                agent_id,
                attempt,
                -1,
                False,
                script,
                "",
                record.stderr,
                container_id,
            )
            prior = record
            continue

        forged = script_forges_evidence(script)
        if forged is not None:
            record = PocAttempt(
                attempt=attempt,
                script=script,
                exit_code=-1,
                stdout="",
                stderr=f"rejected: {forged}",
                success=False,
            )
            result.attempts.append(record)
            await db.add_poc_run(
                finding.id,
                agent_id,
                attempt,
                -1,
                False,
                script,
                "",
                record.stderr,
                container_id,
            )
            prior = record
            continue

        started = time.monotonic()
        try:
            run = await containers.exec(
                container_id,
                ["python3", f"/work/{script_rel}"],
                timeout=120,
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((time.monotonic() - started) * 1000)
            record = PocAttempt(
                attempt=attempt,
                script=script,
                exit_code=-1,
                stdout="",
                stderr=f"sandbox exec error: {exc}",
                success=False,
                duration_ms=duration_ms,
            )
            result.attempts.append(record)
            await db.add_poc_run(
                finding.id,
                agent_id,
                attempt,
                -1,
                False,
                script,
                "",
                record.stderr,
                container_id,
            )
            prior = record
            continue
        duration_ms = int((time.monotonic() - started) * 1000)
        evidence = poc_crash_evidence(
            run.exit_code,
            run.stdout,
            run.stderr,
            timed_out=bool(getattr(run, "timed_out", False)),
        )
        success = evidence is not None
        record = PocAttempt(
            attempt=attempt,
            script=script,
            exit_code=run.exit_code,
            stdout=run.stdout,
            stderr=run.stderr,
            success=success,
            duration_ms=duration_ms,
        )
        result.attempts.append(record)
        await db.add_poc_run(
            finding.id,
            agent_id,
            attempt,
            run.exit_code,
            success,
            script,
            _clip(run.stdout, 8000),
            _clip(run.stderr, 8000),
            container_id,
        )
        if logger is not None:
            logger.tool_call(
                agent_id=agent_id,
                name="run_poc",
                args={"finding_id": finding.id, "attempt": attempt},
                result={
                    "exit_code": run.exit_code,
                    "success": success,
                    "duration_ms": duration_ms,
                },
                latency_ms=duration_ms,
            )
        if success:
            result.success = True
            result.reason = evidence
            result.script_path = str(script_path)
            return result
        prior = record

    result.script_path = str(script_path) if script_path.exists() else None
    result.reason = "all attempts failed in the sandbox"
    return result
