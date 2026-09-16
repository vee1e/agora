"""Append-only JSONL event logging.

One JSON object per line. No third-party imports: json, time, pathlib, sys.
Every log call is best-effort and must never raise into the caller.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

LLM_LOG = "llm.jsonl"
TOOL_LOG = "tools.jsonl"
EVENT_LOG = "events.jsonl"

DEFAULT_LOG_MAX_BYTES = 50_000_000
FAILURE_REPORT_INTERVAL = 100


class EventLogger:
    def __init__(
        self,
        log_dir: Path,
        agent_id: str | None = None,
        max_bytes: int = DEFAULT_LOG_MAX_BYTES,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.agent_id = agent_id
        self.max_bytes = max_bytes
        self._consecutive_failures = 0

    def event(self, kind: str, **fields: Any) -> None:
        payload: dict[str, Any] = {
            "ts": time.time(),
            "kind": kind,
            "agent_id": self.agent_id,
            **fields,
        }
        self._append(EVENT_LOG, payload)

    def llm_call(
        self,
        *,
        agent_id: str | None,
        model: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: int,
        error: str | None = None,
        note: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "ts": time.time(),
            "kind": "llm_call",
            "agent_id": agent_id or self.agent_id,
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "error": error,
            "note": note,
        }
        self._append(LLM_LOG, payload)

    def tool_call(
        self,
        *,
        agent_id: str | None,
        name: str,
        args: Any,
        result: Any = None,
        error: str | None = None,
        latency_ms: int | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "ts": time.time(),
            "kind": "tool_call",
            "agent_id": agent_id or self.agent_id,
            "name": name,
            "args": args,
            "result": result,
            "error": error,
            "latency_ms": latency_ms,
        }
        self._append(TOOL_LOG, payload)

    def close(self) -> None:
        return None

    def _rotate_if_needed(self, path: Path) -> None:
        try:
            if path.exists() and path.stat().st_size > self.max_bytes:
                path.replace(path.with_name(f"{path.name}.1"))
        except OSError:
            pass

    def _append(self, filename: str, payload: dict[str, Any]) -> None:
        handle: TextIO | None = None
        try:
            line = json.dumps(payload, default=str, ensure_ascii=False)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / filename
            self._rotate_if_needed(path)
            handle = path.open("a", encoding="utf-8")
            handle.write(line + "\n")
        except Exception as exc:
            self._consecutive_failures += 1
            try:
                print(f"EventLogger failed to write {filename}: {exc}", file=sys.stderr)
            except Exception:
                pass
            if (
                self._consecutive_failures == 1
                or self._consecutive_failures % FAILURE_REPORT_INTERVAL == 0
            ):
                try:
                    print(
                        "EventLogger ERROR: "
                        f"{self._consecutive_failures} consecutive write failure(s); "
                        f"last file {filename}: {exc}",
                        file=sys.stderr,
                    )
                except Exception:
                    pass
        else:
            self._consecutive_failures = 0
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
