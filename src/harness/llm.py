"""OpenAI-compatible async LLM client.

Talks to ``{llm_base_url}/chat/completions`` with httpx. Checks and updates the
daily token budget around every call and logs each call through EventLogger.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from . import models as m
from .budget import BudgetExceeded, BudgetGuard
from .config import Settings
from .logging_ import EventLogger

DEFAULT_TIMEOUT = 120.0
DEFAULT_TEMPERATURE = 0.7
MAX_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 0.5
ERROR_BODY_LIMIT = 2000


class LLMError(Exception):
    pass


class LLMBudgetError(LLMError):
    pass


def _truncate(raw: bytes, limit: int = ERROR_BODY_LIMIT) -> str:
    text = raw.decode("utf-8", errors="replace")
    if len(text) > limit:
        return f"{text[:limit]}... ({len(text)} chars total)"
    return text


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: m.Usage
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        logger: EventLogger | None = None,
        budget: BudgetGuard | None = None,
        agent_id: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger
        self.budget = budget
        self.agent_id = agent_id
        self._client = client
        self._owns_client = client is None

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        agent_id: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        resolved_agent = agent_id or self.agent_id

        if self.budget is not None and resolved_agent is not None:
            try:
                await self.budget.check_agent(resolved_agent)
                await self.budget.check_global()
            except BudgetExceeded as exc:
                raise LLMBudgetError(str(exc)) from exc

        body: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.turn_token_budget,
            "temperature": temperature if temperature is not None else DEFAULT_TEMPERATURE,
        }
        if tools is not None:
            body["tools"] = tools
        if response_format is not None:
            body["response_format"] = response_format

        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }

        client = await self._get_client()
        started = time.perf_counter()
        status = 0
        raw = b""
        retry_after: str | None = None
        last_error: str | None = None

        for attempt in range(MAX_ATTEMPTS):
            try:
                status, raw, retry_after = await self._send(client, url, headers, body)
            except LLMError as exc:
                self._log_llm(resolved_agent, 0, 0, started, error=str(exc))
                raise
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < MAX_ATTEMPTS:
                    await self._sleep_before_retry(attempt, None)
                    continue
                self._log_llm(resolved_agent, 0, 0, started, error=last_error)
                raise LLMError(last_error) from exc

            if 200 <= status < 300:
                break

            last_error = f"HTTP {status}: {_truncate(raw)}"
            retryable = status == 429 or (500 <= status < 600 and status != 501)
            if retryable and attempt + 1 < MAX_ATTEMPTS:
                await self._sleep_before_retry(attempt, retry_after)
                continue
            self._log_llm(resolved_agent, 0, 0, started, error=last_error)
            raise LLMError(last_error)

        text_body = raw.decode("utf-8", errors="replace")
        try:
            data = json.loads(text_body)
        except ValueError as exc:
            last_error = f"invalid JSON response: {exc}; body={_truncate(raw)}"
            self._log_llm(resolved_agent, 0, 0, started, error=last_error)
            raise LLMError(last_error) from exc

        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            last_error = f"missing choices in response: {data!r}"
            self._log_llm(resolved_agent, 0, 0, started, error=last_error)
            raise LLMError(last_error) from exc

        message = choice.get("message") or {}
        if not isinstance(message, dict):
            message = {}
        text = message.get("content") or ""
        raw_tool_calls = message.get("tool_calls")
        tool_calls: list[dict[str, Any]] = (
            raw_tool_calls if isinstance(raw_tool_calls, list) else []
        )
        finish_reason = choice.get("finish_reason")

        usage_raw = data.get("usage")
        usage_missing = not isinstance(usage_raw, dict)
        if usage_missing:
            usage = m.Usage()
        else:
            usage = m.Usage(
                tokens_in=int(usage_raw.get("prompt_tokens") or 0),
                tokens_out=int(usage_raw.get("completion_tokens") or 0),
            )
        model = data.get("model") or self.settings.llm_model

        if self.budget is not None and resolved_agent is not None:
            await self.budget.record(resolved_agent, usage.total)

        notes: list[str] = []
        if usage_missing:
            notes.append("usage missing")
        if finish_reason:
            notes.append(str(finish_reason))
        note = "; ".join(notes) if notes else None

        self._log_llm(
            resolved_agent,
            usage.tokens_in,
            usage.tokens_out,
            started,
            note=note,
            model=model,
        )

        return LLMResponse(
            text=text,
            model=model,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw=data,
        )

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
            self._owns_client = True
        return self._client

    async def _send(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> tuple[int, bytes, str | None]:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            status = response.status_code
            retry_after = response.headers.get("retry-after")
            chunks = bytearray()
            async for chunk in response.aiter_bytes():
                chunks.extend(chunk)
                if len(chunks) > self.settings.max_response_bytes:
                    raise LLMError(
                        "response body exceeded max_response_bytes "
                        f"({self.settings.max_response_bytes})"
                    )
            return status, bytes(chunks), retry_after

    async def _sleep_before_retry(self, attempt: int, retry_after: str | None) -> None:
        delay = RETRY_BACKOFF_BASE * (2 ** attempt)
        if retry_after is not None:
            try:
                requested = float(retry_after.strip())
            except (TypeError, ValueError):
                requested = None
            if requested is not None and requested > delay:
                delay = requested
        await asyncio.sleep(delay)

    def _log_llm(
        self,
        agent_id: str | None,
        tokens_in: int,
        tokens_out: int,
        started: float,
        error: str | None = None,
        note: str | None = None,
        model: str | None = None,
    ) -> None:
        if self.logger is None:
            return
        latency_ms = int((time.perf_counter() - started) * 1000)
        self.logger.llm_call(
            agent_id=agent_id,
            model=model or self.settings.llm_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            error=error,
            note=note,
        )
