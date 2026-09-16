"""Orchestrator: reconcile loop, worker pool, and turn execution.

The reconciler keeps ``settings.floor`` agents RUNNING while the global budget
allows, promotes WAITING agents while a slot is free, wakes PAUSED agents that
hold unread mail, and caps live agents at ``settings.hard_cap``.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from . import models as m
from .agent_runtime import (
    DEFAULT_ROLES,
    build_messages,
    clip_untrusted,
    escape_untrusted,
    tool_specs,
)
from .budget import BudgetExceeded, BudgetGuard
from .config import Settings
from .council.agents import MAX_TOOL_CALLS_PER_RESPONSE
from .council.executor import CombinedTools, build_combined_tools
from .containers import ContainerManager
from .db import Database
from .message_board import MessageBoard
from .llm import LLMClient
from .logging_ import EventLogger
from .mailbox import Mailbox
from .state_machine import InvalidTransition, StateMachine
from .summarizer import DIGEST_POST_LIMIT, summarize_topic

DEFAULT_INTERVAL = 1.0
MAX_LLM_CALLS = 8
GENERAL_TOPIC_TITLE = "general"
DIGEST_SCAN_LIMIT = 20
MAX_TOOL_RESULT_CHARS = 64 * 1024
MAX_LOG_PAYLOAD_CHARS = 2000
MAX_TRACKED_MAIL = 1000
MAX_CONTEXT_CHARS = 200_000
MAX_CONTEXT_MESSAGE_CHARS = 8_000
STALE_TURN_MAX_AGE_SECONDS = 3600.0
TRUNCATION_MARKER = " ...[truncated]"
UNTRUSTED_CODE_OPEN = "<untrusted_code>"
UNTRUSTED_CODE_CLOSE = "</untrusted_code>"
UNTRUSTED_CODE_OVERHEAD = len(UNTRUSTED_CODE_OPEN) + len(UNTRUSTED_CODE_CLOSE)


class SpawnRefused(Exception):
    pass


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(TRUNCATION_MARKER))
    return text[:keep] + TRUNCATION_MARKER


def _serialize_tool_result(result: dict[str, Any], used: int) -> tuple[str, int]:
    budget = MAX_TOOL_RESULT_CHARS - used
    available = budget - UNTRUSTED_CODE_OVERHEAD
    if available < 0:
        return "", used
    encoded = escape_untrusted(json.dumps(result, default=str))
    if len(encoded) > available:
        keep = max(0, available - len(TRUNCATION_MARKER))
        encoded = encoded[:keep] + TRUNCATION_MARKER
    content = f"{UNTRUSTED_CODE_OPEN}{encoded}{UNTRUSTED_CODE_CLOSE}"
    return content, used + len(content)


def _context_messages(messages: list[dict]) -> list[dict]:
    kept: list[dict] = []
    used = 0
    for message in messages:
        entry = dict(message)
        content = entry.get("content")
        if isinstance(content, str):
            entry["content"] = clip_untrusted(content, MAX_CONTEXT_MESSAGE_CHARS)
        encoded = json.dumps(entry, default=str)
        if used + len(encoded) > MAX_CONTEXT_CHARS:
            break
        kept.append(entry)
        used += len(encoded)
    return kept


def _log_payload(value: Any, limit: int = MAX_LOG_PAYLOAD_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return _truncate_text(text, limit)


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _tool_call_parts(
    call: Any, index: int
) -> tuple[str, str | None, dict[str, Any]]:
    if not isinstance(call, dict):
        return f"call_{index}", None, {}
    raw_id = call.get("id") or call.get("tool_call_id") or f"call_{index}"
    function = call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        raw_args = function.get("arguments")
    else:
        name = call.get("name")
        raw_args = call.get("arguments")
    resolved = name.strip() if isinstance(name, str) and name.strip() else None
    return str(raw_id), resolved, _parse_tool_arguments(raw_args)


async def ensure_topic(
    db: Database, title: str, owner_id: str | None = None
) -> m.Topic:
    existing = await db.topic_by_title(title)
    if existing is not None:
        return existing
    return await db.create_topic(title, owner_id=owner_id)


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        logger: EventLogger,
        containers: ContainerManager,
        mailbox: Mailbox,
        budget: BudgetGuard,
        state_machine: StateMachine,
        llm: LLMClient,
        message_board: MessageBoard,
        tools: CombinedTools | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.logger = logger
        self.containers = containers
        self.mailbox = mailbox
        self.budget = budget
        self.state_machine = state_machine
        self.llm = llm
        self.message_board = message_board
        self.tools = tools
        self.interval = DEFAULT_INTERVAL

        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._queued: set[str] = set()
        self._workers: list[asyncio.Task[None]] = []
        self._reconcile: asyncio.Task[None] | None = None
        self._stopping = False
        self._role_cursor = 0
        self._digest_cursor = 0
        self._slot_lock = asyncio.Lock()

    @property
    def stopping(self) -> bool:
        return self._stopping

    def _log(self, kind: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.event(kind, **fields)

    def _log_tool(
        self,
        agent_id: str,
        name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        error: str | None,
        latency_ms: int,
    ) -> None:
        if self.logger is None:
            return
        self.logger.tool_call(
            agent_id=agent_id,
            name=name,
            args=_log_payload(arguments),
            result=_log_payload(result),
            error=_log_payload(error) if error is not None else None,
            latency_ms=latency_ms,
        )

    async def _apply_tool_calls(
        self, agent_id: str, messages: list[dict], response: Any, used: int
    ) -> int:
        assert self.tools is not None
        calls = list(response.tool_calls)[:MAX_TOOL_CALLS_PER_RESPONSE]
        messages.append(
            {"role": "assistant", "content": response.text, "tool_calls": calls}
        )
        for index, call in enumerate(calls):
            call_id, name, arguments = _tool_call_parts(call, index)
            started = time.perf_counter()
            if name is None:
                result: dict[str, Any] = {
                    "ok": False,
                    "error": "tool call missing a function name",
                }
            else:
                try:
                    result = self.tools.dispatch(name, arguments)
                except Exception as exc:  # noqa: BLE001
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                if not isinstance(result, dict):
                    result = {"ok": False, "error": "invalid tool result"}
            latency_ms = int((time.perf_counter() - started) * 1000)
            error = None if result.get("ok") else str(result.get("error"))
            self._log_tool(agent_id, name or "", arguments, result, error, latency_ms)
            content, used = _serialize_tool_result(result, used)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }
            )
        return used

    def _next_role(self) -> m.AgentRole:
        role = DEFAULT_ROLES[self._role_cursor % len(DEFAULT_ROLES)]
        self._role_cursor += 1
        return role

    async def _enqueue(self, agent_id: str) -> None:
        if agent_id in self._queued:
            return
        self._queued.add(agent_id)
        self._queue.put_nowait(agent_id)

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self._queue.task_done()
        self._queued.clear()

    async def _running_count(self) -> int:
        return await self.db.count_agents(m.AgentState.RUNNING)

    async def _live_count(self) -> int:
        total = await self.db.count_agents()
        dead = await self.db.count_agents(m.AgentState.DEAD)
        return total - dead

    async def _ensure_container(self, agent_id: str) -> str | None:
        agent = await self.db.get_agent(agent_id)
        if agent is None:
            return None
        if agent.container_id is not None:
            return agent.container_id
        try:
            container = await self.containers.create(agent_id)
        except Exception as exc:
            await self.db.set_agent_state(agent_id, m.AgentState.DEAD)
            self._log(
                "spawn",
                action="container_failed",
                agent_id=agent_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        await self.db.set_agent_container(agent_id, container.id)
        self._log(
            "container",
            action="create",
            agent_id=agent_id,
            container_id=container.id,
        )
        return container.id

    async def _drop_container(self, agent_id: str) -> None:
        agent = await self.db.get_agent(agent_id)
        if agent is None or agent.container_id is None:
            return
        container_id = agent.container_id
        await self.db.set_agent_container(agent_id, None)
        try:
            await self.containers.remove(container_id)
        except Exception as exc:
            self._log(
                "container_error",
                agent_id=agent_id,
                container_id=container_id,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def spawn_agent(self, role: m.AgentRole) -> m.Agent:
        async with self._slot_lock:
            await self.budget.check_global()
            live = await self._live_count()
            if live >= self.settings.hard_cap:
                raise SpawnRefused(
                    f"live agent cap reached ({live}/{self.settings.hard_cap})"
                )

            agent = await self.db.create_agent(role, state=m.AgentState.SPAWNED)

            topic = await ensure_topic(self.db, GENERAL_TOPIC_TITLE)
            await self.db.subscribe(agent.id, topic.id)
            await self.db.set_agent_topic(agent.id, topic.id)

            running = await self._running_count()
            if running < self.settings.floor:
                reserved = await self.db.transition_state(
                    agent.id, m.AgentState.SPAWNED, m.AgentState.RUNNING
                )
                if reserved:
                    await self._ensure_container(agent.id)
                    await self.db.touch_agent_wake(agent.id)
                    await self._enqueue(agent.id)
                else:
                    await self.db.transition_state(
                        agent.id, m.AgentState.SPAWNED, m.AgentState.WAITING
                    )
            else:
                await self.db.transition_state(
                    agent.id, m.AgentState.SPAWNED, m.AgentState.WAITING
                )

        refreshed = await self.db.get_agent(agent.id)
        assert refreshed is not None
        self._log(
            "spawn",
            action="done",
            agent_id=refreshed.id,
            role=refreshed.role.value,
            state=refreshed.state.value,
        )
        return refreshed

    async def request_wake(self, agent_id: str) -> None:
        agent = await self.db.get_agent(agent_id)
        if agent is None:
            return
        if agent.state not in (m.AgentState.PAUSED, m.AgentState.WAITING):
            return

        async with self._slot_lock:
            running = await self._running_count()
            if running < self.settings.hard_cap:
                try:
                    await self.state_machine.transition(
                        agent_id, m.AgentState.RUNNING, reason="wake"
                    )
                except InvalidTransition:
                    return
                try:
                    await self._ensure_container(agent_id)
                except Exception:
                    return
                await self.db.touch_agent_wake(agent_id)
                await self._enqueue(agent_id)
                self._log("wake", agent_id=agent_id, state=m.AgentState.RUNNING.value)
            elif agent.state is not m.AgentState.WAITING:
                try:
                    await self.state_machine.transition(
                        agent_id, m.AgentState.WAITING, reason="wake:no_slot"
                    )
                except InvalidTransition:
                    return
                await self._drop_container(agent_id)
                self._log("wake", agent_id=agent_id, state=m.AgentState.WAITING.value)

    async def tick(self) -> None:
        budget_ok = True
        try:
            await self.budget.check_global()
        except BudgetExceeded as exc:
            budget_ok = False
            self._log("tick_budget_block", error=str(exc))

        running = await self.db.list_agents(m.AgentState.RUNNING)
        if len(running) > self.settings.hard_cap:
            for agent in reversed(running[self.settings.hard_cap:]):
                await self._shelve(agent.id)
            running = await self.db.list_agents(m.AgentState.RUNNING)

        if not budget_ok:
            self._drain_queue()
            return

        await self._maybe_digest()

        for agent in await self.db.list_agents(m.AgentState.WAITING):
            if len(running) >= self.settings.hard_cap:
                break
            async with self._slot_lock:
                if await self._running_count() >= self.settings.hard_cap:
                    break
                try:
                    await self.state_machine.transition(
                        agent.id, m.AgentState.RUNNING, reason="promote"
                    )
                except InvalidTransition:
                    continue
                try:
                    await self._ensure_container(agent.id)
                except Exception:
                    continue
                await self.db.touch_agent_wake(agent.id)
                await self._enqueue(agent.id)
            running.append(agent)
            self._log("promote", agent_id=agent.id, reason="promote")

        for agent in await self.db.list_agents(m.AgentState.PAUSED):
            if len(running) >= self.settings.hard_cap:
                break
            if not await self.mailbox.has_unread(agent.id):
                continue
            async with self._slot_lock:
                if await self._running_count() >= self.settings.hard_cap:
                    break
                try:
                    await self.state_machine.transition(
                        agent.id, m.AgentState.RUNNING, reason="mail"
                    )
                except InvalidTransition:
                    continue
                try:
                    await self._ensure_container(agent.id)
                except Exception:
                    continue
                await self.db.touch_agent_wake(agent.id)
                await self._enqueue(agent.id)
            running.append(agent)
            self._log("resume", agent_id=agent.id, reason="mail")

        while len(running) < self.settings.floor:
            if await self._live_count() >= self.settings.hard_cap:
                break
            if not await self.budget.can_spawn():
                break
            try:
                agent = await self.spawn_agent(self._next_role())
            except (BudgetExceeded, SpawnRefused):
                break
            if agent.state is not m.AgentState.RUNNING:
                break
            running.append(agent)

    async def _maybe_digest(self) -> None:
        threshold = max(
            1, min(self.settings.digest_every_n_posts, DIGEST_POST_LIMIT)
        )
        topics = await self.db.list_topics()
        if not topics:
            self._digest_cursor = 0
            return
        count = len(topics)
        if self._digest_cursor >= count:
            self._digest_cursor = 0
        scan = min(count, DIGEST_SCAN_LIMIT)
        for offset in range(scan):
            index = (self._digest_cursor + offset) % count
            topic = topics[index]
            pending = await self.db.list_posts(
                topic.id, since_id=topic.last_digest_post_id, limit=threshold
            )
            if len(pending) < threshold:
                continue
            try:
                await summarize_topic(
                    self.db,
                    self.llm,
                    topic.id,
                    every_n=threshold,
                    logger=self.logger,
                    agent_id="summarizer",
                )
            except Exception as exc:
                self._log(
                    "digest_error",
                    topic_id=topic.id,
                    error=f"{type(exc).__name__}: {exc}",
                )
            self._digest_cursor = (index + 1) % count
            return
        self._digest_cursor = (self._digest_cursor + scan) % count

    async def _shelve(self, agent_id: str) -> None:
        try:
            await self.state_machine.transition(
                agent_id, m.AgentState.WAITING, reason="over_cap"
            )
        except InvalidTransition:
            return
        await self._drop_container(agent_id)
        self._log("shelve", agent_id=agent_id)

    async def _known_unread_ids(self, agent_id: str) -> set[int]:
        messages = await self.db.unread_mail(agent_id, limit=MAX_TRACKED_MAIL)
        return {message.id for message in messages if message.id is not None}

    async def run_turn(self, agent_id: str) -> None:
        if await self.db.running_turn(agent_id) is not None:
            self._log("turn_skip", agent_id=agent_id, reason="turn_in_progress")
            return

        agent = await self.db.get_agent(agent_id)
        if agent is None or agent.state is not m.AgentState.RUNNING:
            return

        container_id = agent.container_id
        turn = await self.db.start_turn(agent_id)
        tokens_in = 0
        tokens_out = 0
        messages: list[dict] = []
        tool_result_chars = 0
        final = m.TurnState.DONE
        reason: str | None = None

        try:
            try:
                if container_id is not None:
                    status = await self.containers.status(container_id)
                    if status is not m.ContainerStatus.RUNNING:
                        await self.containers.start(container_id)

                topic = (
                    await self.db.get_topic(agent.topic_id)
                    if agent.topic_id is not None
                    else None
                )
                digest = (
                    await self.db.latest_digest(agent.topic_id)
                    if agent.topic_id is not None
                    else None
                )
                seeded = False
                known_mail_ids: set[int] = set()
                for _ in range(MAX_LLM_CALLS):
                    if self.tools is None or not seeded:
                        unread = await self.mailbox.drain(agent_id)
                        posts = (
                            await self.db.list_posts_recent(agent.topic_id)
                            if agent.topic_id is not None
                            else []
                        )
                        messages = build_messages(
                            agent, unread, posts, topic, digest=digest
                        )
                        known_mail_ids = {
                            message.id
                            for message in unread
                            if message.id is not None
                        }
                        known_mail_ids |= await self._known_unread_ids(agent_id)
                        seeded = True
                    spent = tokens_in + tokens_out
                    remaining = self.settings.turn_token_budget - spent
                    response = await self.llm.complete(
                        messages,
                        tools=(
                            self.tools.specs()
                            if self.tools is not None
                            else tool_specs()
                        ),
                        max_tokens=max(1, remaining),
                        agent_id=agent_id,
                    )
                    tokens_in += response.usage.tokens_in
                    tokens_out += response.usage.tokens_out

                    if tokens_in + tokens_out >= self.settings.turn_token_budget:
                        final = m.TurnState.BUDGET_EXCEEDED
                        reason = "turn_token_budget"
                        break
                    if response.tool_calls:
                        if self.tools is not None:
                            tool_result_chars = await self._apply_tool_calls(
                                agent_id, messages, response, tool_result_chars
                            )
                            continue
                    new_mail_ids = await self._known_unread_ids(agent_id)
                    if new_mail_ids - known_mail_ids:
                        seeded = False
                        continue
                    break
            except Exception as exc:
                final = m.TurnState.FAILED
                reason = f"{type(exc).__name__}: {exc}"
                self._log("turn_error", agent_id=agent_id, error=reason)
            else:
                try:
                    await self._save_context(
                        agent, messages, turn.id, tokens_in, tokens_out
                    )
                except Exception as exc:
                    final = m.TurnState.FAILED
                    reason = f"{type(exc).__name__}: {exc}"
                    self._log("turn_error", agent_id=agent_id, error=reason)
        except asyncio.CancelledError:
            final = m.TurnState.FAILED
            reason = "cancelled"
            raise
        finally:
            cleanup = asyncio.ensure_future(
                self._finish_turn(
                    agent_id,
                    container_id,
                    turn.id,
                    final,
                    tokens_in,
                    tokens_out,
                    reason,
                )
            )
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise

    async def _finish_turn(
        self,
        agent_id: str,
        container_id: str | None,
        turn_id: int | None,
        final: m.TurnState,
        tokens_in: int,
        tokens_out: int,
        reason: str | None,
    ) -> None:
        if container_id is not None:
            try:
                await self.containers.stop(container_id)
            except Exception as exc:
                self._log(
                    "turn_error",
                    agent_id=agent_id,
                    error=f"container_stop: {type(exc).__name__}: {exc}",
                )
        await self.db.end_turn(turn_id, final, tokens_in, tokens_out, reason)
        await self._pause(agent_id)
        self._log(
            "turn",
            agent_id=agent_id,
            turn_id=turn_id,
            state=final.value,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    async def _save_context(
        self,
        agent: m.Agent,
        messages: list[dict],
        turn_id: int | None,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        context_dir = self.settings.workspace_dir / agent.id
        context_dir.mkdir(parents=True, exist_ok=True)
        path = context_dir / "context.json"
        payload = {
            "agent_id": agent.id,
            "role": agent.role.value,
            "topic_id": agent.topic_id,
            "turn_id": turn_id,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "messages": _context_messages(messages),
        }
        tmp_path = context_dir / f".{path.name}.tmp"
        tmp_path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        os.replace(tmp_path, path)
        await self.db.set_agent_context_path(agent.id, str(path))

    async def _pause(self, agent_id: str) -> None:
        agent = await self.db.get_agent(agent_id)
        if agent is None or agent.state is not m.AgentState.RUNNING:
            return
        try:
            await self.state_machine.transition(
                agent_id, m.AgentState.PAUSED, reason="turn_end"
            )
        except InvalidTransition:
            return

    async def _worker(self) -> None:
        while True:
            agent_id = await self._queue.get()
            try:
                await self.run_turn(agent_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log(
                    "worker_error",
                    agent_id=agent_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                self._queued.discard(agent_id)
                self._queue.task_done()

    async def start(
        self, stale_max_age_seconds: float = STALE_TURN_MAX_AGE_SECONDS
    ) -> None:
        self._stopping = False
        stale_agents = await self.db.close_stale_turns(stale_max_age_seconds)
        for agent_id in stale_agents:
            agent = await self.db.get_agent(agent_id)
            if agent is None or agent.state is m.AgentState.DEAD:
                continue
            try:
                await self.state_machine.transition(
                    agent_id, m.AgentState.PAUSED, reason="stale_turn"
                )
            except InvalidTransition:
                continue
            self._log("stale_turn", agent_id=agent_id)
        count = max(1, self.settings.worker_pool_size)
        self._workers = [
            asyncio.create_task(self._worker()) for _ in range(count)
        ]
        self._reconcile = asyncio.create_task(self.run_forever(self.interval))

    async def stop(self) -> None:
        self._stopping = True
        tasks: list[asyncio.Task[None]] = list(self._workers)
        if self._reconcile is not None:
            tasks.append(self._reconcile)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._workers = []
        self._reconcile = None
        self._drain_queue()
        for agent in await self.db.list_agents(m.AgentState.RUNNING):
            orphan = await self.db.running_turn(agent.id)
            if orphan is not None and orphan.id is not None:
                await self.db.end_turn(
                    orphan.id, m.TurnState.FAILED, reason="stopped"
                )
            await self._pause(agent.id)

    async def run_forever(self, interval: float = 1.0) -> None:
        while not self._stopping:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log(
                    "tick_error", error=f"{type(exc).__name__}: {exc}"
                )
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise

    async def close(self) -> None:
        await self.containers.aclose()
        await self.llm.aclose()
        await self.db.close()

    async def aclose(self) -> None:
        await self.stop()
        await self.close()


async def build_orchestrator(
    settings: Settings, checkout: str | None = None
) -> Orchestrator:
    settings.ensure_dirs()
    db = Database(settings.db_path)
    await db.connect()
    await db.init_schema()
    logger = EventLogger(settings.log_dir, max_bytes=settings.log_max_bytes)
    budget = BudgetGuard(db, settings, logger=logger)
    state_machine = StateMachine(db, logger=logger)
    mailbox = Mailbox(db, logger=logger)
    containers = ContainerManager(settings, db, logger=logger)
    llm = LLMClient(settings, logger=logger, budget=budget)
    message_board = MessageBoard(db, mailbox, settings, logger=logger)
    tools = build_combined_tools(checkout) if checkout is not None else None
    return Orchestrator(
        settings,
        db,
        logger,
        containers,
        mailbox,
        budget,
        state_machine,
        llm,
        message_board,
        tools=tools,
    )
