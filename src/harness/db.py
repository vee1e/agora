"""Async SQLite access layer.

All persistence goes through this class. One connection, one writer, WAL mode.
Rows come back as dataclasses from ``harness.models``.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

import aiosqlite

from . import models as m

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _now() -> float:
    return time.time()


def new_id(prefix: str = "") -> str:
    token = uuid.uuid4().hex[:12]
    return f"{prefix}{token}" if prefix else token


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None
        self._tx_lock = asyncio.Lock()

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database.connect() has not been called")
        return self._db

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode = WAL")
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.execute("PRAGMA recursive_triggers = ON")
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def init_schema(self) -> None:
        sql = SCHEMA_PATH.read_text()
        await self.conn.executescript(sql)
        await self.conn.commit()

    async def _fetchone(self, sql: str, params: Sequence[Any] = ()) -> Any:
        cur = await self.conn.execute(sql, params)
        row = await cur.fetchone()
        await cur.close()
        return row

    async def _fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[Any]:
        cur = await self.conn.execute(sql, params)
        rows = await cur.fetchall()
        await cur.close()
        return list(rows)

    async def _write(self, sql: str, params: Sequence[Any] = ()) -> tuple[int, int | None]:
        async with self._tx_lock:
            cur = await self.conn.execute(sql, params)
            await self.conn.commit()
            rowcount = cur.rowcount
            lastrowid = cur.lastrowid
            await cur.close()
            return rowcount, lastrowid

    async def _execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        rowcount, _ = await self._write(sql, params)
        return rowcount

    # ----- agents -----

    async def create_agent(
        self,
        role: m.AgentRole,
        state: m.AgentState = m.AgentState.SPAWNED,
        agent_id: str | None = None,
        topic_id: str | None = None,
    ) -> m.Agent:
        agent_id = agent_id or new_id("ag_")
        ts = _now()
        await self._execute(
            """INSERT INTO agents (id, role, state, container_id, topic_id,
                                   context_path, created_at, updated_at, last_wake_at)
               VALUES (?, ?, ?, NULL, ?, NULL, ?, ?, ?)""",
            (agent_id, role.value, state.value, topic_id, ts, ts, ts),
        )
        agent = await self.get_agent(agent_id)
        assert agent is not None
        return agent

    async def get_agent(self, agent_id: str) -> m.Agent | None:
        row = await self._fetchone("SELECT * FROM agents WHERE id = ?", (agent_id,))
        return m.Agent.from_row(row) if row else None

    async def list_agents(
        self, state: m.AgentState | None = None, limit: int | None = None
    ) -> list[m.Agent]:
        if state is None:
            rows = await self._fetchall(
                "SELECT * FROM agents ORDER BY created_at LIMIT ?", (limit or -1,)
            )
        else:
            rows = await self._fetchall(
                "SELECT * FROM agents WHERE state = ? ORDER BY created_at LIMIT ?",
                (state.value, limit or -1),
            )
        return [m.Agent.from_row(r) for r in rows]

    async def count_agents(self, state: m.AgentState | None = None) -> int:
        if state is None:
            row = await self._fetchone("SELECT COUNT(*) AS n FROM agents")
        else:
            row = await self._fetchone(
                "SELECT COUNT(*) AS n FROM agents WHERE state = ?", (state.value,)
            )
        return int(row["n"]) if row else 0

    async def set_agent_state(self, agent_id: str, state: m.AgentState) -> None:
        await self._execute(
            "UPDATE agents SET state = ?, updated_at = ? WHERE id = ?",
            (state.value, _now(), agent_id),
        )

    async def transition_state(
        self, agent_id: str, expected: m.AgentState, dst: m.AgentState
    ) -> bool:
        """Atomic compare-and-set. Returns True only if the row was in `expected`."""
        rowcount, _ = await self._write(
            "UPDATE agents SET state = ?, updated_at = ? WHERE id = ? AND state = ?",
            (dst.value, _now(), agent_id, expected.value),
        )
        return rowcount == 1

    async def close_stale_turns(self, max_age_seconds: float = 3600.0) -> list[str]:
        """Fail running turns older than the threshold and return their agent ids."""
        cutoff = _now() - max_age_seconds
        rows = await self._fetchall(
            "SELECT DISTINCT agent_id FROM turns WHERE ended_at IS NULL AND started_at < ?",
            (cutoff,),
        )
        agents = [r["agent_id"] for r in rows]
        if agents:
            await self._execute(
                """UPDATE turns SET ended_at = ?, state = 'failed',
                          reason = 'stale turn closed on startup'
                   WHERE ended_at IS NULL AND started_at < ?""",
                (_now(), cutoff),
            )
        return agents

    async def set_agent_container(self, agent_id: str, container_id: str | None) -> None:
        await self._execute(
            "UPDATE agents SET container_id = ?, updated_at = ? WHERE id = ?",
            (container_id, _now(), agent_id),
        )

    async def set_agent_context_path(self, agent_id: str, path: str) -> None:
        await self._execute(
            "UPDATE agents SET context_path = ?, updated_at = ? WHERE id = ?",
            (path, _now(), agent_id),
        )

    async def set_agent_topic(self, agent_id: str, topic_id: str | None) -> None:
        await self._execute(
            "UPDATE agents SET topic_id = ?, updated_at = ? WHERE id = ?",
            (topic_id, _now(), agent_id),
        )

    async def touch_agent_wake(self, agent_id: str) -> None:
        await self._execute(
            "UPDATE agents SET last_wake_at = ?, updated_at = ? WHERE id = ?",
            (_now(), _now(), agent_id),
        )

    # ----- turns -----

    async def start_turn(self, agent_id: str) -> m.Turn:
        ts = _now()
        _, turn_id = await self._write(
            "INSERT INTO turns (agent_id, started_at, state) VALUES (?, ?, ?)",
            (agent_id, ts, m.TurnState.RUNNING.value),
        )
        return m.Turn(id=turn_id, agent_id=agent_id, started_at=ts)

    async def end_turn(
        self,
        turn_id: int,
        state: m.TurnState,
        tokens_in: int = 0,
        tokens_out: int = 0,
        reason: str | None = None,
    ) -> None:
        await self._execute(
            """UPDATE turns SET ended_at = ?, state = ?, tokens_in = ?, tokens_out = ?,
                               reason = ? WHERE id = ?""",
            (_now(), state.value, tokens_in, tokens_out, reason, turn_id),
        )

    async def get_turn(self, turn_id: int) -> m.Turn | None:
        row = await self._fetchone("SELECT * FROM turns WHERE id = ?", (turn_id,))
        return m.Turn.from_row(row) if row else None

    async def list_turns(self, agent_id: str, limit: int = 50) -> list[m.Turn]:
        rows = await self._fetchall(
            "SELECT * FROM turns WHERE agent_id = ? ORDER BY id DESC LIMIT ?",
            (agent_id, limit),
        )
        return [m.Turn.from_row(r) for r in rows]

    async def running_turn(self, agent_id: str) -> m.Turn | None:
        row = await self._fetchone(
            "SELECT * FROM turns WHERE agent_id = ? AND ended_at IS NULL ORDER BY id DESC LIMIT 1",
            (agent_id,),
        )
        return m.Turn.from_row(row) if row else None

    # ----- containers -----

    async def upsert_container(
        self,
        container_id: str,
        agent_id: str | None,
        image: str,
        workspace_path: str,
        status: m.ContainerStatus,
    ) -> None:
        ts = _now()
        await self._execute(
            """INSERT INTO containers (id, agent_id, image, workspace_path, status,
                                       created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 agent_id = excluded.agent_id,
                 image = excluded.image,
                 workspace_path = excluded.workspace_path,
                 status = excluded.status,
                 updated_at = excluded.updated_at""",
            (container_id, agent_id, image, workspace_path, status.value, ts, ts),
        )

    async def set_container_status(self, container_id: str, status: m.ContainerStatus) -> None:
        await self._execute(
            "UPDATE containers SET status = ?, updated_at = ? WHERE id = ?",
            (status.value, _now(), container_id),
        )

    async def get_container(self, container_id: str) -> m.Container | None:
        row = await self._fetchone("SELECT * FROM containers WHERE id = ?", (container_id,))
        return m.Container.from_row(row) if row else None

    async def list_containers(self, status: m.ContainerStatus | None = None) -> list[m.Container]:
        if status is None:
            rows = await self._fetchall("SELECT * FROM containers ORDER BY created_at")
        else:
            rows = await self._fetchall(
                "SELECT * FROM containers WHERE status = ? ORDER BY created_at", (status.value,)
            )
        return [m.Container.from_row(r) for r in rows]

    # ----- mailbox -----

    async def send_mail(
        self,
        agent_id: str,
        body: str,
        sender_id: str | None = None,
        topic_id: str | None = None,
        post_id: int | None = None,
    ) -> m.MailMessage:
        ts = _now()
        _, mail_id = await self._write(
            """INSERT INTO mailbox (agent_id, sender_id, topic_id, post_id, body, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (agent_id, sender_id, topic_id, post_id, body, ts),
        )
        return m.MailMessage(
            id=mail_id,
            agent_id=agent_id,
            sender_id=sender_id,
            topic_id=topic_id,
            post_id=post_id,
            body=body,
            created_at=ts,
        )

    async def unread_mail(self, agent_id: str, limit: int = 50) -> list[m.MailMessage]:
        rows = await self._fetchall(
            "SELECT * FROM mailbox WHERE agent_id = ? AND read_at IS NULL ORDER BY id LIMIT ?",
            (agent_id, limit),
        )
        return [m.MailMessage.from_row(r) for r in rows]

    async def drain_mail(self, agent_id: str, limit: int = 50) -> list[m.MailMessage]:
        """Read unread mail and mark it read in one transaction."""
        ts = _now()
        rows: list[Any] = []
        async with self._tx_lock:
            try:
                await self.conn.execute("BEGIN")
                cur = await self.conn.execute(
                    """SELECT * FROM mailbox WHERE agent_id = ? AND read_at IS NULL
                       ORDER BY id LIMIT ?""",
                    (agent_id, limit),
                )
                rows = list(await cur.fetchall())
                await cur.close()
                if rows:
                    ids = [r["id"] for r in rows]
                    placeholders = ",".join("?" for _ in ids)
                    await self.conn.execute(
                        f"UPDATE mailbox SET read_at = ? WHERE id IN ({placeholders})",
                        (ts, *ids),
                    )
                await self.conn.commit()
            except BaseException:
                await self.conn.rollback()
                raise
        return [m.MailMessage.from_row(r) for r in rows]

    async def count_unread(self, agent_id: str) -> int:
        row = await self._fetchone(
            "SELECT COUNT(*) AS n FROM mailbox WHERE agent_id = ? AND read_at IS NULL",
            (agent_id,),
        )
        return int(row["n"]) if row else 0

    async def mark_mail_read(self, mail_ids: Iterable[int]) -> None:
        ids = list(mail_ids)
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        await self._execute(
            f"UPDATE mailbox SET read_at = ? WHERE id IN ({placeholders})",
            (_now(), *ids),
        )

    # ----- budget -----

    async def get_budget(self, scope: str, key: str, day: str) -> m.BudgetRow | None:
        row = await self._fetchone(
            "SELECT * FROM budget WHERE scope = ? AND key = ? AND day = ?",
            (scope, key, day),
        )
        return m.BudgetRow.from_row(row) if row else None

    async def add_tokens(
        self,
        scope: str,
        key: str,
        day: str,
        tokens: int,
        cap: int,
    ) -> m.BudgetRow:
        ts = _now()
        await self._execute(
            """INSERT INTO budget (scope, key, day, tokens_used, cap, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(scope, key, day) DO UPDATE SET
                 tokens_used = budget.tokens_used + excluded.tokens_used,
                 cap = excluded.cap,
                 updated_at = excluded.updated_at""",
            (scope, key, day, tokens, cap, ts),
        )
        row = await self.get_budget(scope, key, day)
        assert row is not None
        return row

    async def add_tokens_many(
        self, entries: Sequence[tuple[str, str, str, int, int]]
    ) -> None:
        """Apply several budget increments in one transaction.

        Each entry is (scope, key, day, tokens, cap).
        """
        ts = _now()
        async with self._tx_lock:
            try:
                await self.conn.execute("BEGIN")
                for scope, key, day, tokens, cap in entries:
                    await self.conn.execute(
                        """INSERT INTO budget (scope, key, day, tokens_used, cap, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(scope, key, day) DO UPDATE SET
                             tokens_used = budget.tokens_used + excluded.tokens_used,
                             cap = excluded.cap,
                             updated_at = excluded.updated_at""",
                        (scope, key, day, tokens, cap, ts),
                    )
                await self.conn.commit()
            except BaseException:
                await self.conn.rollback()
                raise

    # ----- topics -----

    async def create_topic(
        self, title: str, owner_id: str | None = None, topic_id: str | None = None
    ) -> m.Topic:
        existing = await self.topic_by_title(title)
        if existing is not None:
            return existing
        topic_id = topic_id or new_id("tp_")
        ts = _now()
        await self._execute(
            """INSERT INTO topics (id, title, owner_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(title) DO NOTHING""",
            (topic_id, title, owner_id, ts, ts),
        )
        topic = await self.topic_by_title(title)
        assert topic is not None
        return topic

    async def topic_by_title(self, title: str) -> m.Topic | None:
        row = await self._fetchone("SELECT * FROM topics WHERE title = ?", (title,))
        return m.Topic.from_row(row) if row else None

    async def get_topic(self, topic_id: str) -> m.Topic | None:
        row = await self._fetchone("SELECT * FROM topics WHERE id = ?", (topic_id,))
        return m.Topic.from_row(row) if row else None

    async def list_topics(self) -> list[m.Topic]:
        rows = await self._fetchall("SELECT * FROM topics ORDER BY updated_at DESC")
        return [m.Topic.from_row(r) for r in rows]

    async def set_topic_owner(self, topic_id: str, owner_id: str | None) -> None:
        await self._execute(
            "UPDATE topics SET owner_id = ?, updated_at = ? WHERE id = ?",
            (owner_id, _now(), topic_id),
        )

    async def set_topic_owner_cas(
        self, topic_id: str, from_agent: str, to_agent: str
    ) -> bool:
        """Atomically hand ownership from `from_agent` to `to_agent`."""
        rowcount, _ = await self._write(
            "UPDATE topics SET owner_id = ?, updated_at = ? WHERE id = ? AND owner_id = ?",
            (to_agent, _now(), topic_id, from_agent),
        )
        return rowcount == 1

    async def bump_topic(self, topic_id: str, by: int = 1) -> None:
        await self._execute(
            """UPDATE topics SET post_count = post_count + ?, updated_at = ?
               WHERE id = ?""",
            (by, _now(), topic_id),
        )

    async def set_topic_digest_mark(self, topic_id: str, post_id: int) -> None:
        await self._execute(
            "UPDATE topics SET last_digest_post_id = ?, updated_at = ? WHERE id = ?",
            (post_id, _now(), topic_id),
        )

    # ----- posts -----

    async def create_post(
        self,
        agent_id: str,
        topic_id: str,
        body: str,
        parent_id: int | None = None,
        refs: list[dict[str, Any]] | None = None,
    ) -> m.Post:
        ts = _now()
        payload = json.dumps(refs or [])
        _, post_id = await self._write(
            """INSERT INTO posts (agent_id, time, topic_id, parent_id, body, refs)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (agent_id, ts, topic_id, parent_id, body, payload),
        )
        return m.Post(
            id=post_id,
            agent_id=agent_id,
            time=ts,
            topic_id=topic_id,
            parent_id=parent_id,
            body=body,
            refs=refs or [],
        )

    async def insert_post_limited(
        self,
        agent_id: str,
        topic_id: str,
        body: str,
        parent_id: int | None,
        refs: list[dict[str, Any]],
        window_start: float,
        limit: int,
    ) -> int | None:
        """Insert a post only if the agent is under its rate limit.

        The count and the insert are one statement, so concurrent posts cannot
        slip past the limit. Returns the new post id, or None when rate limited.
        """
        ts = _now()
        rowcount, post_id = await self._write(
            """INSERT INTO posts (agent_id, time, topic_id, parent_id, body, refs)
               SELECT ?, ?, ?, ?, ?, ?
               WHERE (SELECT COUNT(*) FROM posts WHERE agent_id = ? AND time >= ?) < ?""",
            (
                agent_id,
                ts,
                topic_id,
                parent_id,
                body,
                json.dumps(refs),
                agent_id,
                window_start,
                limit,
            ),
        )
        return post_id if rowcount == 1 else None

    async def get_post(self, post_id: int) -> m.Post | None:
        row = await self._fetchone("SELECT * FROM posts WHERE id = ?", (post_id,))
        return m.Post.from_row(row) if row else None

    async def list_posts(
        self, topic_id: str, since_id: int = 0, limit: int = 100
    ) -> list[m.Post]:
        rows = await self._fetchall(
            "SELECT * FROM posts WHERE topic_id = ? AND id > ? ORDER BY id LIMIT ?",
            (topic_id, since_id, limit),
        )
        return [m.Post.from_row(r) for r in rows]

    async def list_posts_recent(self, topic_id: str, limit: int = 100) -> list[m.Post]:
        """Newest posts first, returned oldest-to-newest for prompt assembly."""
        rows = await self._fetchall(
            "SELECT * FROM posts WHERE topic_id = ? ORDER BY id DESC LIMIT ?",
            (topic_id, limit),
        )
        return [m.Post.from_row(r) for r in reversed(rows)]

    async def count_posts_since(self, agent_id: str, since: float) -> int:
        row = await self._fetchone(
            "SELECT COUNT(*) AS n FROM posts WHERE agent_id = ? AND time >= ?",
            (agent_id, since),
        )
        return int(row["n"]) if row else 0

    async def last_post_id(self, topic_id: str) -> int:
        row = await self._fetchone(
            "SELECT MAX(id) AS n FROM posts WHERE topic_id = ?", (topic_id,)
        )
        return int(row["n"]) if row and row["n"] is not None else 0

    # ----- subscriptions -----

    async def subscribe(self, agent_id: str, topic_id: str) -> None:
        await self._execute(
            """INSERT INTO subscriptions (agent_id, topic_id, created_at)
               VALUES (?, ?, ?)
               ON CONFLICT(agent_id, topic_id) DO NOTHING""",
            (agent_id, topic_id, _now()),
        )

    async def unsubscribe(self, agent_id: str, topic_id: str) -> None:
        await self._execute(
            "DELETE FROM subscriptions WHERE agent_id = ? AND topic_id = ?",
            (agent_id, topic_id),
        )

    async def list_subscribers(self, topic_id: str) -> list[str]:
        rows = await self._fetchall(
            "SELECT agent_id FROM subscriptions WHERE topic_id = ?", (topic_id,)
        )
        return [r["agent_id"] for r in rows]

    async def list_subscriptions(self, agent_id: str) -> list[str]:
        rows = await self._fetchall(
            "SELECT topic_id FROM subscriptions WHERE agent_id = ?", (agent_id,)
        )
        return [r["topic_id"] for r in rows]

    # ----- digests -----

    async def add_digest(
        self, topic_id: str, from_post_id: int, to_post_id: int, body: str
    ) -> None:
        await self._execute(
            """INSERT INTO digests (topic_id, from_post_id, to_post_id, body, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (topic_id, from_post_id, to_post_id, body, _now()),
        )

    async def add_digest_and_mark(
        self,
        topic_id: str,
        from_post_id: int,
        to_post_id: int,
        body: str,
    ) -> None:
        """Insert the digest and advance the topic mark in one transaction."""
        ts = _now()
        async with self._tx_lock:
            try:
                await self.conn.execute("BEGIN")
                await self.conn.execute(
                    """INSERT INTO digests (topic_id, from_post_id, to_post_id, body, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (topic_id, from_post_id, to_post_id, body, ts),
                )
                await self.conn.execute(
                    "UPDATE topics SET last_digest_post_id = ?, updated_at = ? WHERE id = ?",
                    (to_post_id, ts, topic_id),
                )
                await self.conn.commit()
            except BaseException:
                await self.conn.rollback()
                raise

    async def latest_digest(self, topic_id: str) -> dict[str, Any] | None:
        row = await self._fetchone(
            "SELECT * FROM digests WHERE topic_id = ? ORDER BY id DESC LIMIT 1",
            (topic_id,),
        )
        return dict(row) if row else None

    # ----- findings -----

    async def create_finding(
        self,
        topic_id: str,
        agent_id: str,
        file: str,
        line: int,
        sink: str,
        call_path: list[str] | None = None,
        detail: str | None = None,
        finding_id: str | None = None,
    ) -> m.Finding:
        finding_id = finding_id or new_id("fd_")
        await self._execute(
            """INSERT INTO findings (id, topic_id, agent_id, file, line, sink,
                                     call_path, status, detail, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                finding_id,
                topic_id,
                agent_id,
                file,
                line,
                sink,
                json.dumps(call_path or []),
                m.FindingStatus.CANDIDATE.value,
                detail,
                _now(),
            ),
        )
        finding = await self.get_finding(finding_id)
        assert finding is not None
        return finding

    async def get_finding(self, finding_id: str) -> m.Finding | None:
        row = await self._fetchone("SELECT * FROM findings WHERE id = ?", (finding_id,))
        return m.Finding.from_row(row) if row else None

    async def list_findings(
        self,
        topic_id: str | None = None,
        status: m.FindingStatus | None = None,
        limit: int = 1000,
    ) -> list[m.Finding]:
        clauses: list[str] = []
        params: list[Any] = []
        if topic_id is not None:
            clauses.append("topic_id = ?")
            params.append(topic_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = await self._fetchall(
            f"SELECT * FROM findings{where} ORDER BY created_at LIMIT ?", tuple(params)
        )
        return [m.Finding.from_row(r) for r in rows]

    async def set_finding_status(self, finding_id: str, status: m.FindingStatus) -> None:
        await self._execute(
            "UPDATE findings SET status = ? WHERE id = ?", (status.value, finding_id)
        )

    async def add_poc_run(
        self,
        finding_id: str,
        agent_id: str | None,
        attempt: int,
        exit_code: int,
        success: bool,
        script: str,
        stdout: str,
        stderr: str,
        container_id: str | None,
    ) -> None:
        await self._execute(
            """INSERT INTO poc_runs (finding_id, agent_id, attempt, exit_code, success,
                                     script, stdout, stderr, container_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                finding_id,
                agent_id,
                attempt,
                exit_code,
                1 if success else 0,
                script,
                stdout,
                stderr,
                container_id,
                _now(),
            ),
        )

    async def list_poc_runs(self, finding_id: str) -> list[dict[str, Any]]:
        rows = await self._fetchall(
            "SELECT * FROM poc_runs WHERE finding_id = ? ORDER BY attempt", (finding_id,)
        )
        return [dict(r) for r in rows]
