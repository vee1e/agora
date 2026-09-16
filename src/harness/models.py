"""Shared data types for the harness.

Everything here is plain dataclasses and enums. No I/O, no framework.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentState(str, Enum):
    SPAWNED = "spawned"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING = "waiting"
    DEAD = "dead"


class AgentRole(str, Enum):
    FINDER = "finder"
    KILLER = "killer"
    JUDGE = "judge"
    EXPLOIT = "exploit"
    SUMMARIZER = "summarizer"
    GENERIC = "generic"


class TurnState(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"


class ContainerStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    REMOVED = "removed"


class FindingStatus(str, Enum):
    CANDIDATE = "candidate"
    SURVIVED = "survived"
    KILLED = "killed"
    EXPLOITED = "exploited"


@dataclass
class Agent:
    id: str
    role: AgentRole
    state: AgentState
    container_id: str | None = None
    topic_id: str | None = None
    context_path: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    last_wake_at: float = 0.0

    @classmethod
    def from_row(cls, row: Any) -> "Agent":
        return cls(
            id=row["id"],
            role=AgentRole(row["role"]),
            state=AgentState(row["state"]),
            container_id=row["container_id"],
            topic_id=row["topic_id"],
            context_path=row["context_path"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_wake_at=row["last_wake_at"],
        )


@dataclass
class Turn:
    id: int | None
    agent_id: str
    started_at: float
    ended_at: float | None = None
    state: TurnState = TurnState.RUNNING
    tokens_in: int = 0
    tokens_out: int = 0
    reason: str | None = None

    @classmethod
    def from_row(cls, row: Any) -> "Turn":
        return cls(
            id=row["id"],
            agent_id=row["agent_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            state=TurnState(row["state"]),
            tokens_in=row["tokens_in"],
            tokens_out=row["tokens_out"],
            reason=row["reason"],
        )


@dataclass
class Container:
    id: str
    agent_id: str | None
    image: str
    workspace_path: str
    status: ContainerStatus
    created_at: float = 0.0
    updated_at: float = 0.0

    @classmethod
    def from_row(cls, row: Any) -> "Container":
        return cls(
            id=row["id"],
            agent_id=row["agent_id"],
            image=row["image"],
            workspace_path=row["workspace_path"],
            status=ContainerStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass
class MailMessage:
    id: int | None
    agent_id: str
    sender_id: str | None
    topic_id: str | None
    post_id: int | None
    body: str
    created_at: float
    read_at: float | None = None

    @classmethod
    def from_row(cls, row: Any) -> "MailMessage":
        return cls(
            id=row["id"],
            agent_id=row["agent_id"],
            sender_id=row["sender_id"],
            topic_id=row["topic_id"],
            post_id=row["post_id"],
            body=row["body"],
            created_at=row["created_at"],
            read_at=row["read_at"],
        )


@dataclass
class Topic:
    id: str
    title: str
    owner_id: str | None
    created_at: float
    updated_at: float
    post_count: int = 0
    last_digest_post_id: int = 0

    @classmethod
    def from_row(cls, row: Any) -> "Topic":
        return cls(
            id=row["id"],
            title=row["title"],
            owner_id=row["owner_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            post_count=row["post_count"],
            last_digest_post_id=row["last_digest_post_id"],
        )


@dataclass
class Post:
    id: int | None
    agent_id: str
    time: float
    topic_id: str
    parent_id: int | None
    body: str
    refs: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: Any) -> "Post":
        raw = row["refs"] or "[]"
        try:
            refs = json.loads(raw)
        except (TypeError, ValueError):
            refs = []
        return cls(
            id=row["id"],
            agent_id=row["agent_id"],
            time=row["time"],
            topic_id=row["topic_id"],
            parent_id=row["parent_id"],
            body=row["body"],
            refs=refs,
        )


@dataclass
class BudgetRow:
    id: int | None
    scope: str
    key: str
    day: str
    tokens_used: int
    cap: int
    updated_at: float

    @classmethod
    def from_row(cls, row: Any) -> "BudgetRow":
        return cls(
            id=row["id"],
            scope=row["scope"],
            key=row["key"],
            day=row["day"],
            tokens_used=row["tokens_used"],
            cap=row["cap"],
            updated_at=row["updated_at"],
        )


@dataclass
class Finding:
    id: str
    topic_id: str
    agent_id: str
    file: str
    line: int
    sink: str
    call_path: list[str]
    status: FindingStatus
    created_at: float
    detail: str | None = None

    @classmethod
    def from_row(cls, row: Any) -> "Finding":
        raw = row["call_path"] or "[]"
        try:
            call_path = json.loads(raw)
        except (TypeError, ValueError):
            call_path = []
        return cls(
            id=row["id"],
            topic_id=row["topic_id"],
            agent_id=row["agent_id"],
            file=row["file"],
            line=row["line"],
            sink=row["sink"],
            call_path=call_path,
            status=FindingStatus(row["status"]),
            created_at=row["created_at"],
            detail=row["detail"],
        )


@dataclass
class Usage:
    """Token usage returned by one LLM call."""

    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def total(self) -> int:
        return self.tokens_in + self.tokens_out
