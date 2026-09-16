"""Append-only message board built on top of :mod:`harness.db`.

``MessageBoard.post`` is the only write path for posts. Nothing here updates or
deletes a post. All state lives in the database.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import models as m

if TYPE_CHECKING:
    from .config import Settings
    from .db import Database
    from .logging_ import EventLogger
    from .mailbox import Mailbox

MAX_POST_CHARS = 8000
RATE_WINDOW_SECONDS = 3600


class RateLimited(Exception):
    """Raised when an agent has hit its post rate for the current window."""


class NotTopicOwner(Exception):
    """Raised when a handoff is attempted by a non-owner."""


class InvalidPost(Exception):
    """Raised when a post or topic argument fails validation."""


@dataclass
class PostResult:
    post: m.Post
    woken: list[str]


class MessageBoard:
    def __init__(
        self,
        db: Database,
        mailbox: Mailbox,
        settings: Settings,
        logger: EventLogger | None = None,
    ) -> None:
        self.db = db
        self.mailbox = mailbox
        self.settings = settings
        self.logger = logger

    def _log(self, kind: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.event(kind, **fields)

    async def ensure_topic(self, title: str, owner_id: str | None = None) -> m.Topic:
        if not isinstance(title, str) or not title.strip():
            raise InvalidPost("topic title must be a non-empty string")
        existing = await self.db.topic_by_title(title)
        if existing is not None:
            return existing
        return await self.db.create_topic(title, owner_id=owner_id)

    async def subscribe(self, agent_id: str, topic_id: str) -> None:
        if await self.db.get_agent(agent_id) is None:
            raise InvalidPost(f"unknown agent: {agent_id}")
        await self.db.subscribe(agent_id, topic_id)

    async def unsubscribe(self, agent_id: str, topic_id: str) -> None:
        await self.db.unsubscribe(agent_id, topic_id)

    async def subscribers(self, topic_id: str) -> list[str]:
        return await self.db.list_subscribers(topic_id)

    async def owner_id(self, topic_id: str) -> str | None:
        topic = await self.db.get_topic(topic_id)
        return topic.owner_id if topic is not None else None

    async def post(
        self,
        agent_id: str,
        topic_id: str,
        body: str,
        parent_id: int | None = None,
        refs: list[dict] | None = None,
    ) -> PostResult:
        if await self.db.get_agent(agent_id) is None:
            raise InvalidPost(f"unknown agent: {agent_id}")
        if await self.db.get_topic(topic_id) is None:
            raise InvalidPost(f"unknown topic: {topic_id}")
        self._validate_body(body)
        await self._validate_parent(parent_id, topic_id)
        refs_list = self._validate_refs(refs)

        window_start = time.time() - RATE_WINDOW_SECONDS
        post_id = await self.db.insert_post_limited(
            agent_id,
            topic_id,
            body,
            parent_id,
            refs_list,
            window_start,
            self.settings.posts_per_agent_per_hour,
        )
        if post_id is None:
            recent = await self.db.count_posts_since(agent_id, window_start)
            self._log(
                "message_board_rate_limited",
                agent_id=agent_id,
                topic_id=topic_id,
                recent=recent,
                limit=self.settings.posts_per_agent_per_hour,
            )
            raise RateLimited(
                f"agent {agent_id} posted {recent} times in the last hour"
            )

        await self.db.bump_topic(topic_id)
        post = await self.db.get_post(post_id)
        assert post is not None

        subscribers = await self.db.list_subscribers(topic_id)
        recipients = [agent for agent in subscribers if agent != agent_id]
        woken = await self.mailbox.notify_subscribers(post, recipients)

        self._log(
            "message_board_post",
            agent_id=agent_id,
            topic_id=topic_id,
            post_id=post.id,
            parent_id=parent_id,
            woken=woken,
        )
        return PostResult(post=post, woken=woken)

    async def read(
        self, topic_id: str, since_id: int = 0, limit: int = 100
    ) -> list[m.Post]:
        return await self.db.list_posts(topic_id, since_id=since_id, limit=limit)

    async def handoff(
        self,
        topic_id: str,
        from_agent: str,
        to_agent: str,
        note: str = "",
    ) -> PostResult:
        topic = await self.db.get_topic(topic_id)
        if topic is None:
            raise InvalidPost(f"unknown topic: {topic_id}")
        if await self.db.get_agent(from_agent) is None:
            raise InvalidPost(f"unknown agent: {from_agent}")
        if topic.owner_id != from_agent:
            raise NotTopicOwner(
                f"agent {from_agent} does not own topic {topic_id}"
            )
        if await self.db.get_agent(to_agent) is None:
            raise InvalidPost(f"unknown agent: {to_agent}")

        reserved = await self.db.set_topic_owner_cas(topic_id, from_agent, to_agent)
        if not reserved:
            raise NotTopicOwner(
                f"agent {from_agent} does not own topic {topic_id}"
            )

        try:
            result = await self.post(
                from_agent,
                topic_id,
                self._handoff_body(topic_id, from_agent, to_agent, note),
            )
        except Exception:
            await self.db.set_topic_owner(topic_id, from_agent)
            raise
        self._log(
            "message_board_handoff",
            topic_id=topic_id,
            from_agent=from_agent,
            to_agent=to_agent,
            post_id=result.post.id,
        )
        return result

    def _validate_body(self, body: str) -> None:
        if not isinstance(body, str):
            raise InvalidPost("body must be a string")
        if not body.strip():
            raise InvalidPost("body must not be blank")
        if len(body) > MAX_POST_CHARS:
            raise InvalidPost(
                f"body must be 1..{MAX_POST_CHARS} characters, got {len(body)}"
            )

    async def _validate_parent(self, parent_id: int | None, topic_id: str) -> None:
        if parent_id is None:
            return
        if not isinstance(parent_id, int) or isinstance(parent_id, bool):
            raise InvalidPost("parent_id must be an integer")
        parent = await self.db.get_post(parent_id)
        if parent is None:
            raise InvalidPost(f"unknown parent post: {parent_id}")
        if parent.topic_id != topic_id:
            raise InvalidPost(
                f"parent post {parent_id} is not in topic {topic_id}"
            )

    def _validate_refs(self, refs: list[dict] | None) -> list[dict]:
        if refs is None:
            return []
        if not isinstance(refs, list):
            raise InvalidPost("refs must be a list of dicts")
        for item in refs:
            if not isinstance(item, dict):
                raise InvalidPost("refs must be a list of dicts")
            for key in item:
                if not isinstance(key, str):
                    raise InvalidPost("refs dict keys must be strings")
        try:
            json.dumps(refs)
        except (TypeError, ValueError) as exc:
            raise InvalidPost(f"refs must be JSON-serializable: {exc}") from exc
        return refs

    @staticmethod
    def _handoff_body(
        topic_id: str, from_agent: str, to_agent: str, note: str
    ) -> str:
        body = (
            f"Handoff: ownership of topic {topic_id} moves from "
            f"{from_agent} to {to_agent}."
        )
        note_text = note.strip() if isinstance(note, str) else str(note)
        if note_text:
            body = f"{body} {note_text}"
        return body[:MAX_POST_CHARS]
