from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from . import models as m

if TYPE_CHECKING:
    from .db import Database
    from .logging_ import EventLogger


class Mailbox:
    def __init__(self, db: Database, logger: EventLogger | None = None) -> None:
        self.db = db
        self.logger = logger
        self._drain_lock = asyncio.Lock()

    def _log(self, kind: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.event(kind, **fields)

    async def send(
        self,
        agent_id: str,
        body: str,
        sender_id: str | None = None,
        topic_id: str | None = None,
        post_id: int | None = None,
    ) -> m.MailMessage:
        message = await self.db.send_mail(
            agent_id,
            body,
            sender_id=sender_id,
            topic_id=topic_id,
            post_id=post_id,
        )
        self._log(
            "mail",
            action="send",
            agent_id=agent_id,
            sender_id=sender_id,
            topic_id=topic_id,
            post_id=post_id,
            mail_id=message.id,
        )
        return message

    async def drain(self, agent_id: str, limit: int = 50) -> list[m.MailMessage]:
        async with self._drain_lock:
            messages = await self.db.drain_mail(agent_id, limit)
        self._log(
            "mail",
            action="drain",
            agent_id=agent_id,
            count=len(messages),
        )
        return messages

    async def has_unread(self, agent_id: str) -> bool:
        return await self.db.count_unread(agent_id) > 0

    async def notify_subscribers(
        self, post: m.Post, subscribers: list[str]
    ) -> list[str]:
        candidates = list(
            dict.fromkeys(agent for agent in subscribers if agent != post.agent_id)
        )
        recipients: list[str] = []
        for agent_id in candidates:
            agent = await self.db.get_agent(agent_id)
            if agent is None:
                self._log(
                    "mail_unknown_recipient",
                    agent_id=agent_id,
                    topic_id=post.topic_id,
                    post_id=post.id,
                )
                continue
            if agent.state is m.AgentState.DEAD:
                self._log(
                    "mail_dead_recipient",
                    agent_id=agent_id,
                    topic_id=post.topic_id,
                    post_id=post.id,
                )
                continue
            recipients.append(agent_id)
        body = f"New post {post.id} in topic {post.topic_id}"
        for agent_id in recipients:
            await self.send(
                agent_id,
                body,
                sender_id=post.agent_id,
                topic_id=post.topic_id,
                post_id=post.id,
            )
        self._log(
            "mail",
            action="notify",
            topic_id=post.topic_id,
            post_id=post.id,
            recipients=recipients,
        )
        return recipients
