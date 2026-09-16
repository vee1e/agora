from __future__ import annotations

import asyncio
from typing import Any

from harness import models as m
from harness.db import Database
from harness.mailbox import Mailbox


class FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def event(self, kind: str, **fields: Any) -> None:
        self.events.append((kind, fields))


async def _add_agent(db: Database, agent_id: str) -> None:
    await db.create_agent(m.AgentRole.GENERIC, agent_id=agent_id)


async def test_send_returns_message(db: Database) -> None:
    box = Mailbox(db)
    message = await box.send(
        "ag_a", "hello", sender_id="ag_b", topic_id="tp_1", post_id=7
    )
    assert message.id is not None
    assert message.agent_id == "ag_a"
    assert message.sender_id == "ag_b"
    assert message.topic_id == "tp_1"
    assert message.post_id == 7
    assert message.body == "hello"
    assert message.read_at is None


async def test_drain_returns_unread_then_marks_read(db: Database) -> None:
    box = Mailbox(db)
    await box.send("ag_a", "one")
    await box.send("ag_a", "two")
    first = await box.drain("ag_a")
    assert [message.body for message in first] == ["one", "two"]
    assert all(message.id is not None for message in first)
    second = await box.drain("ag_a")
    assert second == []
    remaining = await db.unread_mail("ag_a")
    assert remaining == []


async def test_drain_only_touches_target_agent(db: Database) -> None:
    box = Mailbox(db)
    await box.send("ag_a", "for a")
    await box.send("ag_b", "for b")
    drained = await box.drain("ag_a")
    assert [message.body for message in drained] == ["for a"]
    assert await box.has_unread("ag_b") is True


async def test_concurrent_drain_returns_disjoint_messages(db: Database) -> None:
    box = Mailbox(db)
    for index in range(6):
        await box.send("ag_a", f"message {index}")

    first, second = await asyncio.gather(box.drain("ag_a"), box.drain("ag_a"))

    delivered = [message.id for message in first] + [
        message.id for message in second
    ]
    assert len(delivered) == 6
    assert len(set(delivered)) == 6
    assert await box.has_unread("ag_a") is False


async def test_has_unread_transitions(db: Database) -> None:
    box = Mailbox(db)
    assert await box.has_unread("ag_a") is False
    await box.send("ag_a", "hi")
    assert await box.has_unread("ag_a") is True
    await box.drain("ag_a")
    assert await box.has_unread("ag_a") is False


async def test_notify_subscribers_skips_author(db: Database) -> None:
    box = Mailbox(db)
    await _add_agent(db, "ag_b")
    await _add_agent(db, "ag_c")
    post = m.Post(
        id=42,
        agent_id="ag_author",
        time=0.0,
        topic_id="tp_news",
        parent_id=None,
        body="body",
    )
    recipients = await box.notify_subscribers(post, ["ag_author", "ag_b", "ag_c"])
    assert recipients == ["ag_b", "ag_c"]
    assert await box.has_unread("ag_author") is False


async def test_notify_subscribers_skips_unknown_recipients(db: Database) -> None:
    logger = FakeLogger()
    box = Mailbox(db, logger=logger)
    await _add_agent(db, "ag_b")
    post = m.Post(
        id=43,
        agent_id="ag_author",
        time=0.0,
        topic_id="tp_news",
        parent_id=None,
        body="body",
    )
    recipients = await box.notify_subscribers(post, ["ag_b", "ag_missing"])
    assert recipients == ["ag_b"]
    assert await box.has_unread("ag_b") is True
    assert any(kind == "mail_unknown_recipient" for kind, _ in logger.events)


async def test_notify_subscribers_skips_dead_recipients(db: Database) -> None:
    logger = FakeLogger()
    box = Mailbox(db, logger=logger)
    await _add_agent(db, "ag_live")
    await _add_agent(db, "ag_dead")
    await db.set_agent_state("ag_dead", m.AgentState.DEAD)
    post = m.Post(
        id=44,
        agent_id="ag_author",
        time=0.0,
        topic_id="tp_news",
        parent_id=None,
        body="body",
    )
    recipients = await box.notify_subscribers(post, ["ag_live", "ag_dead"])
    assert recipients == ["ag_live"]
    assert await box.has_unread("ag_live") is True
    assert await box.has_unread("ag_dead") is False
    assert any(kind == "mail_dead_recipient" for kind, _ in logger.events)


async def test_notify_subscribers_body_names_topic_and_post(db: Database) -> None:
    box = Mailbox(db)
    await _add_agent(db, "ag_b")
    post = m.Post(
        id=42,
        agent_id="ag_author",
        time=0.0,
        topic_id="tp_news",
        parent_id=None,
        body="body",
    )
    await box.notify_subscribers(post, ["ag_author", "ag_b"])
    messages = await db.unread_mail("ag_b")
    assert len(messages) == 1
    message = messages[0]
    assert str(post.id) in message.body
    assert post.topic_id in message.body
    assert message.post_id == post.id
    assert message.topic_id == post.topic_id
    assert message.sender_id == post.agent_id


async def test_notify_subscribers_deduplicates(db: Database) -> None:
    box = Mailbox(db)
    await _add_agent(db, "ag_b")
    await _add_agent(db, "ag_c")
    post = m.Post(
        id=1,
        agent_id="ag_author",
        time=0.0,
        topic_id="tp_news",
        parent_id=None,
        body="body",
    )
    recipients = await box.notify_subscribers(post, ["ag_b", "ag_b", "ag_c"])
    assert recipients == ["ag_b", "ag_c"]
    assert len(await db.unread_mail("ag_b")) == 1


async def test_notify_logs_mail_event(db: Database) -> None:
    logger = FakeLogger()
    box = Mailbox(db, logger=logger)
    await _add_agent(db, "ag_b")
    post = m.Post(
        id=5,
        agent_id="ag_author",
        time=0.0,
        topic_id="tp_news",
        parent_id=None,
        body="body",
    )
    await box.notify_subscribers(post, ["ag_author", "ag_b"])
    kinds = [kind for kind, _ in logger.events]
    assert "mail" in kinds
    assert any(fields.get("action") == "notify" for _, fields in logger.events)
