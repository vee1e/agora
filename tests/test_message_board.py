from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import replace

import pytest

from harness import models as m
from harness.config import Settings
from harness.db import Database
from harness.message_board import MessageBoard, InvalidPost, NotTopicOwner, PostResult, RateLimited
from harness.logging_ import EventLogger
from harness.mailbox import Mailbox


def _message_board(db: Database, settings: Settings, logger: EventLogger) -> MessageBoard:
    return MessageBoard(db, Mailbox(db, logger=logger), settings, logger=logger)


async def _add_agent(db: Database, agent_id: str) -> m.Agent:
    return await db.create_agent(m.AgentRole.GENERIC, agent_id=agent_id)


async def _setup(
    db: Database, settings: Settings, logger: EventLogger, owner: str = "ag_owner"
) -> tuple[MessageBoard, m.Agent, m.Topic]:
    message_board = _message_board(db, settings, logger)
    agent = await _add_agent(db, owner)
    topic = await message_board.ensure_topic("topic one", owner_id=owner)
    return message_board, agent, topic


async def _insert_old_post(
    db: Database, agent_id: str, topic_id: str, when: float
) -> None:
    await db.conn.execute(
        """INSERT INTO posts (agent_id, time, topic_id, parent_id, body, refs)
           VALUES (?, ?, ?, NULL, ?, '[]')""",
        (agent_id, when, topic_id, "old post"),
    )
    await db.conn.commit()


async def test_post_happy_path(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, agent, topic = await _setup(db, settings, logger)
    result = await message_board.post(agent.id, topic.id, "hello board")

    assert result.post.id is not None
    assert result.post.agent_id == agent.id
    assert result.post.topic_id == topic.id
    assert result.post.parent_id is None
    assert result.post.body == "hello board"
    assert result.woken == []

    refreshed = await db.get_topic(topic.id)
    assert refreshed is not None
    assert refreshed.post_count == 1

    posts = await message_board.read(topic.id)
    assert [p.id for p in posts] == [result.post.id]


async def test_subscriber_woken_author_not(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, author, topic = await _setup(db, settings, logger)
    reader = await _add_agent(db, "ag_reader")
    await message_board.subscribe(author.id, topic.id)
    await message_board.subscribe(reader.id, topic.id)

    result = await message_board.post(author.id, topic.id, "ping")

    assert result.woken == [reader.id]
    assert await db.count_unread(reader.id) == 1
    assert await db.count_unread(author.id) == 0


async def test_unknown_agent_rejected(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, _agent, topic = await _setup(db, settings, logger)

    with pytest.raises(InvalidPost):
        await message_board.post("ag_missing", topic.id, "body")


async def test_unknown_topic_rejected(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, agent, _topic = await _setup(db, settings, logger)

    with pytest.raises(InvalidPost):
        await message_board.post(agent.id, "tp_missing", "body")


async def test_empty_body_rejected(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, agent, topic = await _setup(db, settings, logger)

    with pytest.raises(InvalidPost):
        await message_board.post(agent.id, topic.id, "")


async def test_whitespace_only_body_rejected(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, agent, topic = await _setup(db, settings, logger)

    with pytest.raises(InvalidPost):
        await message_board.post(agent.id, topic.id, "   \n\t  ")

    assert await message_board.read(topic.id) == []


async def test_oversized_body_rejected(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, agent, topic = await _setup(db, settings, logger)

    with pytest.raises(InvalidPost):
        await message_board.post(agent.id, topic.id, "x" * 8001)


async def test_body_at_limit_accepted(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, agent, topic = await _setup(db, settings, logger)

    result = await message_board.post(agent.id, topic.id, "x" * 8000)

    assert len(result.post.body) == 8000


async def test_unknown_parent_rejected(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, agent, topic = await _setup(db, settings, logger)

    with pytest.raises(InvalidPost):
        await message_board.post(agent.id, topic.id, "reply", parent_id=9999)


async def test_parent_from_other_topic_rejected(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, agent, topic = await _setup(db, settings, logger)
    other = await message_board.ensure_topic("topic two")
    first = await message_board.post(agent.id, topic.id, "root")

    with pytest.raises(InvalidPost):
        await message_board.post(agent.id, other.id, "cross reply", parent_id=first.post.id)


async def test_valid_parent_in_same_topic_accepted(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, agent, topic = await _setup(db, settings, logger)
    first = await message_board.post(agent.id, topic.id, "root")

    reply = await message_board.post(agent.id, topic.id, "reply", parent_id=first.post.id)

    assert reply.post.parent_id == first.post.id


async def test_bad_refs_rejected(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, agent, topic = await _setup(db, settings, logger)

    with pytest.raises(InvalidPost):
        await message_board.post(agent.id, topic.id, "body", refs="not a list")
    with pytest.raises(InvalidPost):
        await message_board.post(agent.id, topic.id, "body", refs=[{"file": "a.c"}, "x"])
    with pytest.raises(InvalidPost):
        await message_board.post(agent.id, topic.id, "body", refs=[{1: "a"}])
    with pytest.raises(InvalidPost):
        await message_board.post(agent.id, topic.id, "body", refs={"file": "a.c"})


async def test_non_serializable_refs_rejected(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, agent, topic = await _setup(db, settings, logger)

    with pytest.raises(InvalidPost):
        await message_board.post(agent.id, topic.id, "body", refs=[{"tags": {"a", "b"}}])
    with pytest.raises(InvalidPost):
        await message_board.post(agent.id, topic.id, "body", refs=[{"obj": object()}])

    assert await message_board.read(topic.id) == []


async def test_valid_refs_accepted(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, agent, topic = await _setup(db, settings, logger)
    refs = [{"file": "a.c", "line": 12}, {"commit": "deadbeef"}]

    result = await message_board.post(agent.id, topic.id, "finding", refs=refs)

    assert result.post.refs == refs


async def test_rate_limit_triggers_on_next_post(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    limited = replace(settings, posts_per_agent_per_hour=3)
    message_board, agent, topic = await _setup(db, limited, logger)

    for index in range(3):
        await message_board.post(agent.id, topic.id, f"post {index}")

    with pytest.raises(RateLimited):
        await message_board.post(agent.id, topic.id, "one too many")

    assert len(await message_board.read(topic.id)) == 3


async def test_rate_limit_ignores_posts_older_than_an_hour(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    limited = replace(settings, posts_per_agent_per_hour=3)
    message_board, agent, topic = await _setup(db, limited, logger)
    old = time.time() - 7200
    for index in range(3):
        await _insert_old_post(db, agent.id, topic.id, old - index)

    result = await message_board.post(agent.id, topic.id, "fresh post")

    assert result.post.body == "fresh post"
    assert len(await message_board.read(topic.id)) == 4


async def test_concurrent_posts_cannot_exceed_rate_limit(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    limited = replace(settings, posts_per_agent_per_hour=3)
    message_board, agent, topic = await _setup(db, limited, logger)

    results = await asyncio.gather(
        *(message_board.post(agent.id, topic.id, f"post {index}") for index in range(10)),
        return_exceptions=True,
    )

    accepted = [r for r in results if isinstance(r, PostResult)]
    rejected = [r for r in results if isinstance(r, RateLimited)]
    assert len(accepted) == 3
    assert len(rejected) == 7
    assert len(await message_board.read(topic.id)) == 3


async def test_handoff_moves_owner_and_appends_post(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, owner, topic = await _setup(db, settings, logger)
    successor = await _add_agent(db, "ag_next")

    result = await message_board.handoff(topic.id, owner.id, successor.id, note="take over")

    assert result.post.agent_id == owner.id
    assert result.post.topic_id == topic.id
    assert await message_board.owner_id(topic.id) == successor.id

    posts = await message_board.read(topic.id)
    assert [p.id for p in posts] == [result.post.id]
    assert "take over" in result.post.body


async def test_handoff_by_non_owner_rejected(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, _owner, topic = await _setup(db, settings, logger)
    await _add_agent(db, "ag_other")

    with pytest.raises(NotTopicOwner):
        await message_board.handoff(topic.id, "ag_other", "ag_other")

    assert await message_board.owner_id(topic.id) == "ag_owner"
    assert await message_board.read(topic.id) == []


async def test_handoff_unknown_agent_rejected(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, owner, topic = await _setup(db, settings, logger)

    with pytest.raises(InvalidPost):
        await message_board.handoff(topic.id, owner.id, "ag_missing")

    assert await message_board.owner_id(topic.id) == owner.id


async def test_rate_limited_handoff_leaves_ownership_unchanged(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    limited = replace(settings, posts_per_agent_per_hour=1)
    message_board, owner, topic = await _setup(db, limited, logger)
    successor = await _add_agent(db, "ag_next")
    await message_board.post(owner.id, topic.id, "first")

    with pytest.raises(RateLimited):
        await message_board.handoff(topic.id, owner.id, successor.id)

    assert await message_board.owner_id(topic.id) == owner.id
    assert len(await message_board.read(topic.id)) == 1


async def test_handoff_by_ghost_agent_leaves_ownership_unchanged(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, owner, topic = await _setup(db, settings, logger)
    successor = await _add_agent(db, "ag_next")

    with pytest.raises(InvalidPost):
        await message_board.handoff(topic.id, "ag_ghost", successor.id)

    assert await message_board.owner_id(topic.id) == owner.id
    assert await message_board.read(topic.id) == []


async def test_subscribe_unknown_agent_rejected(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, _agent, topic = await _setup(db, settings, logger)

    with pytest.raises(InvalidPost):
        await message_board.subscribe("ag_missing", topic.id)

    assert await message_board.subscribers(topic.id) == []


async def test_ensure_topic_is_idempotent(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, _agent, _topic = await _setup(db, settings, logger)

    first = await message_board.ensure_topic("shared topic")
    second = await message_board.ensure_topic("shared topic")

    assert first.id == second.id
    assert len(await db.list_topics()) == 2


async def test_subscribe_unsubscribe_round_trip(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, agent, topic = await _setup(db, settings, logger)

    await message_board.subscribe(agent.id, topic.id)
    assert await message_board.subscribers(topic.id) == [agent.id]

    await message_board.unsubscribe(agent.id, topic.id)
    assert await message_board.subscribers(topic.id) == []


async def test_read_honours_since_id_and_limit(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, agent, topic = await _setup(db, settings, logger)
    first = await message_board.post(agent.id, topic.id, "one")
    await message_board.post(agent.id, topic.id, "two")
    await message_board.post(agent.id, topic.id, "three")

    after = await message_board.read(topic.id, since_id=first.post.id)
    assert [p.body for p in after] == ["two", "three"]

    limited = await message_board.read(topic.id, limit=2)
    assert [p.body for p in limited] == ["one", "two"]


async def test_update_on_posts_is_rejected(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, agent, topic = await _setup(db, settings, logger)
    result = await message_board.post(agent.id, topic.id, "immutable")

    with pytest.raises(sqlite3.Error) as excinfo:
        await db.conn.execute(
            "UPDATE posts SET body = ? WHERE id = ?", ("changed", result.post.id)
        )
    assert "append-only" in str(excinfo.value)

    still = await db.get_post(result.post.id)
    assert still is not None
    assert still.body == "immutable"


async def test_delete_on_posts_is_rejected(
    db: Database, settings: Settings, logger: EventLogger
) -> None:
    message_board, agent, topic = await _setup(db, settings, logger)
    result = await message_board.post(agent.id, topic.id, "immutable")

    with pytest.raises(sqlite3.Error) as excinfo:
        await db.conn.execute("DELETE FROM posts WHERE id = ?", (result.post.id,))
    assert "append-only" in str(excinfo.value)

    still = await db.get_post(result.post.id)
    assert still is not None
