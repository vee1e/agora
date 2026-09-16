from __future__ import annotations

from typing import Any

import pytest

from harness import models as m
from harness.db import Database
from harness.state_machine import InvalidTransition, StateMachine


class FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def event(self, kind: str, **fields: Any) -> None:
        self.events.append((kind, fields))


def test_initial_is_spawned() -> None:
    assert StateMachine.initial() is m.AgentState.SPAWNED


def test_allowed_targets() -> None:
    assert StateMachine.allowed_targets(m.AgentState.SPAWNED) == {
        m.AgentState.RUNNING,
        m.AgentState.WAITING,
        m.AgentState.DEAD,
    }
    assert StateMachine.allowed_targets(m.AgentState.RUNNING) == {
        m.AgentState.PAUSED,
        m.AgentState.WAITING,
        m.AgentState.DEAD,
    }
    assert StateMachine.allowed_targets(m.AgentState.PAUSED) == {
        m.AgentState.WAITING,
        m.AgentState.RUNNING,
        m.AgentState.DEAD,
    }
    assert StateMachine.allowed_targets(m.AgentState.WAITING) == {
        m.AgentState.RUNNING,
        m.AgentState.PAUSED,
        m.AgentState.DEAD,
    }
    assert StateMachine.allowed_targets(m.AgentState.DEAD) == set()


def test_allowed_targets_returns_copy() -> None:
    targets = StateMachine.allowed_targets(m.AgentState.RUNNING)
    targets.add(m.AgentState.SPAWNED)
    assert m.AgentState.SPAWNED not in StateMachine.allowed_targets(
        m.AgentState.RUNNING
    )


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (m.AgentState.SPAWNED, m.AgentState.RUNNING),
        (m.AgentState.SPAWNED, m.AgentState.WAITING),
        (m.AgentState.SPAWNED, m.AgentState.DEAD),
        (m.AgentState.RUNNING, m.AgentState.PAUSED),
        (m.AgentState.RUNNING, m.AgentState.WAITING),
        (m.AgentState.RUNNING, m.AgentState.DEAD),
        (m.AgentState.PAUSED, m.AgentState.WAITING),
        (m.AgentState.PAUSED, m.AgentState.RUNNING),
        (m.AgentState.PAUSED, m.AgentState.DEAD),
        (m.AgentState.WAITING, m.AgentState.RUNNING),
        (m.AgentState.WAITING, m.AgentState.PAUSED),
        (m.AgentState.WAITING, m.AgentState.DEAD),
    ],
)
def test_can_transition_allowed(
    src: m.AgentState, dst: m.AgentState
) -> None:
    assert StateMachine.can_transition(src, dst) is True


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (m.AgentState.SPAWNED, m.AgentState.PAUSED),
        (m.AgentState.SPAWNED, m.AgentState.SPAWNED),
        (m.AgentState.RUNNING, m.AgentState.RUNNING),
        (m.AgentState.RUNNING, m.AgentState.SPAWNED),
        (m.AgentState.PAUSED, m.AgentState.PAUSED),
        (m.AgentState.PAUSED, m.AgentState.SPAWNED),
        (m.AgentState.WAITING, m.AgentState.WAITING),
        (m.AgentState.WAITING, m.AgentState.SPAWNED),
        (m.AgentState.DEAD, m.AgentState.RUNNING),
        (m.AgentState.DEAD, m.AgentState.WAITING),
        (m.AgentState.DEAD, m.AgentState.DEAD),
    ],
)
def test_can_transition_denied(src: m.AgentState, dst: m.AgentState) -> None:
    assert StateMachine.can_transition(src, dst) is False


async def test_transition_updates_agent_and_db(db: Database) -> None:
    machine = StateMachine(db)
    agent = await db.create_agent(m.AgentRole.GENERIC)
    assert agent.state is m.AgentState.SPAWNED

    updated = await machine.transition(agent.id, m.AgentState.RUNNING)
    assert updated.id == agent.id
    assert updated.state is m.AgentState.RUNNING

    row = await db.get_agent(agent.id)
    assert row is not None
    assert row.state is m.AgentState.RUNNING


async def test_transition_invalid_keeps_state(db: Database) -> None:
    machine = StateMachine(db)
    agent = await db.create_agent(m.AgentRole.GENERIC)
    with pytest.raises(InvalidTransition):
        await machine.transition(agent.id, m.AgentState.PAUSED)
    row = await db.get_agent(agent.id)
    assert row is not None
    assert row.state is m.AgentState.SPAWNED


async def test_transition_unknown_agent_raises(db: Database) -> None:
    machine = StateMachine(db)
    with pytest.raises(InvalidTransition):
        await machine.transition("ag_missing", m.AgentState.RUNNING)


async def test_dead_is_terminal(db: Database) -> None:
    machine = StateMachine(db)
    agent = await db.create_agent(m.AgentRole.GENERIC)
    await machine.transition(agent.id, m.AgentState.DEAD)
    with pytest.raises(InvalidTransition):
        await machine.transition(agent.id, m.AgentState.RUNNING)
    row = await db.get_agent(agent.id)
    assert row is not None
    assert row.state is m.AgentState.DEAD


async def test_transition_logs_event(db: Database) -> None:
    logger = FakeLogger()
    machine = StateMachine(db, logger=logger)
    agent = await db.create_agent(m.AgentRole.GENERIC)
    await machine.transition(agent.id, m.AgentState.RUNNING, reason="go")
    assert logger.events
    kind, fields = logger.events[0]
    assert kind == "state"
    assert fields["agent_id"] == agent.id
    assert fields["src"] == "spawned"
    assert fields["dst"] == "running"
    assert fields["reason"] == "go"
