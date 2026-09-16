from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from . import models as m

if TYPE_CHECKING:
    from .db import Database
    from .logging_ import EventLogger


class InvalidTransition(Exception):
    pass


class StateMachine:
    ALLOWED: ClassVar[dict[m.AgentState, set[m.AgentState]]] = {
        m.AgentState.SPAWNED: {
            m.AgentState.RUNNING,
            m.AgentState.WAITING,
            m.AgentState.DEAD,
        },
        m.AgentState.RUNNING: {
            m.AgentState.PAUSED,
            m.AgentState.WAITING,
            m.AgentState.DEAD,
        },
        m.AgentState.PAUSED: {
            m.AgentState.WAITING,
            m.AgentState.RUNNING,
            m.AgentState.DEAD,
        },
        m.AgentState.WAITING: {
            m.AgentState.RUNNING,
            m.AgentState.PAUSED,
            m.AgentState.DEAD,
        },
        m.AgentState.DEAD: set(),
    }

    def __init__(self, db: Database, logger: EventLogger | None = None) -> None:
        self.db = db
        self.logger = logger

    def _log(self, kind: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.event(kind, **fields)

    @staticmethod
    def can_transition(src: m.AgentState, dst: m.AgentState) -> bool:
        return dst in StateMachine.ALLOWED.get(src, set())

    @staticmethod
    def initial() -> m.AgentState:
        return m.AgentState.SPAWNED

    @staticmethod
    def allowed_targets(state: m.AgentState) -> set[m.AgentState]:
        return set(StateMachine.ALLOWED.get(state, set()))

    async def transition(
        self, agent_id: str, dst: m.AgentState, reason: str | None = None
    ) -> m.Agent:
        for _ in range(3):
            agent = await self.db.get_agent(agent_id)
            if agent is None:
                raise InvalidTransition(f"unknown agent: {agent_id}")
            if agent.state == dst:
                return agent
            if not self.can_transition(agent.state, dst):
                raise InvalidTransition(
                    f"invalid transition for {agent_id}: {agent.state.value} -> {dst.value}"
                )
            if await self.db.transition_state(agent_id, agent.state, dst):
                self._log(
                    "state",
                    agent_id=agent_id,
                    src=agent.state.value,
                    dst=dst.value,
                    reason=reason,
                )
                refreshed = await self.db.get_agent(agent_id)
                if refreshed is None:
                    raise InvalidTransition(f"agent disappeared during transition: {agent_id}")
                return refreshed
        raise InvalidTransition(
            f"transition race unresolved for {agent_id} -> {dst.value}"
        )
