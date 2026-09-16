"""Daily token budget accounting on top of the ``budget`` table.

Two rows per day: one per agent (``scope="agent"``) and one global
(``scope="global"``). Caps come from Settings.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .config import Settings
from .db import Database
from .logging_ import EventLogger

GLOBAL_KEY = "global"


class BudgetExceeded(Exception):
    pass


class BudgetGuard:
    def __init__(
        self,
        db: Database,
        settings: Settings,
        logger: EventLogger | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self.logger = logger

    @staticmethod
    def today() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    async def agent_used(self, agent_id: str) -> int:
        row = await self.db.get_budget("agent", agent_id, self.today())
        return row.tokens_used if row is not None else 0

    async def global_used(self) -> int:
        row = await self.db.get_budget("global", GLOBAL_KEY, self.today())
        return row.tokens_used if row is not None else 0

    async def check_agent(self, agent_id: str) -> None:
        used = await self.agent_used(agent_id)
        if used >= self.settings.daily_agent_token_cap:
            self._log_exceeded("agent", used, self.settings.daily_agent_token_cap)
            raise BudgetExceeded(
                f"agent {agent_id} used {used} tokens, cap "
                f"{self.settings.daily_agent_token_cap}"
            )

    async def check_global(self) -> None:
        used = await self.global_used()
        if used >= self.settings.global_token_cap:
            self._log_exceeded("global", used, self.settings.global_token_cap)
            raise BudgetExceeded(
                f"global used {used} tokens, cap {self.settings.global_token_cap}"
            )

    async def can_spawn(self) -> bool:
        return await self.global_used() < self.settings.global_token_cap

    async def record(self, agent_id: str, tokens: int) -> None:
        day = self.today()
        await self.db.add_tokens_many(
            [
                ("agent", agent_id, day, tokens, self.settings.daily_agent_token_cap),
                ("global", GLOBAL_KEY, day, tokens, self.settings.global_token_cap),
            ]
        )
        if self.logger is not None:
            self.logger.event(
                "budget_record", agent_id=agent_id, tokens=tokens, day=day
            )

    async def remaining_global(self) -> int:
        used = await self.global_used()
        return max(0, self.settings.global_token_cap - used)

    def _log_exceeded(self, scope: str, used: int, cap: int) -> None:
        if self.logger is not None:
            self.logger.event(
                "budget_exceeded", scope=scope, used=used, cap=cap
            )
