import re

import pytest

from harness.budget import BudgetExceeded, BudgetGuard
from harness.config import Settings
from harness.db import Database


def test_today_is_utc_iso_date() -> None:
    today = BudgetGuard.today()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", today)


async def test_record_writes_agent_and_global_rows(
    db: Database, settings: Settings
) -> None:
    guard = BudgetGuard(db, settings)
    await guard.record("ag_1", 100)

    day = guard.today()
    agent_row = await db.get_budget("agent", "ag_1", day)
    global_row = await db.get_budget("global", "global", day)
    assert agent_row is not None and agent_row.tokens_used == 100
    assert agent_row.cap == settings.daily_agent_token_cap
    assert global_row is not None and global_row.tokens_used == 100
    assert global_row.cap == settings.global_token_cap


async def test_record_accumulates(db: Database, settings: Settings) -> None:
    guard = BudgetGuard(db, settings)
    await guard.record("ag_1", 100)
    await guard.record("ag_1", 50)
    assert await guard.agent_used("ag_1") == 150
    assert await guard.global_used() == 150


async def test_agent_used_defaults_to_zero(db: Database, settings: Settings) -> None:
    guard = BudgetGuard(db, settings)
    assert await guard.agent_used("missing") == 0
    assert await guard.global_used() == 0


async def test_check_agent_raises_at_cap(db: Database, settings: Settings) -> None:
    guard = BudgetGuard(db, settings)
    await guard.record("ag_1", settings.daily_agent_token_cap)
    with pytest.raises(BudgetExceeded):
        await guard.check_agent("ag_1")


async def test_check_agent_allows_below_cap(db: Database, settings: Settings) -> None:
    guard = BudgetGuard(db, settings)
    await guard.record("ag_1", settings.daily_agent_token_cap - 1)
    await guard.check_agent("ag_1")


async def test_agent_cap_is_per_agent(db: Database, settings: Settings) -> None:
    guard = BudgetGuard(db, settings)
    await guard.record("ag_1", settings.daily_agent_token_cap)
    await guard.check_agent("ag_2")


async def test_check_global_raises_at_cap(db: Database, settings: Settings) -> None:
    guard = BudgetGuard(db, settings)
    await guard.record("ag_1", settings.global_token_cap)
    with pytest.raises(BudgetExceeded):
        await guard.check_global()


async def test_can_spawn(db: Database, settings: Settings) -> None:
    guard = BudgetGuard(db, settings)
    assert await guard.can_spawn() is True
    await guard.record("ag_1", settings.global_token_cap)
    assert await guard.can_spawn() is False


async def test_remaining_global(db: Database, settings: Settings) -> None:
    guard = BudgetGuard(db, settings)
    assert await guard.remaining_global() == settings.global_token_cap
    await guard.record("ag_1", 400)
    assert await guard.remaining_global() == settings.global_token_cap - 400


async def test_remaining_global_never_negative(
    db: Database, settings: Settings
) -> None:
    guard = BudgetGuard(db, settings)
    await guard.record("ag_1", settings.global_token_cap + 100)
    assert await guard.remaining_global() == 0


async def test_logger_receives_record_event(
    db: Database, settings: Settings, logger
) -> None:
    guard = BudgetGuard(db, settings, logger=logger)
    await guard.record("ag_1", 7)
    records = (settings.log_dir / "events.jsonl").read_text().splitlines()
    assert any('"budget_record"' in line for line in records)
