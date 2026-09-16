import asyncio
import pathlib

import pytest
import pytest_asyncio

from harness.config import Settings
from harness.db import Database
from harness.logging_ import EventLogger


@pytest.fixture
def settings(tmp_path: pathlib.Path) -> Settings:
    s = Settings(
        db_path=tmp_path / "test.db",
        data_dir=tmp_path,
        llm_base_url="http://127.0.0.1:18000/v1",
        llm_api_key="test-key",
        llm_model="deepseek-v4.1-flash",
        docker_enabled=False,
        floor=2,
        hard_cap=4,
        worker_pool_size=2,
        turn_token_budget=1000,
        daily_agent_token_cap=5000,
        global_token_cap=10000,
    )
    s.ensure_dirs()
    return s


@pytest_asyncio.fixture
async def db(settings: Settings) -> Database:
    database = Database(settings.db_path)
    await database.connect()
    await database.init_schema()
    yield database
    await database.close()


@pytest.fixture
def logger(settings: Settings) -> EventLogger:
    return EventLogger(settings.log_dir)
