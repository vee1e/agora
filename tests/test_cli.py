from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from harness import models as m
from harness.cli import main
from harness.containers import ContainerManager
from harness.council.agents import CouncilResult
from harness.db import Database
from harness.llm import LLMError


class _FailingLLM:
    def __init__(self, *args: object, **kwargs: object) -> None:
        return None

    async def complete(self, *args: object, **kwargs: object) -> None:
        raise LLMError("transport down")

    async def aclose(self) -> None:
        return None


class _NoopLLM:
    def __init__(self, *args: object, **kwargs: object) -> None:
        return None

    async def complete(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected llm call")

    async def aclose(self) -> None:
        return None


def _set_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HARNESS_DB", str(tmp_path / "cli.db"))
    monkeypatch.setenv("HARNESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HARNESS_DOCKER", "0")
    monkeypatch.setenv("HARNESS_FLOOR", "1")
    monkeypatch.setenv("HARNESS_HARD_CAP", "2")


async def _counts(db_path: Path) -> dict[str, int]:
    db = Database(db_path)
    await db.connect()
    try:
        return {
            "agents": await db.count_agents(),
            "running": await db.count_agents(m.AgentState.RUNNING),
        }
    finally:
        await db.close()


async def _posts(db_path: Path) -> tuple[list[m.Topic], list[m.Post]]:
    db = Database(db_path)
    await db.connect()
    try:
        topics = await db.list_topics()
        general = [topic for topic in topics if topic.title == "general"]
        posts = await db.list_posts(general[0].id) if general else []
        return topics, posts
    finally:
        await db.close()


async def _seed_agent(
    db_path: Path, state: m.AgentState = m.AgentState.RUNNING
) -> str:
    db = Database(db_path)
    await db.connect()
    try:
        await db.init_schema()
        agent = await db.create_agent(m.AgentRole.GENERIC, state=state)
        return agent.id
    finally:
        await db.close()


async def _seed_topic(db_path: Path, title: str, owner_id: str | None = None) -> str:
    db = Database(db_path)
    await db.connect()
    try:
        await db.init_schema()
        topic = await db.create_topic(title, owner_id=owner_id)
        return topic.id
    finally:
        await db.close()


async def _seed_posts(
    db_path: Path, topic_id: str, agent_id: str, count: int
) -> None:
    db = Database(db_path)
    await db.connect()
    try:
        await db.init_schema()
        for index in range(count):
            await db.create_post(agent_id, topic_id, f"post {index}")
    finally:
        await db.close()


async def _subscribers(db_path: Path, topic_id: str) -> list[str]:
    db = Database(db_path)
    await db.connect()
    try:
        return await db.list_subscribers(topic_id)
    finally:
        await db.close()


def test_cli_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_env(monkeypatch, tmp_path)

    assert main(["init"]) == 0
    assert main(["status"]) == 0
    capsys.readouterr()

    assert main(["spawn", "--role", "finder"]) == 0
    spawn_out = capsys.readouterr().out
    agent_id = spawn_out.split()[1]
    assert agent_id.startswith("ag_")

    assert main(["post", "--topic", "general", "--agent", agent_id, "--body", "hi"]) == 0
    post_out = capsys.readouterr().out.strip()
    assert post_out.isdigit()

    db_path = tmp_path / "cli.db"
    counts = asyncio.run(_counts(db_path))
    assert counts == {"agents": 1, "running": 1}

    topics, posts = asyncio.run(_posts(db_path))
    assert len([topic for topic in topics if topic.title == "general"]) == 1
    assert len(posts) == 1
    assert posts[0].body == "hi"
    assert posts[0].agent_id == agent_id
    assert posts[0].id == int(post_out)


def test_cli_post_rate_limit_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HARNESS_POST_RATE", "1")

    assert main(["init"]) == 0
    capsys.readouterr()

    db_path = tmp_path / "cli.db"
    agent_id = asyncio.run(_seed_agent(db_path))

    assert main(["post", "--topic", "general", "--agent", agent_id, "--body", "one"]) == 0
    capsys.readouterr()

    assert main(["post", "--topic", "general", "--agent", agent_id, "--body", "two"]) == 1
    err = capsys.readouterr().err
    assert "rate" in err.lower()


def test_cli_post_unknown_agent_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_env(monkeypatch, tmp_path)

    assert main(["init"]) == 0
    capsys.readouterr()

    assert main(["post", "--topic", "general", "--agent", "ag_missing", "--body", "hi"]) == 1
    err = capsys.readouterr().err
    assert "invalid" in err.lower()


def test_cli_subscribe_and_topics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_env(monkeypatch, tmp_path)

    assert main(["init"]) == 0
    capsys.readouterr()

    db_path = tmp_path / "cli.db"
    agent_id = asyncio.run(_seed_agent(db_path))
    topic_id = asyncio.run(_seed_topic(db_path, "general", owner_id=agent_id))

    assert main(["subscribe", "--agent", agent_id, "--topic", "general"]) == 0
    assert "subscribed" in capsys.readouterr().out
    assert asyncio.run(_subscribers(db_path, topic_id)) == [agent_id]

    assert main(["topics"]) == 0
    out = capsys.readouterr().out
    assert "general" in out
    assert f"owner={agent_id}" in out
    assert "posts=0" in out


def test_cli_digest_prints_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_env(monkeypatch, tmp_path)

    assert main(["init"]) == 0
    capsys.readouterr()

    db_path = tmp_path / "cli.db"
    asyncio.run(_seed_topic(db_path, "general"))
    seen: dict[str, object] = {}

    async def fake_summarize(
        db: Database,
        llm: object,
        topic_id: str,
        every_n: int = 25,
        force: bool = False,
        logger: object = None,
        agent_id: str = "summarizer",
    ) -> dict[str, object]:
        seen["agent_id"] = agent_id
        return {
            "topic_id": topic_id,
            "from_post_id": 1,
            "to_post_id": 2,
            "body": "forced digest body",
        }

    monkeypatch.setattr("harness.cli.summarize_topic", fake_summarize)

    assert main(["digest", "--topic", "general"]) == 0
    assert "forced digest body" in capsys.readouterr().out
    assert seen["agent_id"] == "summarizer"


def test_cli_digest_enforces_global_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HARNESS_GLOBAL_CAP", "0")

    assert main(["init"]) == 0
    capsys.readouterr()

    db_path = tmp_path / "cli.db"
    agent_id = asyncio.run(_seed_agent(db_path))
    topic_id = asyncio.run(_seed_topic(db_path, "general", owner_id=agent_id))
    asyncio.run(_seed_posts(db_path, topic_id, agent_id, 3))

    assert main(["digest", "--topic", "general"]) == 1
    err = capsys.readouterr().err
    assert "budget" in err.lower()


def test_cli_digest_unknown_topic_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_env(monkeypatch, tmp_path)

    assert main(["init"]) == 0
    capsys.readouterr()

    assert main(["digest", "--topic", "nope"]) == 1
    err = capsys.readouterr().err
    assert "unknown topic" in err.lower()


def test_cli_status_reports_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_env(monkeypatch, tmp_path)

    assert main(["init"]) == 0
    capsys.readouterr()

    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "agents by state:" in out
    assert "running:" in out
    assert "tokens today (global):" in out


def test_cli_spawn_unknown_role_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_env(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        main(["spawn", "--role", "nope"])


def test_cli_spawn_blocked_by_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HARNESS_GLOBAL_CAP", "0")

    assert main(["init"]) == 0
    capsys.readouterr()

    assert main(["spawn", "--role", "finder"]) == 1
    err = capsys.readouterr().err
    assert "budget" in err.lower()


def test_cli_council_tools_prints_schemas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_env(monkeypatch, tmp_path)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "sample.c").write_text("int main(void) { return 0; }\n")

    assert main(["council", "tools", "--checkout", str(checkout)]) == 0

    out = capsys.readouterr().out
    for name in (
        "callers_of",
        "call_path",
        "alloc_of",
        "size_expr",
        "tainted",
        "sg_search",
    ):
        assert name in out


async def _state_counts(db_path: Path) -> dict[str, int]:
    db = Database(db_path)
    await db.connect()
    try:
        return {
            state.value: await db.count_agents(state) for state in m.AgentState
        }
    finally:
        await db.close()


def test_cli_council_run_llm_error_marks_agents_dead(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_env(monkeypatch, tmp_path)
    monkeypatch.setattr("harness.cli.LLMClient", _FailingLLM)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "sample.c").write_text("int main(void) { return 0; }\n")

    assert main(["council", "run", "--checkout", str(checkout)]) == 1

    err = capsys.readouterr().err
    assert "council" in err.lower()
    assert "spawned" not in err.lower()

    counts = asyncio.run(_state_counts(tmp_path / "cli.db"))
    assert counts["spawned"] == 0
    assert counts["dead"] == 4


def test_cli_council_run_success_marks_agents_dead(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_env(monkeypatch, tmp_path)
    monkeypatch.setattr("harness.cli.LLMClient", _NoopLLM)

    async def _fake_run_council(*args: object, **kwargs: object) -> CouncilResult:
        return CouncilResult(
            topic_id="tp_council",
            candidates=1,
            survived=1,
            killed=0,
            exploited=1,
            precision=1.0,
        )

    monkeypatch.setattr("harness.cli.run_council", _fake_run_council)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "sample.c").write_text("int main(void) { return 0; }\n")

    assert main(["council", "run", "--checkout", str(checkout)]) == 0

    out = capsys.readouterr().out
    assert "council" in out.lower()

    counts = asyncio.run(_state_counts(tmp_path / "cli.db"))
    assert counts["spawned"] == 0
    assert counts["dead"] == 4


def _fake_result(pocs_run: int, pocs_succeeded: int) -> CouncilResult:
    result = CouncilResult(
        topic_id="tp_council",
        candidates=1,
        survived=1,
        killed=0,
        exploited=1,
        precision=1.0,
    )
    setattr(result, "pocs_run", pocs_run)
    setattr(result, "pocs_succeeded", pocs_succeeded)
    return result


def test_cli_council_run_no_docker_passes_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_env(monkeypatch, tmp_path)
    monkeypatch.setattr("harness.cli.LLMClient", _NoopLLM)
    seen: dict[str, object] = {}

    async def _fake_run_council(*args: object, **kwargs: object) -> CouncilResult:
        seen.update(kwargs)
        return _fake_result(2, 1)

    monkeypatch.setattr("harness.cli.run_council", _fake_run_council)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "sample.c").write_text("int main(void) { return 0; }\n")

    assert main(["council", "run", "--checkout", str(checkout), "--no-docker"]) == 0

    out = capsys.readouterr().out
    assert "pocs=2 ok=1" in out
    assert seen["containers"] is None


def test_cli_council_run_builds_container_manager(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_env(monkeypatch, tmp_path)
    monkeypatch.setattr("harness.cli.LLMClient", _NoopLLM)
    seen: dict[str, object] = {}

    async def _fake_run_council(*args: object, **kwargs: object) -> CouncilResult:
        seen.update(kwargs)
        return _fake_result(0, 0)

    monkeypatch.setattr("harness.cli.run_council", _fake_run_council)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "sample.c").write_text("int main(void) { return 0; }\n")

    assert main(["council", "run", "--checkout", str(checkout)]) == 0

    assert isinstance(seen["containers"], ContainerManager)
    assert "pocs=0 ok=0" in capsys.readouterr().out


