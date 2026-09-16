from __future__ import annotations

import sys
from typing import Any

import pytest

from harness import models as m
from harness.config import Settings
from harness.containers import ContainerManager, ExecResult
from harness.db import Database


class FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def event(self, kind: str, **fields: Any) -> None:
        self.events.append((kind, fields))


def test_fixture_runs_without_docker(settings: Settings) -> None:
    assert settings.docker_enabled is False


def test_workspace_for_creates_directory(settings: Settings, db: Database) -> None:
    manager = ContainerManager(settings, db)
    path = manager.workspace_for("ag_ws")
    assert path == settings.workspace_dir / "ag_ws"
    assert path.is_dir()


def test_workspace_for_rejects_traversal(settings: Settings, db: Database) -> None:
    manager = ContainerManager(settings, db)
    for bad in ["../escape", "a/b", "..", "", "a b", "a" * 65, "ag.dot"]:
        with pytest.raises(ValueError):
            manager.workspace_for(bad)


def test_workspace_for_rejects_absolute_path(settings: Settings, db: Database) -> None:
    manager = ContainerManager(settings, db)
    with pytest.raises(ValueError):
        manager.workspace_for("/etc/passwd")


async def test_create_rejects_bad_agent_id(settings: Settings, db: Database) -> None:
    manager = ContainerManager(settings, db)
    with pytest.raises(ValueError):
        await manager.create("../escape")
    assert await db.list_containers() == []


async def test_fake_status_reads_db(settings: Settings, db: Database) -> None:
    manager = ContainerManager(settings, db)
    container = await manager.create("ag_db_status")
    await db.set_container_status(container.id, m.ContainerStatus.STOPPED)
    assert await manager.status(container.id) is m.ContainerStatus.STOPPED


async def test_fake_status_removed_after_remove(settings: Settings, db: Database) -> None:
    manager = ContainerManager(settings, db)
    container = await manager.create("ag_rm_status")
    assert await manager.status(container.id) is m.ContainerStatus.RUNNING
    await manager.remove(container.id)
    assert await manager.status(container.id) is m.ContainerStatus.REMOVED


async def test_create_returns_fake_running_container(settings: Settings, db: Database) -> None:
    manager = ContainerManager(settings, db)
    container = await manager.create("ag_create")
    assert container.id == "fake-ag_create"
    assert container.agent_id == "ag_create"
    assert container.image == settings.docker_image
    assert container.status is m.ContainerStatus.RUNNING
    assert container.workspace_path == str(settings.workspace_dir / "ag_create")
    assert (settings.workspace_dir / "ag_create").is_dir()


async def test_create_upserts_container_row(settings: Settings, db: Database) -> None:
    manager = ContainerManager(settings, db)
    await manager.create("ag_row")
    row = await db.get_container("fake-ag_row")
    assert row is not None
    assert row.agent_id == "ag_row"
    assert row.image == settings.docker_image
    assert row.status is m.ContainerStatus.RUNNING


async def test_create_honours_explicit_image(settings: Settings, db: Database) -> None:
    manager = ContainerManager(settings, db)
    container = await manager.create("ag_img", image="custom:1")
    row = await db.get_container(container.id)
    assert container.image == "custom:1"
    assert row is not None and row.image == "custom:1"


async def test_exec_returns_empty_success(settings: Settings, db: Database) -> None:
    manager = ContainerManager(settings, db)
    container = await manager.create("ag_exec")
    result = await manager.exec(container.id, ["ls", "-la"])
    assert result == ExecResult(0, "", "")
    assert not result.stdout
    assert not result.stderr


async def test_status_is_running(settings: Settings, db: Database) -> None:
    manager = ContainerManager(settings, db)
    container = await manager.create("ag_status")
    assert await manager.status(container.id) is m.ContainerStatus.RUNNING


async def test_stop_then_start_then_remove_updates_db(settings: Settings, db: Database) -> None:
    manager = ContainerManager(settings, db)
    container = await manager.create("ag_life")

    await manager.stop(container.id)
    stopped = await db.get_container(container.id)
    assert stopped is not None and stopped.status is m.ContainerStatus.STOPPED

    await manager.start(container.id)
    running = await db.get_container(container.id)
    assert running is not None and running.status is m.ContainerStatus.RUNNING

    await manager.remove(container.id)
    removed = await db.get_container(container.id)
    assert removed is not None and removed.status is m.ContainerStatus.REMOVED


async def test_create_logs_event(settings: Settings, db: Database) -> None:
    logger = FakeLogger()
    manager = ContainerManager(settings, db, logger=logger)
    await manager.create("ag_log")
    kinds = [kind for kind, _ in logger.events]
    assert "container" in kinds
    fields = logger.events[0][1]
    assert fields["action"] == "create"
    assert fields["container_id"] == "fake-ag_log"


async def test_aclose_without_client_is_noop(settings: Settings, db: Database) -> None:
    manager = ContainerManager(settings, db)
    await manager.create("ag_close")
    await manager.aclose()


async def test_disabled_path_never_imports_docker(settings: Settings, db: Database) -> None:
    sys.modules.pop("docker", None)
    manager = ContainerManager(settings, db)
    container = await manager.create("ag_noimport")
    await manager.start(container.id)
    await manager.stop(container.id)
    await manager.exec(container.id, ["true"])
    await manager.status(container.id)
    await manager.remove(container.id)
    await manager.aclose()
    assert "docker" not in sys.modules
