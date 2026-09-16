from __future__ import annotations

import asyncio
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import models as m

if TYPE_CHECKING:
    from .config import Settings
    from .db import Database
    from .logging_ import EventLogger

_AGENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_CONCURRENT_EXEC = 4

_STATUS_MAP: dict[str, m.ContainerStatus] = {
    "created": m.ContainerStatus.CREATED,
    "running": m.ContainerStatus.RUNNING,
    "restarting": m.ContainerStatus.RUNNING,
    "removing": m.ContainerStatus.REMOVED,
    "removed": m.ContainerStatus.REMOVED,
    "paused": m.ContainerStatus.STOPPED,
    "exited": m.ContainerStatus.STOPPED,
    "dead": m.ContainerStatus.STOPPED,
}


def _validate_agent_id(agent_id: str) -> str:
    if not _AGENT_ID_PATTERN.fullmatch(agent_id):
        raise ValueError(f"invalid agent id: {agent_id!r}")
    return agent_id


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str


class ContainerManager:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        logger: EventLogger | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.logger = logger
        self._client: Any = None
        self._client_lock = threading.Lock()
        self._exec_semaphores: dict[str, asyncio.Semaphore] = {}

    def workspace_for(self, agent_id: str) -> Path:
        _validate_agent_id(agent_id)
        workspace_root = self.settings.workspace_dir
        resolved_root = workspace_root.resolve()
        resolved = (workspace_root / agent_id).resolve()
        if not resolved.is_relative_to(resolved_root):
            raise ValueError(f"agent id escapes workspace: {agent_id!r}")
        resolved.mkdir(parents=True, exist_ok=True)
        resolved.chmod(0o777)
        return resolved

    def _log(self, kind: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.event(kind, **fields)

    def _client_sync(self) -> Any:
        import docker

        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = docker.from_env()
        return self._client

    def _run_container_sync(self, image: str, workspace: Path) -> Any:
        client = self._client_sync()
        return client.containers.run(
            image,
            command=["sleep", "infinity"],
            detach=True,
            user=self.settings.sandbox_user,
            mem_limit=self.settings.sandbox_mem_limit,
            pids_limit=self.settings.sandbox_pids_limit,
            cpu_period=self.settings.sandbox_cpu_period,
            cpu_quota=self.settings.sandbox_cpu_quota,
            read_only=self.settings.sandbox_read_only,
            tmpfs={"/tmp": f"size={self.settings.sandbox_tmpfs_size}"},
            network_mode=self.settings.sandbox_network,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            volumes={str(workspace): {"bind": "/work", "mode": "rw"}},
            working_dir="/work",
        )

    def _force_remove_sync(self, container_id: str) -> None:
        container = self._client_sync().containers.get(container_id)
        container.remove(force=True)

    def _exec_semaphore(self, container_id: str) -> asyncio.Semaphore:
        semaphore = self._exec_semaphores.get(container_id)
        if semaphore is None:
            semaphore = asyncio.Semaphore(_MAX_CONCURRENT_EXEC)
            self._exec_semaphores[container_id] = semaphore
        return semaphore

    async def _set_status(
        self, container_id: str, status: m.ContainerStatus, action: str
    ) -> None:
        await self.db.set_container_status(container_id, status)
        self._log(
            "container",
            action=action,
            container_id=container_id,
            status=status.value,
        )

    async def create(self, agent_id: str, image: str | None = None) -> m.Container:
        resolved_image = image or self.settings.docker_image
        workspace = self.workspace_for(agent_id)
        container_id = f"fake-{agent_id}"
        raw: Any = None
        if self.settings.docker_enabled:
            raw = await asyncio.to_thread(
                self._run_container_sync, resolved_image, workspace
            )
            container_id = raw.id

        ts = time.time()
        container = m.Container(
            id=container_id,
            agent_id=agent_id,
            image=resolved_image,
            workspace_path=str(workspace),
            status=m.ContainerStatus.RUNNING,
            created_at=ts,
            updated_at=ts,
        )
        try:
            await self.db.upsert_container(
                container_id=container_id,
                agent_id=agent_id,
                image=resolved_image,
                workspace_path=str(workspace),
                status=m.ContainerStatus.RUNNING,
            )
        except BaseException:
            if raw is not None:
                await asyncio.to_thread(self._force_remove_sync, container_id)
            raise
        self._log(
            "container",
            action="create",
            container_id=container_id,
            agent_id=agent_id,
            image=resolved_image,
            status=m.ContainerStatus.RUNNING.value,
            fake=raw is None,
        )
        return container

    async def start(self, container_id: str) -> None:
        if self.settings.docker_enabled:
            await asyncio.to_thread(self._start_sync, container_id)
        await self._set_status(container_id, m.ContainerStatus.RUNNING, "start")

    def _start_sync(self, container_id: str) -> None:
        container = self._client_sync().containers.get(container_id)
        container.start()

    async def stop(self, container_id: str, timeout: int = 10) -> None:
        if self.settings.docker_enabled:
            await asyncio.to_thread(self._stop_sync, container_id, timeout)
        await self._set_status(container_id, m.ContainerStatus.STOPPED, "stop")

    def _stop_sync(self, container_id: str, timeout: int) -> None:
        container = self._client_sync().containers.get(container_id)
        container.stop(timeout=timeout)

    async def remove(self, container_id: str) -> None:
        if self.settings.docker_enabled:
            await asyncio.to_thread(self._force_remove_sync, container_id)
        await self._set_status(container_id, m.ContainerStatus.REMOVED, "remove")

    async def exec(
        self, container_id: str, argv: list[str], timeout: int = 60
    ) -> ExecResult:
        if not self.settings.docker_enabled:
            return ExecResult(0, "", "")

        def _exec() -> ExecResult:
            container = self._client_sync().containers.get(container_id)
            result = container.exec_run(argv, demux=True, workdir="/work")
            output = result.output if result.output is not None else (None, None)
            raw_out, raw_err = output
            return ExecResult(
                exit_code=result.exit_code if result.exit_code is not None else -1,
                stdout=(raw_out or b"").decode("utf-8", errors="replace"),
                stderr=(raw_err or b"").decode("utf-8", errors="replace"),
            )

        semaphore = self._exec_semaphore(container_id)
        async with semaphore:
            try:
                return await asyncio.wait_for(asyncio.to_thread(_exec), timeout=timeout)
            except asyncio.TimeoutError:
                return ExecResult(-1, "", f"exec timed out after {timeout}s")

    async def status(self, container_id: str) -> m.ContainerStatus:
        if not self.settings.docker_enabled:
            row = await self.db.get_container(container_id)
            if row is None:
                return m.ContainerStatus.REMOVED
            return row.status

        def _status() -> str:
            container = self._client_sync().containers.get(container_id)
            return str(container.status)

        try:
            raw = await asyncio.to_thread(_status)
        except Exception:
            return m.ContainerStatus.REMOVED
        return _STATUS_MAP.get(raw, m.ContainerStatus.STOPPED)

    async def aclose(self) -> None:
        client = self._client
        if client is not None:
            self._client = None
            await asyncio.to_thread(client.close)
