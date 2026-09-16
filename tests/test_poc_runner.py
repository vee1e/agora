from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from harness import models as m
from harness.containers import ExecResult
from harness.council.poc import PocResult, run_and_correct_poc
from harness.council.task import _run_poc_safe
from harness.db import Database
from harness.llm import LLMResponse
from harness.models import FindingStatus, Usage


class _ScriptedLLM:
    def __init__(self, scripts: list[str]) -> None:
        self._scripts = list(scripts)
        self.calls = 0

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        self.calls += 1
        if not self._scripts:
            raise AssertionError("scripted LLM ran out of responses")
        return LLMResponse(
            text=self._scripts.pop(0), model="test-model", usage=Usage(1, 1)
        )

    async def aclose(self) -> None:
        return None


class _FakeSandbox:
    def __init__(self, outcomes: list[ExecResult | Exception], workspace_dir: Path) -> None:
        self.settings = SimpleNamespace(
            workspace_dir=workspace_dir, docker_enabled=True
        )
        self._outcomes = list(outcomes)
        self.exec_calls: list[list[str]] = []
        self.created: list[str] = []
        self.removed: list[str] = []

    async def create(self, agent_id: str) -> m.Container:
        self.created.append(agent_id)
        return m.Container(
            id=f"fake-{agent_id}",
            agent_id=agent_id,
            image="img",
            workspace_path=str(self.settings.workspace_dir / agent_id),
            status=m.ContainerStatus.RUNNING,
        )

    async def exec(
        self, container_id: str, argv: list[str], timeout: int = 60
    ) -> ExecResult:
        self.exec_calls.append(argv)
        if not self._outcomes:
            raise AssertionError("fake sandbox ran out of exec outcomes")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def remove(self, container_id: str) -> None:
        self.removed.append(container_id)

    async def aclose(self) -> None:
        return None


def _finding(db_path: Path, finding_id: str = "fd_poc") -> m.Finding:
    return m.Finding(
        id=finding_id,
        topic_id="topic",
        agent_id="exploit",
        file="sample.c",
        line=6,
        sink="strcpy",
        call_path=["run"],
        status=FindingStatus.SURVIVED,
        created_at=0.0,
    )


def _checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout-src"
    checkout.mkdir()
    (checkout / "sample.c").write_text("int main(void) { return 0; }\n")
    return checkout


async def test_poc_runner_corrects_and_succeeds(db: Database, tmp_path: Path) -> None:
    finding = _finding(tmp_path)
    await db.create_finding(
        "topic",
        "exploit",
        file="sample.c",
        line=6,
        sink="strcpy",
        call_path=["run"],
        finding_id=finding.id,
    )
    await db.set_finding_status(finding.id, FindingStatus.SURVIVED)
    checkout = _checkout(tmp_path)
    workspace = tmp_path / "workspaces" / "exploit"
    llm = _ScriptedLLM(["print('# /work/checkout x')", "print('# /work/checkout\nPOC_SUCCESS')"])
    sandbox = _FakeSandbox(
        [ExecResult(1, "", "boom"), ExecResult(0, "POC_SUCCESS", "")],
        tmp_path / "workspaces",
    )

    result = await run_and_correct_poc(
        llm,
        db,
        sandbox,
        "fake-exploit",
        finding,
        checkout,
        workspace,
        "exploit",
    )

    assert isinstance(result, PocResult)
    assert result.success is True
    assert len(result.attempts) == 2
    assert [attempt.exit_code for attempt in result.attempts] == [1, 0]
    assert llm.calls == 2

    await db.set_finding_status(finding.id, FindingStatus.EXPLOITED)
    stored = await db.get_finding(finding.id)
    assert stored is not None
    assert stored.status is FindingStatus.EXPLOITED

    runs = await db.list_poc_runs(finding.id)
    assert len(runs) == 2
    assert [row["exit_code"] for row in runs] == [1, 0]
    assert [row["success"] for row in runs] == [0, 1]

    assert (workspace / "checkout" / "sample.c").exists()
    assert sandbox.exec_calls
    assert sandbox.exec_calls[0][0] == "python3"


async def test_poc_runner_refuses_invalid_finding_id(
    db: Database, tmp_path: Path
) -> None:
    finding = _finding(tmp_path, finding_id="../evil")
    checkout = _checkout(tmp_path)
    sandbox = _FakeSandbox([], tmp_path / "workspaces")
    llm = _ScriptedLLM([])

    result = await run_and_correct_poc(
        llm,
        db,
        sandbox,
        "fake-exploit",
        finding,
        checkout,
        tmp_path / "workspaces" / "exploit",
        "exploit",
    )

    assert result.success is False
    assert result.reason == "invalid finding id"
    assert sandbox.exec_calls == []
    assert llm.calls == 0


async def test_run_poc_safe_absorbs_exec_exception(
    db: Database, tmp_path: Path
) -> None:
    finding = _finding(tmp_path)
    checkout = _checkout(tmp_path)
    sandbox = _FakeSandbox(
        [RuntimeError("docker exploded")], tmp_path / "workspaces"
    )
    llm = _ScriptedLLM(["print('hi')"])

    result = await _run_poc_safe(
        llm,
        db,
        sandbox,
        "fake-exploit",
        finding,
        str(checkout),
        tmp_path / "workspaces" / "exploit",
        "exploit",
        None,
    )

    assert result is None
