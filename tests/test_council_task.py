from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from harness import models as m
from harness.config import Settings
from harness.containers import ExecResult
from harness.council.agents import CouncilResult
from harness.council.graph import CodeGraph
from harness.council.task import (
    _announce,
    _workspace_for,
    run_council,
)
from harness.council.tools import CouncilTools
from harness.db import Database
from harness.message_board import MessageBoard, RateLimited
from harness.llm import LLMResponse
from harness.logging_ import EventLogger
from harness.mailbox import Mailbox
from harness.models import FindingStatus, Usage

C_SOURCE = """#include <string.h>
#include <stdlib.h>

void copy_payload(char *input) {
    char buf[32];
    strcpy(buf, input);
    strcat(buf, input);
}

void dispatch(char *arg) {
    copy_payload(arg);
}

int run(int argc, char **argv) {
    char *heap = malloc(128);
    dispatch(argv[1]);
    printf(argv[1]);
    system(argv[1]);
    free(heap);
    return 0;
}
"""


class ScriptedLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        agent_id: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "agent_id": agent_id})
        if not self._responses:
            raise AssertionError("scripted LLM ran out of responses")
        return self._responses.pop(0)


def _response(text: str) -> LLMResponse:
    return LLMResponse(
        text=text,
        model="deepseek-v4.1-flash",
        usage=Usage(tokens_in=1, tokens_out=1),
    )


def _verdict(verdict: str, reason: str = "r") -> str:
    return json.dumps({"verdict": verdict, "reason": reason})


@pytest.fixture
def tools(tmp_path: Path) -> CouncilTools:
    (tmp_path / "sample.c").write_text(C_SOURCE, encoding="utf-8")
    (tmp_path / "a.c").write_text(C_SOURCE, encoding="utf-8")
    (tmp_path / "b.c").write_text(C_SOURCE, encoding="utf-8")
    graph = CodeGraph(tmp_path)
    graph.build()
    return CouncilTools(graph)


async def _seed_agents(db: Database) -> None:
    await db.create_agent(m.AgentRole.FINDER, agent_id="finder")
    await db.create_agent(m.AgentRole.KILLER, agent_id="killer")
    await db.create_agent(m.AgentRole.JUDGE, agent_id="judge")
    await db.create_agent(m.AgentRole.EXPLOIT, agent_id="exploit")


def _message_board(settings: Settings, db: Database, logger: EventLogger) -> MessageBoard:
    return MessageBoard(db, Mailbox(db, logger), settings, logger)


async def test_run_council_end_to_end(
    settings: Settings,
    db: Database,
    tools: CouncilTools,
    tmp_path: Path,
    logger: EventLogger,
) -> None:
    await _seed_agents(db)
    topic = await db.create_topic("audit")
    message_board = _message_board(settings, db, logger)

    candidates = json.dumps(
        [
            {
                "file": "sample.c",
                "line": 6,
                "sink": "strcpy",
                "call_path": ["run", "dispatch", "copy_payload"],
            },
            {
                "file": "sample.c",
                "line": 18,
                "sink": "system",
                "call_path": ["run"],
            },
        ]
    )
    llm = ScriptedLLM(
        [
            _response(candidates),
            _response("[]"),
            _response(_verdict("survives", "reachable")),
            _response(_verdict("survives", "path holds")),
            _response(_verdict("killed", "bound check")),
            _response(_verdict("killed", "dead path")),
            _response("print('poc')"),
        ]
    )

    result = await run_council(
        llm,
        tools,
        db,
        message_board,
        topic.id,
        str(tmp_path),
        "finder",
        "killer",
        "judge",
        "exploit",
    )

    assert isinstance(result, CouncilResult)
    assert result.candidates == 2
    assert result.survived == 1
    assert result.killed == 1
    assert result.exploited == 0
    assert result.pocs_written == 1
    assert result.precision == 0.5
    assert len(result.findings) == 2

    survived = [f for f in result.findings if f.status is FindingStatus.SURVIVED]
    killed = [f for f in result.findings if f.status is FindingStatus.KILLED]
    assert len(survived) == 1
    assert len(killed) == 1
    poc = settings.workspace_dir / "exploit" / f"poc_{survived[0].id}.py"
    assert poc.exists()

    posts = await db.list_posts(topic.id)
    assert len(posts) == 1
    assert "Finding" in posts[0].body
    assert survived[0].id in posts[0].body


async def test_run_council_precision_zero_without_survivors(
    settings: Settings,
    db: Database,
    tools: CouncilTools,
    tmp_path: Path,
    logger: EventLogger,
) -> None:
    await _seed_agents(db)
    topic = await db.create_topic("audit")
    message_board = _message_board(settings, db, logger)

    llm = ScriptedLLM(
        [
            _response(
                json.dumps(
                    [
                        {
                            "file": "sample.c",
                            "line": 6,
                            "sink": "strcpy",
                            "call_path": ["run"],
                        }
                    ]
                )
            ),
            _response("[]"),
            _response(_verdict("killed", "clamped")),
            _response(_verdict("killed", "dead")),
        ]
    )

    result = await run_council(
        llm,
        tools,
        db,
        message_board,
        topic.id,
        str(tmp_path),
        "finder",
        "killer",
        "judge",
        "exploit",
    )

    assert result.candidates == 1
    assert result.survived == 0
    assert result.killed == 1
    assert result.exploited == 0
    assert result.precision == 0.0


async def test_run_council_groups_two_sinks_on_one_path(
    settings: Settings,
    db: Database,
    tools: CouncilTools,
    tmp_path: Path,
    logger: EventLogger,
) -> None:
    await _seed_agents(db)
    topic = await db.create_topic("audit")
    message_board = _message_board(settings, db, logger)

    shared_path = ["run", "dispatch", "copy_payload"]
    candidates = json.dumps(
        [
            {
                "file": "sample.c",
                "line": 6,
                "sink": "strcpy",
                "call_path": shared_path,
            },
            {
                "file": "sample.c",
                "line": 7,
                "sink": "strcat",
                "call_path": shared_path,
            },
        ]
    )
    llm = ScriptedLLM(
        [
            _response(candidates),
            _response("[]"),
            _response(_verdict("survives", "reachable")),
            _response(_verdict("survives", "path holds")),
            _response(_verdict("survives", "reachable")),
            _response(_verdict("survives", "path holds")),
            _response("print('poc')"),
            _response("print('poc')"),
        ]
    )

    result = await run_council(
        llm,
        tools,
        db,
        message_board,
        topic.id,
        str(tmp_path),
        "finder",
        "killer",
        "judge",
        "exploit",
    )

    assert result.survived == 2
    assert len(result.notes) == 1
    note = result.notes[0]
    assert "Grouped note" in note
    assert "run -&gt; dispatch -&gt; copy_payload" in note
    assert "strcpy" in note and "strcat" in note


async def test_run_council_no_note_for_distinct_paths(
    settings: Settings,
    db: Database,
    tools: CouncilTools,
    tmp_path: Path,
    logger: EventLogger,
) -> None:
    await _seed_agents(db)
    topic = await db.create_topic("audit")
    message_board = _message_board(settings, db, logger)

    candidates = json.dumps(
        [
            {
                "file": "sample.c",
                "line": 6,
                "sink": "strcpy",
                "call_path": ["run", "dispatch"],
            },
            {
                "file": "sample.c",
                "line": 18,
                "sink": "system",
                "call_path": ["run"],
            },
        ]
    )
    llm = ScriptedLLM(
        [
            _response(candidates),
            _response("[]"),
            _response(_verdict("killed", "clamped")),
            _response(_verdict("killed", "clamped")),
            _response(_verdict("killed", "clamped")),
            _response(_verdict("killed", "clamped")),
        ]
    )

    result = await run_council(
        llm,
        tools,
        db,
        message_board,
        topic.id,
        str(tmp_path),
        "finder",
        "killer",
        "judge",
        "exploit",
    )

    assert result.notes == []


async def test_run_council_undecided_excluded_from_precision(
    settings: Settings,
    db: Database,
    tools: CouncilTools,
    tmp_path: Path,
    logger: EventLogger,
) -> None:
    await _seed_agents(db)
    topic = await db.create_topic("audit")
    message_board = _message_board(settings, db, logger)

    candidates = json.dumps(
        [
            {"file": "sample.c", "line": 6, "sink": "strcpy", "call_path": ["run"]},
            {"file": "sample.c", "line": 18, "sink": "system", "call_path": ["run"]},
            {"file": "sample.c", "line": 17, "sink": "printf", "call_path": ["run"]},
        ]
    )
    llm = ScriptedLLM(
        [
            _response(candidates),
            _response("[]"),
            _response(_verdict("survives", "reachable")),
            _response(_verdict("survives", "path holds")),
            _response("no verdict here"),
            _response("still cannot decide"),
            _response(_verdict("killed", "clamped")),
            _response(_verdict("killed", "dead path")),
            _response("print('poc')"),
        ]
    )

    result = await run_council(
        llm,
        tools,
        db,
        message_board,
        topic.id,
        str(tmp_path),
        "finder",
        "killer",
        "judge",
        "exploit",
    )

    assert result.candidates == 3
    assert result.survived == 1
    assert result.killed == 1
    assert result.exploited == 0
    assert result.pocs_written == 1
    assert result.precision == 0.5
    statuses = {finding.sink: finding.status for finding in result.findings}
    assert statuses["strcpy"] is FindingStatus.SURVIVED
    assert statuses["system"] is FindingStatus.CANDIDATE
    assert statuses["printf"] is FindingStatus.KILLED


async def test_run_council_notes_only_among_survivors(
    settings: Settings,
    db: Database,
    tools: CouncilTools,
    tmp_path: Path,
    logger: EventLogger,
) -> None:
    await _seed_agents(db)
    topic = await db.create_topic("audit")
    message_board = _message_board(settings, db, logger)

    shared_path = ["run", "dispatch", "copy_payload"]
    candidates = json.dumps(
        [
            {
                "file": "sample.c",
                "line": 6,
                "sink": "strcpy",
                "call_path": shared_path,
            },
            {
                "file": "sample.c",
                "line": 7,
                "sink": "strcat",
                "call_path": shared_path,
            },
        ]
    )
    llm = ScriptedLLM(
        [
            _response(candidates),
            _response("[]"),
            _response(_verdict("survives", "reachable")),
            _response(_verdict("survives", "path holds")),
            _response(_verdict("killed", "clamped")),
            _response(_verdict("killed", "dead path")),
            _response("print('poc')"),
        ]
    )

    result = await run_council(
        llm,
        tools,
        db,
        message_board,
        topic.id,
        str(tmp_path),
        "finder",
        "killer",
        "judge",
        "exploit",
    )

    assert result.survived == 1
    assert result.killed == 1
    assert result.notes == []


async def test_run_council_notes_do_not_group_across_files(
    settings: Settings,
    db: Database,
    tools: CouncilTools,
    tmp_path: Path,
    logger: EventLogger,
) -> None:
    await _seed_agents(db)
    topic = await db.create_topic("audit")
    message_board = _message_board(settings, db, logger)

    shared_path = ["run", "dispatch"]
    candidates = json.dumps(
        [
            {"file": "a.c", "line": 6, "sink": "strcpy", "call_path": shared_path},
            {"file": "b.c", "line": 7, "sink": "strcat", "call_path": shared_path},
        ]
    )
    llm = ScriptedLLM(
        [
            _response(candidates),
            _response("[]"),
            _response(_verdict("survives", "reachable")),
            _response(_verdict("survives", "path holds")),
            _response(_verdict("survives", "reachable")),
            _response(_verdict("survives", "path holds")),
            _response("print('poc')"),
            _response("print('poc')"),
        ]
    )

    result = await run_council(
        llm,
        tools,
        db,
        message_board,
        topic.id,
        str(tmp_path),
        "finder",
        "killer",
        "judge",
        "exploit",
    )

    assert result.survived == 2
    assert result.notes == []


async def test_run_council_subscribes_agents(
    settings: Settings,
    db: Database,
    tools: CouncilTools,
    tmp_path: Path,
    logger: EventLogger,
) -> None:
    await _seed_agents(db)
    topic = await db.create_topic("audit")
    message_board = _message_board(settings, db, logger)
    llm = ScriptedLLM([_response("[]")])

    await run_council(
        llm,
        tools,
        db,
        message_board,
        topic.id,
        str(tmp_path),
        "finder",
        "killer",
        "judge",
        "exploit",
    )

    subscribers = await db.list_subscribers(topic.id)
    assert set(subscribers) == {"finder", "killer", "judge", "exploit"}


class _CapturingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def event(self, kind: str, **fields: Any) -> None:
        self.events.append((kind, fields))


class _RejectingMessageBoard:
    def __init__(self, logger: _CapturingLogger) -> None:
        self.logger = logger

    async def post(self, *args: Any, **kwargs: Any) -> None:
        raise RateLimited("over rate")


async def test_announce_logs_rate_limit(db: Database) -> None:
    logger = _CapturingLogger()
    finding = await db.create_finding(
        "topic",
        "finder",
        file="sample.c",
        line=6,
        sink="strcpy",
        call_path=["run"],
    )

    await _announce(_RejectingMessageBoard(logger), "topic", "finder", finding)

    kinds = [kind for kind, _ in logger.events]
    assert kinds == ["council_announce_rate_limited"]


def test_workspace_for_uses_settings_and_rejects_bad_id(
    settings: Settings, db: Database, logger: EventLogger
) -> None:
    message_board = _message_board(settings, db, logger)

    assert _workspace_for(message_board, "exploit") == settings.workspace_dir / "exploit"
    with pytest.raises(ValueError):
        _workspace_for(message_board, "../evil")


class _TaskSandbox:
    def __init__(
        self,
        outcomes: list[ExecResult],
        workspace_dir: Path,
        create_error: Exception | None = None,
    ) -> None:
        self.settings = SimpleNamespace(
            workspace_dir=workspace_dir, docker_enabled=True
        )
        self._outcomes = list(outcomes)
        self._create_error = create_error
        self.exec_calls: list[list[str]] = []
        self.created: list[str] = []
        self.removed: list[str] = []

    async def create(self, agent_id: str) -> m.Container:
        if self._create_error is not None:
            raise self._create_error
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
        return self._outcomes.pop(0)

    async def remove(self, container_id: str) -> None:
        self.removed.append(container_id)

    async def aclose(self) -> None:
        return None


def _poc_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "poc-src"
    checkout.mkdir()
    (checkout / "sample.c").write_text(C_SOURCE, encoding="utf-8")
    return checkout


async def test_run_council_sandbox_failure_keeps_survived(
    settings: Settings,
    db: Database,
    tools: CouncilTools,
    tmp_path: Path,
    logger: EventLogger,
) -> None:
    await _seed_agents(db)
    topic = await db.create_topic("audit")
    message_board = _message_board(settings, db, logger)
    candidates = json.dumps(
        [
            {
                "file": "sample.c",
                "line": 6,
                "sink": "strcpy",
                "call_path": ["run"],
            }
        ]
    )
    llm = ScriptedLLM(
        [
            _response(candidates),
            _response("[]"),
            _response(_verdict("survives", "reachable")),
            _response(_verdict("survives", "path holds")),
            _response("# /work/checkout\nprint('a')"),
            _response("# /work/checkout\nprint('b')"),
            _response("# /work/checkout\nprint('c')"),
        ]
    )
    sandbox = _TaskSandbox(
        [ExecResult(1, "", "no"), ExecResult(1, "", "no"), ExecResult(1, "", "no")],
        settings.workspace_dir,
    )

    result = await run_council(
        llm,
        tools,
        db,
        message_board,
        topic.id,
        str(_poc_checkout(tmp_path)),
        "finder",
        "killer",
        "judge",
        "exploit",
        containers=sandbox,
    )

    assert result.survived == 1
    assert result.exploited == 0
    assert result.pocs_run == 1
    assert result.pocs_succeeded == 0
    assert result.findings[0].status is FindingStatus.SURVIVED
    assert sandbox.created == ["exploit"]
    assert sandbox.removed == ["fake-exploit"]


async def test_run_council_sandbox_success_sets_exploited(
    settings: Settings,
    db: Database,
    tools: CouncilTools,
    tmp_path: Path,
    logger: EventLogger,
) -> None:
    await _seed_agents(db)
    topic = await db.create_topic("audit")
    message_board = _message_board(settings, db, logger)
    candidates = json.dumps(
        [
            {
                "file": "sample.c",
                "line": 6,
                "sink": "strcpy",
                "call_path": ["run"],
            }
        ]
    )
    llm = ScriptedLLM(
        [
            _response(candidates),
            _response("[]"),
            _response(_verdict("survives", "reachable")),
            _response(_verdict("survives", "path holds")),
            _response("# /work/checkout\nprint('POC_SUCCESS')"),
        ]
    )
    sandbox = _TaskSandbox(
        [ExecResult(0, "POC_SUCCESS", "")], settings.workspace_dir
    )
    checkout = _poc_checkout(tmp_path)

    result = await run_council(
        llm,
        tools,
        db,
        message_board,
        topic.id,
        str(checkout),
        "finder",
        "killer",
        "judge",
        "exploit",
        containers=sandbox,
    )

    assert result.survived == 1
    assert result.exploited == 1
    assert result.pocs_run == 1
    assert result.pocs_succeeded == 1
    assert result.findings[0].status is FindingStatus.EXPLOITED
    assert (settings.workspace_dir / "exploit" / "checkout" / "sample.c").exists()
    assert sandbox.exec_calls


async def test_run_council_creates_one_container_and_removes_it(
    settings: Settings,
    db: Database,
    tools: CouncilTools,
    tmp_path: Path,
    logger: EventLogger,
) -> None:
    await _seed_agents(db)
    topic = await db.create_topic("audit")
    message_board = _message_board(settings, db, logger)
    candidates = json.dumps(
        [
            {
                "file": "sample.c",
                "line": 6,
                "sink": "strcpy",
                "call_path": ["run"],
            },
            {
                "file": "sample.c",
                "line": 7,
                "sink": "strcat",
                "call_path": ["run"],
            },
        ]
    )
    llm = ScriptedLLM(
        [
            _response(candidates),
            _response("[]"),
            _response(_verdict("survives", "reachable")),
            _response(_verdict("survives", "path holds")),
            _response(_verdict("survives", "reachable")),
            _response(_verdict("survives", "path holds")),
            _response("# /work/checkout\nprint('a')"),
            _response("# /work/checkout\nprint('b')"),
            _response("# /work/checkout\nprint('c')"),
            _response("# /work/checkout\nprint('POC_SUCCESS')"),
        ]
    )
    sandbox = _TaskSandbox(
        [
            ExecResult(1, "", "no"),
            ExecResult(1, "", "no"),
            ExecResult(1, "", "no"),
            ExecResult(0, "POC_SUCCESS", ""),
        ],
        settings.workspace_dir,
    )

    result = await run_council(
        llm,
        tools,
        db,
        message_board,
        topic.id,
        str(_poc_checkout(tmp_path)),
        "finder",
        "killer",
        "judge",
        "exploit",
        containers=sandbox,
    )

    assert result.survived == 2
    assert result.exploited == 1
    assert result.pocs_run == 2
    assert result.pocs_succeeded == 1
    assert sandbox.created == ["exploit"]
    assert sandbox.removed == ["fake-exploit"]
    statuses = {finding.sink: finding.status for finding in result.findings}
    assert statuses["strcpy"] is FindingStatus.SURVIVED
    assert statuses["strcat"] is FindingStatus.EXPLOITED


async def test_run_council_sandbox_create_failure_falls_back_to_host(
    settings: Settings,
    db: Database,
    tools: CouncilTools,
    tmp_path: Path,
    logger: EventLogger,
) -> None:
    await _seed_agents(db)
    topic = await db.create_topic("audit")
    message_board = _message_board(settings, db, logger)
    candidates = json.dumps(
        [
            {"file": "sample.c", "line": 6, "sink": "strcpy", "call_path": ["run"]}
        ]
    )
    llm = ScriptedLLM(
        [
            _response(candidates),
            _response("[]"),
            _response(_verdict("survives", "reachable")),
            _response(_verdict("survives", "path holds")),
            _response("print('poc')"),
        ]
    )
    sandbox = _TaskSandbox(
        [], settings.workspace_dir, create_error=RuntimeError("no docker")
    )

    result = await run_council(
        llm,
        tools,
        db,
        message_board,
        topic.id,
        str(_poc_checkout(tmp_path)),
        "finder",
        "killer",
        "judge",
        "exploit",
        containers=sandbox,
    )

    assert result.exploited == 0
    assert result.pocs_written == 1
    assert result.pocs_run == 0
    assert any("sandbox unavailable" in note for note in result.notes)
    assert sandbox.removed == []
    poc = settings.workspace_dir / "exploit" / f"poc_{result.findings[0].id}.py"
    assert poc.exists()


async def test_run_council_host_path_without_containers(
    settings: Settings,
    db: Database,
    tools: CouncilTools,
    tmp_path: Path,
    logger: EventLogger,
) -> None:
    await _seed_agents(db)
    topic = await db.create_topic("audit")
    message_board = _message_board(settings, db, logger)
    candidates = json.dumps(
        [
            {"file": "sample.c", "line": 6, "sink": "strcpy", "call_path": ["run"]}
        ]
    )
    llm = ScriptedLLM(
        [
            _response(candidates),
            _response("[]"),
            _response(_verdict("survives", "reachable")),
            _response(_verdict("survives", "path holds")),
            _response("print('poc')"),
        ]
    )

    result = await run_council(
        llm,
        tools,
        db,
        message_board,
        topic.id,
        str(_poc_checkout(tmp_path)),
        "finder",
        "killer",
        "judge",
        "exploit",
        containers=None,
    )

    assert result.exploited == 0
    assert result.pocs_written == 1
    assert result.pocs_run == 0
    assert result.pocs_succeeded == 0
    poc = settings.workspace_dir / "exploit" / f"poc_{result.findings[0].id}.py"
    assert poc.exists()
