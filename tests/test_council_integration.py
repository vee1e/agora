from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from harness import models as m
from harness.agent_runtime import tool_specs
from harness.budget import BudgetGuard
from harness.config import Settings
from harness.containers import ContainerManager
from harness.council.executor import CombinedTools, build_combined_tools
from harness.db import Database
from harness.message_board import MessageBoard
from harness.llm import LLMResponse
from harness.logging_ import EventLogger
from harness.mailbox import Mailbox
from harness.orchestrator import Orchestrator
from harness.state_machine import StateMachine

C_SOURCE = """#include <string.h>
#include <stdlib.h>

void copy_payload(char *input) {
    char buf[32];
    strcpy(buf, input);
}

void dispatch(char *arg) {
    copy_payload(arg);
}

int run(int argc, char **argv) {
    char *heap = malloc(128);
    dispatch(argv[1]);
    free(heap);
    return 0;
}
"""

EXPECTED_TOOLS = {
    "callers_of",
    "call_path",
    "alloc_of",
    "size_expr",
    "tainted",
    "patterns",
    "sg_search",
}


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
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "agent_id": agent_id,
                "max_tokens": max_tokens,
            }
        )
        if not self._responses:
            raise AssertionError("scripted LLM ran out of responses")
        return self._responses.pop(0)

    async def aclose(self) -> None:
        return None


def _response(
    text: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = "stop",
) -> LLMResponse:
    return LLMResponse(
        text=text,
        model="fake",
        usage=m.Usage(tokens_in=1, tokens_out=1),
        finish_reason=finish_reason,
        tool_calls=tool_calls or [],
    )


def _call(
    name: str, arguments: Any, call_id: str = "call_1"
) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _unwrap_untrusted_code(content: str) -> str:
    opening = "<untrusted_code>"
    closing = "</untrusted_code>"
    assert content.startswith(opening)
    assert content.endswith(closing)
    return content[len(opening) : -len(closing)]


def make_orch(
    settings: Settings,
    db: Database,
    logger: EventLogger,
    llm: Any,
    tools: CombinedTools | None = None,
) -> Orchestrator:
    mailbox = Mailbox(db, logger=logger)
    message_board = MessageBoard(db, mailbox, settings, logger=logger)
    return Orchestrator(
        settings,
        db,
        logger,
        ContainerManager(settings, db, logger=logger),
        mailbox,
        BudgetGuard(db, settings, logger=logger),
        StateMachine(db, logger=logger),
        llm,
        message_board,
        tools=tools,
    )


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "sample.c").write_text(C_SOURCE, encoding="utf-8")
    return root


@pytest.fixture
def combined(checkout: Path) -> CombinedTools:
    return build_combined_tools(checkout)


def test_specs_union_has_no_duplicates(combined: CombinedTools) -> None:
    specs = combined.specs()

    names = [spec["function"]["name"] for spec in specs]
    assert set(names) == EXPECTED_TOOLS
    assert len(names) == len(set(names))


def test_dispatch_routes_to_each_backend(combined: CombinedTools) -> None:
    callers = combined.dispatch("callers_of", {"fn": "copy_payload"})
    assert callers["ok"] is True
    assert callers["result"]["count"] == 1

    path = combined.dispatch("call_path", {"source": "run", "sink": "copy_payload"})
    assert path["ok"] is True
    assert ["run", "dispatch", "copy_payload"] in path["result"]["paths"]

    search = combined.dispatch("sg_search", {"pattern": "strcpy($$$ARGS)"})
    assert search["ok"] is True
    assert search["result"]


def test_dispatch_rejects_unknown_name(combined: CombinedTools) -> None:
    assert combined.dispatch("nope", {}) == {
        "ok": False,
        "error": "unknown tool: nope",
    }


def test_dispatch_reports_bad_arguments(combined: CombinedTools) -> None:
    result = combined.dispatch("callers_of", {})

    assert result["ok"] is False
    assert "fn" in result["error"]


def test_tool_specs_backward_compatible(combined: CombinedTools) -> None:
    assert tool_specs() == []
    assert tool_specs(combined) == combined.specs()


async def test_run_turn_dispatches_tool_and_feeds_result(
    settings: Settings, db: Database, logger: EventLogger, combined: CombinedTools
) -> None:
    call = _call("callers_of", json.dumps({"fn": "copy_payload"}))
    llm = ScriptedLLM(
        [
            _response(tool_calls=[call], finish_reason="tool_calls"),
            _response(text="done"),
        ]
    )
    orch = make_orch(settings, db, logger, llm, combined)
    agent = await orch.spawn_agent(m.AgentRole.FINDER)

    await orch.run_turn(agent.id)

    assert len(llm.calls) == 2
    assert llm.calls[0]["tools"] == combined.specs()
    second = llm.calls[1]["messages"]
    tool_messages = [msg for msg in second if msg.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    payload = json.loads(_unwrap_untrusted_code(tool_messages[0]["content"]))
    assert payload["ok"] is True
    assert payload["result"]["count"] == 1
    assistant = [msg for msg in second if msg.get("role") == "assistant"]
    assert assistant and assistant[0]["tool_calls"] == [call]

    turns = await db.list_turns(agent.id)
    assert turns[0].state is m.TurnState.DONE
    refreshed = await db.get_agent(agent.id)
    assert refreshed is not None and refreshed.state is m.AgentState.PAUSED

    tool_log = settings.log_dir / "tools.jsonl"
    assert tool_log.exists()
    assert "callers_of" in tool_log.read_text(encoding="utf-8")


async def test_run_turn_tool_error_does_not_end_turn(
    settings: Settings, db: Database, logger: EventLogger, combined: CombinedTools
) -> None:
    call = _call("does_not_exist", "{}")
    llm = ScriptedLLM(
        [
            _response(tool_calls=[call], finish_reason="tool_calls"),
            _response(text="recovered"),
        ]
    )
    orch = make_orch(settings, db, logger, llm, combined)
    agent = await orch.spawn_agent(m.AgentRole.FINDER)

    await orch.run_turn(agent.id)

    assert len(llm.calls) == 2
    tool_messages = [
        msg for msg in llm.calls[1]["messages"] if msg.get("role") == "tool"
    ]
    payload = json.loads(_unwrap_untrusted_code(tool_messages[0]["content"]))
    assert payload["ok"] is False

    turns = await db.list_turns(agent.id)
    assert turns[0].state is m.TurnState.DONE
    refreshed = await db.get_agent(agent.id)
    assert refreshed is not None and refreshed.state is m.AgentState.PAUSED
