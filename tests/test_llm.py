import json
from dataclasses import replace
from typing import Any
from unittest import mock

import httpx
import pytest

from harness.budget import BudgetGuard
from harness.config import Settings
from harness.db import Database
from harness.llm import LLMBudgetError, LLMClient, LLMError


def _ok_response(
    content: str = "hello",
    prompt_tokens: int = 3,
    completion_tokens: int = 5,
    model: str = "deepseek-v4.1-flash",
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
) -> httpx.Response:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish_reason}
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        },
    )


def _client_for(
    handler: Any,
    settings: Settings,
    logger: Any = None,
    budget: BudgetGuard | None = None,
    agent_id: str | None = None,
) -> tuple[LLMClient, httpx.AsyncClient]:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LLMClient(
        settings,
        logger=logger,
        budget=budget,
        agent_id=agent_id,
        client=http_client,
    )
    return client, http_client


async def test_complete_parses_response(settings: Settings) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return _ok_response()

    client, http = _client_for(handler, settings)
    try:
        resp = await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert resp.text == "hello"
    assert resp.model == "deepseek-v4.1-flash"
    assert resp.usage.tokens_in == 3
    assert resp.usage.tokens_out == 5
    assert resp.usage.total == 8
    assert resp.finish_reason == "stop"
    assert seen["url"].endswith("/chat/completions")
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "deepseek-v4.1-flash"
    assert seen["body"]["max_tokens"] == settings.turn_token_budget
    assert "temperature" in seen["body"]


async def test_passes_tools_and_response_format(settings: Settings) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return _ok_response()

    tools = [{"type": "function", "function": {"name": "f"}}]
    fmt = {"type": "json_object"}
    client, http = _client_for(handler, settings)
    try:
        await client.complete(
            [{"role": "user", "content": "hi"}],
            tools=tools,
            response_format=fmt,
        )
    finally:
        await http.aclose()

    assert seen["body"]["tools"] == tools
    assert seen["body"]["response_format"] == fmt


async def test_max_tokens_and_temperature_overrides(settings: Settings) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return _ok_response()

    client, http = _client_for(handler, settings)
    try:
        await client.complete(
            [{"role": "user", "content": "hi"}], max_tokens=42, temperature=0.1
        )
    finally:
        await http.aclose()

    assert seen["body"]["max_tokens"] == 42
    assert seen["body"]["temperature"] == 0.1


async def test_agent_id_argument_overrides_default(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response()

    db = Database(settings.db_path)
    await db.connect()
    await db.init_schema()
    guard = BudgetGuard(db, settings)
    client, http = _client_for(handler, settings, budget=guard, agent_id="default")
    try:
        await client.complete([{"role": "user", "content": "hi"}], agent_id="override")
    finally:
        await http.aclose()
        await db.close()

    assert await await_db_value(settings, "agent", "override") == 8
    assert await await_db_value(settings, "agent", "default") == 0


async def await_db_value(settings: Settings, scope: str, key: str) -> int:
    db = Database(settings.db_path)
    await db.connect()
    try:
        row = await db.get_budget(scope, key, BudgetGuard.today())
        return row.tokens_used if row is not None else 0
    finally:
        await db.close()


async def test_budget_recorded_after_call(
    db: Database, settings: Settings
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response(prompt_tokens=10, completion_tokens=15)

    guard = BudgetGuard(db, settings)
    client, http = _client_for(handler, settings, budget=guard, agent_id="ag_1")
    try:
        await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert await guard.agent_used("ag_1") == 25
    assert await guard.global_used() == 25


async def test_budget_exceeded_raises_before_request(
    db: Database, settings: Settings
) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return _ok_response()

    guard = BudgetGuard(db, settings)
    await guard.record("ag_1", settings.daily_agent_token_cap)
    client, http = _client_for(handler, settings, budget=guard, agent_id="ag_1")
    try:
        with pytest.raises(LLMBudgetError):
            await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert called is False


async def test_retries_once_on_429_then_succeeds(settings: Settings) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, text="slow down")
        return _ok_response(content="ok")

    client, http = _client_for(handler, settings)
    try:
        resp = await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert attempts == 2
    assert resp.text == "ok"


async def test_retries_on_500_then_fails(settings: Settings) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, text="server exploded")

    client, http = _client_for(handler, settings)
    try:
        with pytest.raises(LLMError) as excinfo:
            await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert attempts == 3
    assert "500" in str(excinfo.value)
    assert "server exploded" in str(excinfo.value)


async def test_non_retryable_error_raises_immediately(settings: Settings) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, text="bad request")

    client, http = _client_for(handler, settings)
    try:
        with pytest.raises(LLMError) as excinfo:
            await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert attempts == 1
    assert "400" in str(excinfo.value)
    assert "bad request" in str(excinfo.value)


async def test_invalid_json_raises_llm_error(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    client, http = _client_for(handler, settings)
    try:
        with pytest.raises(LLMError):
            await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()


async def test_missing_choices_raises_llm_error(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"usage": {}})

    client, http = _client_for(handler, settings)
    try:
        with pytest.raises(LLMError):
            await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()


async def test_tool_calls_parsed(settings: Settings) -> None:
    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response(content="", tool_calls=tool_calls)

    client, http = _client_for(handler, settings)
    try:
        resp = await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert resp.tool_calls == tool_calls
    assert resp.text == ""


async def test_null_content_becomes_empty_string(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": None},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            },
        )

    client, http = _client_for(handler, settings)
    try:
        resp = await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert resp.text == ""
    assert resp.finish_reason == "length"


async def test_successful_call_is_logged(
    settings: Settings, logger
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response(prompt_tokens=4, completion_tokens=6)

    client, http = _client_for(handler, settings, logger=logger)
    try:
        await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    records = [
        json.loads(line)
        for line in (settings.log_dir / "llm.jsonl").read_text().splitlines()
    ]
    assert records[-1]["tokens_in"] == 4
    assert records[-1]["tokens_out"] == 6
    assert records[-1]["error"] is None


async def test_error_is_logged(settings: Settings, logger) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad")

    client, http = _client_for(handler, settings, logger=logger)
    try:
        with pytest.raises(LLMError):
            await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    records = [
        json.loads(line)
        for line in (settings.log_dir / "llm.jsonl").read_text().splitlines()
    ]
    assert records[-1]["error"] is not None
    assert "400" in records[-1]["error"]


async def test_lazy_client_created_and_closed(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response()

    created: list[httpx.AsyncClient] = []
    real = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        instance = real(*args, **kwargs)
        created.append(instance)
        return instance

    with mock.patch("harness.llm.httpx.AsyncClient", side_effect=factory):
        client = LLMClient(settings)
        assert created == []
        await client.complete([{"role": "user", "content": "hi"}])
        assert len(created) == 1
        await client.aclose()
        assert created[0].is_closed


async def test_injected_client_is_not_closed_by_aclose(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response()

    client, http = _client_for(handler, settings)
    await client.aclose()
    assert not http.is_closed
    await http.aclose()


async def test_global_budget_exceeded_raises_before_request(
    db: Database, settings: Settings
) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return _ok_response()

    guard = BudgetGuard(db, settings)
    await guard.record("ag_other", settings.global_token_cap)
    client, http = _client_for(handler, settings, budget=guard, agent_id="ag_1")
    try:
        with pytest.raises(LLMBudgetError):
            await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert called is False


async def test_response_body_over_cap_raises(settings: Settings) -> None:
    small = replace(settings, max_response_bytes=64)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 500)

    client, http = _client_for(handler, small)
    try:
        with pytest.raises(LLMError) as excinfo:
            await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert "max_response_bytes" in str(excinfo.value)


async def test_error_body_over_cap_raises(settings: Settings) -> None:
    small = replace(settings, max_response_bytes=16)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="y" * 500)

    client, http = _client_for(handler, small)
    try:
        with pytest.raises(LLMError) as excinfo:
            await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert "max_response_bytes" in str(excinfo.value)


async def test_missing_usage_logs_warning(settings: Settings, logger) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client, http = _client_for(handler, settings, logger=logger)
    try:
        resp = await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert resp.usage.total == 0
    records = [
        json.loads(line)
        for line in (settings.log_dir / "llm.jsonl").read_text().splitlines()
    ]
    assert "usage missing" in (records[-1]["note"] or "")


async def test_logs_model_returned_by_provider(settings: Settings, logger) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response(model="provider-model")

    client, http = _client_for(handler, settings, logger=logger)
    try:
        resp = await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert resp.model == "provider-model"
    records = [
        json.loads(line)
        for line in (settings.log_dir / "llm.jsonl").read_text().splitlines()
    ]
    assert records[-1]["model"] == "provider-model"


async def test_invalid_json_error_includes_body(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json at all")

    client, http = _client_for(handler, settings)
    try:
        with pytest.raises(LLMError) as excinfo:
            await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert "not json at all" in str(excinfo.value)


async def test_tool_calls_dict_is_not_coerced(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "hi",
                            "tool_calls": {"id": "call_1"},
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client, http = _client_for(handler, settings)
    try:
        resp = await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert resp.tool_calls == []


async def test_retry_after_header_is_honored(settings: Settings) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, text="slow", headers={"Retry-After": "0"})
        return _ok_response(content="ok")

    client, http = _client_for(handler, settings)
    try:
        resp = await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert attempts == 2
    assert resp.text == "ok"


async def test_501_is_not_retried(settings: Settings) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(501, text="not implemented")

    client, http = _client_for(handler, settings)
    try:
        with pytest.raises(LLMError):
            await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert attempts == 1


async def test_retry_backoff_sequence(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    client, http = _client_for(handler, settings)
    try:
        with mock.patch("harness.llm.asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(LLMError):
                await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await http.aclose()

    assert delays == [0.5, 1.0]
