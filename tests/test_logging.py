import json
import pathlib
from unittest import mock

from harness.logging_ import EventLogger


def _read_lines(path: pathlib.Path) -> list[dict]:
    assert path.exists(), f"{path} was not created"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_event_writes_events_jsonl(tmp_path: pathlib.Path) -> None:
    logger = EventLogger(tmp_path, agent_id="ag_1")
    logger.event("spawned", role="finder")
    records = _read_lines(tmp_path / "events.jsonl")
    assert len(records) == 1
    assert records[0]["kind"] == "spawned"
    assert records[0]["agent_id"] == "ag_1"
    assert records[0]["role"] == "finder"
    assert "ts" in records[0]


def test_llm_call_writes_llm_jsonl(tmp_path: pathlib.Path) -> None:
    logger = EventLogger(tmp_path)
    logger.llm_call(
        agent_id="ag_2",
        model="deepseek-v4.1-flash",
        tokens_in=10,
        tokens_out=20,
        latency_ms=5,
    )
    records = _read_lines(tmp_path / "llm.jsonl")
    assert records[0]["agent_id"] == "ag_2"
    assert records[0]["model"] == "deepseek-v4.1-flash"
    assert records[0]["tokens_in"] == 10
    assert records[0]["tokens_out"] == 20
    assert records[0]["latency_ms"] == 5
    assert records[0]["error"] is None


def test_tool_call_writes_tools_jsonl(tmp_path: pathlib.Path) -> None:
    logger = EventLogger(tmp_path)
    logger.tool_call(agent_id="ag_3", name="read_file", args={"path": "a.py"}, result="ok")
    records = _read_lines(tmp_path / "tools.jsonl")
    assert records[0]["name"] == "read_file"
    assert records[0]["args"] == {"path": "a.py"}
    assert records[0]["result"] == "ok"


def test_default_agent_id_used_and_override(tmp_path: pathlib.Path) -> None:
    logger = EventLogger(tmp_path, agent_id="default")
    logger.llm_call(
        agent_id=None,
        model="m",
        tokens_in=0,
        tokens_out=0,
        latency_ms=0,
    )
    logger.llm_call(
        agent_id="override",
        model="m",
        tokens_in=0,
        tokens_out=0,
        latency_ms=0,
    )
    records = _read_lines(tmp_path / "llm.jsonl")
    assert records[0]["agent_id"] == "default"
    assert records[1]["agent_id"] == "override"


def test_log_dir_created_if_missing(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "nested" / "logs"
    logger = EventLogger(target)
    logger.event("x")
    assert (target / "events.jsonl").exists()


def test_unserializable_payload_is_stringified(tmp_path: pathlib.Path) -> None:
    logger = EventLogger(tmp_path)
    logger.tool_call(agent_id="a", name="t", args={"obj": object()})
    records = _read_lines(tmp_path / "tools.jsonl")
    assert isinstance(records[0]["args"]["obj"], str)


def test_never_raises_when_path_is_a_file(tmp_path: pathlib.Path) -> None:
    bad = tmp_path / "not_a_dir"
    bad.write_text("x")
    logger = EventLogger(bad)
    with mock.patch("sys.stderr") as stderr:
        logger.event("boom", value=1)
    assert bad.is_file()
    assert stderr.write.called


def test_repeated_calls_append(tmp_path: pathlib.Path) -> None:
    logger = EventLogger(tmp_path)
    for i in range(3):
        logger.event("tick", i=i)
    records = _read_lines(tmp_path / "events.jsonl")
    assert [r["i"] for r in records] == [0, 1, 2]


def test_close_is_noop(tmp_path: pathlib.Path) -> None:
    logger = EventLogger(tmp_path)
    assert logger.close() is None


def test_rotation_renames_file_over_max_bytes(tmp_path: pathlib.Path) -> None:
    logger = EventLogger(tmp_path, max_bytes=100)
    logger.event("first", value="x" * 200)
    logger.event("second")

    backup = tmp_path / "events.jsonl.1"
    assert backup.exists()
    assert _read_lines(backup)[0]["kind"] == "first"
    assert _read_lines(tmp_path / "events.jsonl")[-1]["kind"] == "second"


def test_rotation_replaces_existing_backup(tmp_path: pathlib.Path) -> None:
    logger = EventLogger(tmp_path, max_bytes=50)
    logger.event("a", value="x" * 100)
    logger.event("b", value="y" * 100)
    logger.event("c")

    backup = _read_lines(tmp_path / "events.jsonl.1")
    assert backup[0]["kind"] == "b"
    assert _read_lines(tmp_path / "events.jsonl")[-1]["kind"] == "c"


def test_failure_counter_reported_loudly(tmp_path: pathlib.Path) -> None:
    bad = tmp_path / "not_a_dir"
    bad.write_text("x")
    logger = EventLogger(bad)
    with mock.patch("sys.stderr") as stderr:
        for _ in range(3):
            logger.event("boom")
    assert logger._consecutive_failures == 3
    written = " ".join(str(call.args[0]) for call in stderr.write.call_args_list)
    assert "EventLogger failed to write" in written
    assert "EventLogger ERROR" in written
    assert "1 consecutive" in written


def test_failure_counter_reports_every_hundredth(tmp_path: pathlib.Path) -> None:
    bad = tmp_path / "not_a_dir"
    bad.write_text("x")
    logger = EventLogger(bad)
    logger._consecutive_failures = 99
    with mock.patch("sys.stderr") as stderr:
        logger.event("boom")
    written = " ".join(str(call.args[0]) for call in stderr.write.call_args_list)
    assert "EventLogger ERROR" in written
    assert "100 consecutive" in written


def test_failure_counter_resets_on_success(tmp_path: pathlib.Path) -> None:
    logger = EventLogger(tmp_path)
    logger._consecutive_failures = 5
    logger.event("ok")
    assert logger._consecutive_failures == 0
