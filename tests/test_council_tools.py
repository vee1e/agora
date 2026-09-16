from __future__ import annotations

from pathlib import Path

import pytest

from harness.council.graph import CodeGraph
from harness.council.tools import CouncilTools

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
    printf(argv[1]);
    free(heap);
    return 0;
}
"""

PY_SOURCE = """import os


def handle():
    raw = os.environ["TOKEN"]
    alias = raw
    eval(alias)
"""


@pytest.fixture
def tools(tmp_path: Path) -> CouncilTools:
    (tmp_path / "sample.c").write_text(C_SOURCE, encoding="utf-8")
    (tmp_path / "module.py").write_text(PY_SOURCE, encoding="utf-8")
    graph = CodeGraph(tmp_path)
    graph.build()
    return CouncilTools(graph)


def test_specs_shape(tools: CouncilTools) -> None:
    specs = tools.specs()
    assert len(specs) == 6
    names = [spec["function"]["name"] for spec in specs]
    assert names == ["callers_of", "call_path", "alloc_of", "size_expr", "tainted", "patterns"]
    for spec in specs:
        assert spec["type"] == "function"
        function = spec["function"]
        assert function["description"]
        params = function["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["required"], list)
        if function["name"] != "patterns":
            assert params["required"]


def test_dispatch_callers_of_success(tools: CouncilTools) -> None:
    result = tools.dispatch("callers_of", {"fn": "copy_payload"})
    assert result["ok"] is True
    assert result["result"]["count"] == 1
    assert result["result"]["callers"][0]["name"] == "dispatch"


def test_dispatch_callers_of_error(tools: CouncilTools) -> None:
    result = tools.dispatch("callers_of", {})
    assert result["ok"] is False
    assert "fn" in result["error"]


def test_dispatch_call_path_success(tools: CouncilTools) -> None:
    result = tools.dispatch("call_path", {"source": "run", "sink": "copy_payload"})
    assert result["ok"] is True
    assert ["run", "dispatch", "copy_payload"] in result["result"]["paths"]


def test_dispatch_call_path_error(tools: CouncilTools) -> None:
    result = tools.dispatch("call_path", {"source": "run"})
    assert result["ok"] is False
    assert "sink" in result["error"]


def test_dispatch_alloc_of_success(tools: CouncilTools) -> None:
    result = tools.dispatch("alloc_of", {"ptr": "heap", "fn": "run"})
    assert result["ok"] is True
    assert result["result"]["count"] == 1
    assert result["result"]["allocs"][0]["name"] == "heap"


def test_dispatch_alloc_of_without_fn(tools: CouncilTools) -> None:
    result = tools.dispatch("alloc_of", {"ptr": "heap"})
    assert result["ok"] is True
    assert result["result"]["allocs"][0]["file"] == "sample.c"


def test_dispatch_alloc_of_error(tools: CouncilTools) -> None:
    result = tools.dispatch("alloc_of", {"ptr": ""})
    assert result["ok"] is False
    assert "ptr" in result["error"]


def test_dispatch_alloc_of_bad_fn_type(tools: CouncilTools) -> None:
    result = tools.dispatch("alloc_of", {"ptr": "heap", "fn": 5})
    assert result["ok"] is False
    assert "fn" in result["error"]


def test_dispatch_size_expr_success(tools: CouncilTools) -> None:
    result = tools.dispatch("size_expr", {"buf": "buf", "fn": "copy_payload"})
    assert result["ok"] is True
    assert result["result"]["buffers"][0]["size_expr"] == "32"


def test_dispatch_size_expr_error(tools: CouncilTools) -> None:
    result = tools.dispatch("size_expr", {})
    assert result["ok"] is False
    assert "buf" in result["error"]


def test_dispatch_size_expr_includes_heap(tmp_path: Path) -> None:
    (tmp_path / "heap.c").write_text(
        "void make(void){ char *dst = malloc(32); }\n", encoding="utf-8"
    )
    graph = CodeGraph(tmp_path)
    graph.build()
    tools = CouncilTools(graph)
    result = tools.dispatch("size_expr", {"buf": "dst", "fn": "make"})
    assert result["ok"] is True
    buffers = result["result"]["buffers"]
    assert len(buffers) == 1
    assert buffers[0]["size_expr"] == "32"
    assert buffers[0]["kind"] == "heap"
    global_result = tools.dispatch("size_expr", {"buf": "dst"})
    assert global_result["ok"] is True
    assert global_result["result"]["count"] == 1
    assert global_result["result"]["buffers"][0]["kind"] == "heap"
    assert "total" not in global_result["result"]


def test_dispatch_tainted_success(tools: CouncilTools) -> None:
    result = tools.dispatch("tainted", {"expr": "alias", "fn": "handle"})
    assert result["ok"] is True
    assert result["result"]["tainted"] is True


def test_dispatch_tainted_global_scan(tools: CouncilTools) -> None:
    result = tools.dispatch("tainted", {"expr": "alias"})
    assert result["ok"] is True
    assert result["result"]["tainted"] is True
    assert "handle" in result["result"]["functions"]


def test_dispatch_tainted_error(tools: CouncilTools) -> None:
    result = tools.dispatch("tainted", {"expr": 123})
    assert result["ok"] is False
    assert "expr" in result["error"]


def test_dispatch_unknown_tool(tools: CouncilTools) -> None:
    result = tools.dispatch("nope", {"fn": "x"})
    assert result["ok"] is False
    assert "unknown tool" in result["error"]


def test_dispatch_non_dict_arguments(tools: CouncilTools) -> None:
    result = tools.dispatch("callers_of", ["fn"])
    assert result["ok"] is False
    assert "object" in result["error"]


def test_dispatch_never_raises(tools: CouncilTools) -> None:
    for name in ("callers_of", "call_path", "alloc_of", "size_expr", "tainted"):
        result = tools.dispatch(name, {"bogus": True})
        assert result["ok"] is False


def test_dispatch_non_string_name(tools: CouncilTools) -> None:
    list_result = tools.dispatch(["callers_of"], {"fn": "x"})
    assert list_result["ok"] is False
    assert "unknown tool" in list_result["error"]
    dict_result = tools.dispatch({"name": "callers_of"}, {"fn": "x"})
    assert dict_result["ok"] is False


def test_oversized_tool_results_are_capped(tmp_path: Path) -> None:
    lines = [
        f"void alloc_{index}(void){{ char *buf_{index} = malloc(4); }}" for index in range(250)
    ]
    (tmp_path / "many.c").write_text("\n".join(lines), encoding="utf-8")
    graph = CodeGraph(tmp_path)
    graph.build()
    tools = CouncilTools(graph)
    result = tools.dispatch("alloc_of", {"ptr": "buf_"})
    assert result["ok"] is True
    payload = result["result"]
    assert payload["count"] == 250
    assert "total" not in payload
    assert payload["truncated"] is True
    assert len(payload["allocs"]) == 200
