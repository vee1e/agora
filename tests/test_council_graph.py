from __future__ import annotations

import time
from pathlib import Path

import pytest

from harness.council.graph import Alloc, BufferSize, CodeGraph, Function, Sink, TaintSource

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
    char small[8];
    dispatch(argv[1]);
    gets(small);
    printf(argv[1]);
    free(heap);
    return 0;
}
"""

PY_SOURCE = """import os
import sys


def copy_value(value):
    return value


def handle():
    raw = os.environ["TOKEN"]
    alias = raw
    copy_value(alias)
    eval(alias)


def safe():
    name = "literal"
    copy_value(name)
"""


def _write_tree(root: Path) -> None:
    (root / "sample.c").write_text(C_SOURCE, encoding="utf-8")
    (root / "module.py").write_text(PY_SOURCE, encoding="utf-8")
    nested = root / "deep"
    nested.mkdir()
    (nested / "helper.c").write_text(
        "void outer_wrapper(char *x){ copy_payload(x); }\n", encoding="utf-8"
    )
    hidden = root / ".hidden"
    hidden.mkdir()
    (hidden / "skip.c").write_text("void hidden(void){ }\n", encoding="utf-8")


@pytest.fixture
def graph(tmp_path: Path) -> CodeGraph:
    _write_tree(tmp_path)
    g = CodeGraph(tmp_path)
    g.build()
    return g


def test_files_collected_skips_hidden(graph: CodeGraph) -> None:
    files = graph.files
    assert "sample.c" in files
    assert "module.py" in files
    assert "deep/helper.c" in files
    assert all(".hidden" not in f for f in files)


def test_functions_map(graph: CodeGraph) -> None:
    funcs = graph.functions
    assert "copy_payload" in funcs
    assert "handle" in funcs
    assert isinstance(funcs["copy_payload"][0], Function)


def test_function_metadata(graph: CodeGraph) -> None:
    fn = graph.functions["copy_payload"][0]
    assert fn.file == "sample.c"
    assert fn.params == ["input"]
    assert fn.line == 4
    assert fn.end_line == 7


def test_callers_of_returns_pairs(graph: CodeGraph) -> None:
    callers = graph.callers_of("copy_payload")
    names = sorted(caller.name for caller, _line in callers)
    assert names == ["dispatch", "outer_wrapper"]
    for caller, line in callers:
        assert isinstance(caller, Function)
        assert isinstance(line, int)


def test_callers_of_empty(graph: CodeGraph) -> None:
    assert graph.callers_of("does_not_exist") == []


def test_callers_of_returns_call_line(graph: CodeGraph) -> None:
    lines = {caller.name: line for caller, line in graph.callers_of("copy_payload")}
    assert lines == {"dispatch": 10, "outer_wrapper": 1}


def test_call_path_across_two_functions(graph: CodeGraph) -> None:
    paths = graph.call_path("run", "copy_payload")
    assert ["run", "dispatch", "copy_payload"] in paths


def test_call_path_no_match_returns_empty(graph: CodeGraph) -> None:
    assert graph.call_path("copy_payload", "run") == []


def test_call_path_reaches_libc_sink_two_hops(graph: CodeGraph) -> None:
    paths = graph.call_path("run", "strcpy")
    assert ["run", "dispatch", "copy_payload", "strcpy"] in paths


def test_call_path_direct_libc_sink(graph: CodeGraph) -> None:
    paths = graph.call_path("copy_payload", "strcpy")
    assert paths == [["copy_payload", "strcpy"]]


def test_call_path_respects_max_depth(graph: CodeGraph) -> None:
    assert graph.call_path("run", "copy_payload", max_depth=1) == []


def test_call_path_respects_max_paths(graph: CodeGraph) -> None:
    paths = graph.call_path("run", "copy_payload", max_paths=1)
    assert len(paths) <= 1


def test_call_path_terminates_on_cycle(tmp_path: Path) -> None:
    (tmp_path / "cycle.c").write_text(
        """
void alpha(void);
void omega(void);
void alpha(void){ omega(); }
void omega(void){ alpha(); }
""",
        encoding="utf-8",
    )
    g = CodeGraph(tmp_path)
    g.build()
    paths = g.call_path("alpha", "omega")
    assert paths == [["alpha", "omega"]]


def test_duplicate_function_names(tmp_path: Path) -> None:
    (tmp_path / "one.c").write_text(
        "void dup(char *a){ char b[4]; strcpy(b, a); }\n", encoding="utf-8"
    )
    (tmp_path / "two.c").write_text("void dup(char *a){ char c[8]; }\n", encoding="utf-8")
    g = CodeGraph(tmp_path)
    g.build()
    entries = g.functions["dup"]
    assert len(entries) == 2
    files = sorted(fn.file for fn in entries)
    assert files == ["one.c", "two.c"]


def test_nested_python_function(tmp_path: Path) -> None:
    (tmp_path / "nest.py").write_text(
        "def outer():\n    def inner(x):\n        return x\n    return inner\n",
        encoding="utf-8",
    )
    g = CodeGraph(tmp_path)
    g.build()
    assert "inner" in g.functions
    assert "outer" in g.functions


def test_alloc_of(graph: CodeGraph) -> None:
    allocs = graph.alloc_of("run")
    assert any(isinstance(a, Alloc) and a.name == "heap" for a in allocs)
    assert all(a.size_expr for a in allocs)
    target = graph.alloc_of("run", "heap")
    assert len(target) == 1


def test_size_expr(graph: CodeGraph) -> None:
    buffers = graph.size_expr("copy_payload", "buf")
    assert len(buffers) == 1
    assert isinstance(buffers[0], BufferSize)
    assert buffers[0].size_expr == "32"


def test_size_expr_missing(graph: CodeGraph) -> None:
    assert graph.size_expr("copy_payload", "nope") == []


def test_size_expr_includes_heap_allocation(tmp_path: Path) -> None:
    (tmp_path / "heap.c").write_text(
        "void make(void){ char *dst = malloc(32); }\n", encoding="utf-8"
    )
    g = CodeGraph(tmp_path)
    g.build()
    sizes = g.size_expr("make", "dst")
    assert len(sizes) == 1
    assert sizes[0].size_expr == "32"
    assert sizes[0].kind == "heap"


def test_tainted_true_for_read_destination(tmp_path: Path) -> None:
    (tmp_path / "rd.c").write_text(
        "void reader(int fd){ char buf[32]; read(fd, buf, 32); }\n", encoding="utf-8"
    )
    g = CodeGraph(tmp_path)
    g.build()
    assert g.tainted("reader", "buf") is True


def test_tainted_true_for_strcpy_destination(tmp_path: Path) -> None:
    (tmp_path / "cp.c").write_text(
        'void copier(char *src){ char dst[32]; strcpy(dst, src); }\n', encoding="utf-8"
    )
    g = CodeGraph(tmp_path)
    g.build()
    assert g.tainted("copier", "dst") is True


def test_sources_and_sinks_recorded(graph: CodeGraph) -> None:
    fn = graph.functions["run"][0]
    kinds = {source.kind for source in fn.sources}
    assert "argv" in kinds or any("argv" in s.name for s in fn.sources)
    sink_names = {sink.name for sink in fn.sinks}
    assert "gets" in sink_names or "printf" in sink_names
    assert all(isinstance(s, Sink) for s in fn.sinks)
    assert all(isinstance(s, TaintSource) for s in fn.sources)


def test_tainted_true_for_derived_var(graph: CodeGraph) -> None:
    assert graph.tainted("handle", "alias") is True


def test_tainted_false_for_literal(graph: CodeGraph) -> None:
    assert graph.tainted("safe", "name") is False


def test_tainted_unknown_function(graph: CodeGraph) -> None:
    assert graph.tainted("missing_function", "anything") is False


def test_to_dict_json_serializable(graph: CodeGraph) -> None:
    import json

    payload = graph.to_dict()
    encoded = json.dumps(payload)
    assert "functions" in payload
    assert "copy_payload" in payload["functions"]
    assert encoded


def test_missing_root_is_empty(tmp_path: Path) -> None:
    g = CodeGraph(tmp_path / "absent")
    g.build()
    assert g.files == []
    assert g.functions == {}


def test_oversized_file_skipped(tmp_path: Path) -> None:
    big = tmp_path / "big.c"
    big.write_text("void f(void){}\n" + "/* pad */\n" * 400000, encoding="utf-8")
    g = CodeGraph(tmp_path)
    g.build()
    assert "big.c" not in g.files
    assert "big.c" in g.skipped


def test_max_files_cap(tmp_path: Path) -> None:
    for index in range(5):
        (tmp_path / f"f{index}.c").write_text("void f(void){}\n", encoding="utf-8")
    g = CodeGraph(tmp_path, max_files=2)
    g.build()
    assert len(g.files) <= 2


def test_symlinked_file_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "secret.c"
    secret.write_text(
        'void leaked_secret(void){ char *p = getenv("X"); }\n', encoding="utf-8"
    )
    (root / "real.c").write_text("void real(void){}\n", encoding="utf-8")
    link = root / "link.c"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("symlinks are not supported on this platform")
    g = CodeGraph(root)
    g.build()
    assert "real.c" in g.files
    assert "link.c" not in g.files
    assert "leaked_secret" not in g.functions


def test_call_path_prompt_on_high_fanout_unreachable_sink(tmp_path: Path) -> None:
    levels = 9
    width = 8
    lines = ["void goal(void){}"]
    for level in range(levels):
        for index in range(width):
            calls = ""
            if level < levels - 1:
                calls = "".join(f"f{level + 1}_{k}();" for k in range(width))
            lines.append(f"void f{level}_{index}(void){{ {calls} }}")
    (tmp_path / "fan.c").write_text("\n".join(lines), encoding="utf-8")
    g = CodeGraph(tmp_path)
    g.build()
    started = time.perf_counter()
    paths = g.call_path("f0_0", "goal")
    elapsed = time.perf_counter() - started
    assert paths == []
    assert elapsed < 2.0


def test_deep_c_declarator_does_not_crash_build(tmp_path: Path) -> None:
    (tmp_path / "deep.c").write_bytes(b"int " + b"*" * 5000 + b" f(void){ return 0; }\n")
    g = CodeGraph(tmp_path)
    g.build()
    assert "deep.c" in g.files
    assert "f" in g.functions


def test_deep_python_def_does_not_crash_build(tmp_path: Path) -> None:
    lines = ["    " * index + f"def f{index}():" for index in range(900)]
    lines.append("    " * 900 + "pass")
    (tmp_path / "deep.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    g = CodeGraph(tmp_path)
    g.build()
    assert isinstance(g.functions, dict)


def test_recursive_expression_file_is_skipped(tmp_path: Path) -> None:
    expression = "a." * 3000 + "b()"
    (tmp_path / "attr.py").write_text(f"def f():\n    {expression}\n", encoding="utf-8")
    g = CodeGraph(tmp_path)
    g.build()
    assert "attr.py" not in g.files
    assert "f" not in g.functions


def test_tainted_true_for_c_assignment_expression(tmp_path: Path) -> None:
    (tmp_path / "assign.c").write_text(
        'void src(void){ char *p; p = getenv("HOME"); }\n', encoding="utf-8"
    )
    g = CodeGraph(tmp_path)
    g.build()
    assert g.tainted("src", "p") is True


def test_tainted_true_for_c_field_assignment(tmp_path: Path) -> None:
    (tmp_path / "field.c").write_text(
        'void to_field(void){ char *t = getenv("X"); s->f = t; }\n', encoding="utf-8"
    )
    g = CodeGraph(tmp_path)
    g.build()
    assert g.tainted("to_field", "s->f") is True


def test_tainted_true_for_python_attribute_assignment(tmp_path: Path) -> None:
    (tmp_path / "attr.py").write_text(
        "import os\n\n\ndef handle():\n"
        '    raw = os.environ["TOKEN"]\n'
        "    obj.attr = raw\n",
        encoding="utf-8",
    )
    g = CodeGraph(tmp_path)
    g.build()
    assert g.tainted("handle", "obj.attr") is True


def test_tainted_false_for_local_named_after_source(tmp_path: Path) -> None:
    (tmp_path / "local.py").write_text(
        "def local_input():\n"
        "    input = 5\n"
        "    return input\n\n"
        "def local_environ():\n"
        '    environ = "literal"\n'
        "    return environ\n",
        encoding="utf-8",
    )
    g = CodeGraph(tmp_path)
    g.build()
    assert g.tainted("local_input", "input") is False
    assert g.tainted("local_environ", "environ") is False


def test_decorated_and_nested_python_defs_collected(tmp_path: Path) -> None:
    source = (
        '@route("/x")\n'
        "def decorated(handler):\n"
        "    return handler\n\n"
        "def outer():\n"
        "    if flag:\n"
        "        def nested(x):\n"
        "            return x\n"
        "    try:\n"
        "        def in_try(y):\n"
        "            return y\n"
        "    except Exception:\n"
        "        pass\n"
        '    with open("f") as fh:\n'
        "        def in_with(z):\n"
        "            return z\n"
        "    return nested\n"
    )
    (tmp_path / "routes.py").write_text(source, encoding="utf-8")
    g = CodeGraph(tmp_path)
    g.build()
    for name in ("decorated", "nested", "in_try", "in_with", "outer"):
        assert name in g.functions


def test_comment_arguments_do_not_shift_sink_indices(tmp_path: Path) -> None:
    (tmp_path / "sinks.c").write_text(
        "void with_comment(void){ char buf[32]; snprintf(buf, /* size */ n, \"%s\", raw); }\n"
        "void without_comment(void){ char buf[32]; snprintf(buf, n, \"%s\", raw); }\n"
        "void alloc_comment(void){ char *p = malloc(/* size */ 128); }\n",
        encoding="utf-8",
    )
    g = CodeGraph(tmp_path)
    g.build()
    with_sink = g.functions["with_comment"][0].sinks[0]
    without_sink = g.functions["without_comment"][0].sinks[0]
    assert with_sink.arg == "n"
    assert without_sink.arg == "n"
    assert g.functions["alloc_comment"][0].allocs[0].size_expr == "128"
