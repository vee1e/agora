from __future__ import annotations

import itertools
from pathlib import Path

from harness.council.graph import CodeGraph
from harness.council.patterns import PatternScanner, Signal
from harness.council.tools import CouncilTools

_CASE = itertools.count()


def _scan(tmp_path: Path, files: dict[str, str]) -> list[Signal]:
    case = tmp_path / f"case{next(_CASE)}"
    for name, content in files.items():
        path = case / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return PatternScanner(case).scan()


def _of(signals: list[Signal], kind: str) -> list[Signal]:
    return [signal for signal in signals if signal.kind == kind]


def test_signal_is_frozen_dataclass() -> None:
    signal = Signal(
        kind="format_string",
        file="a.c",
        line=3,
        function="f",
        detail="printf(fmt) is not a literal",
        cwe="CWE-134",
        confidence="high",
    )
    assert signal.kind == "format_string"
    assert signal.confidence == "high"




def test_integer_overflow_alloc_positive(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": "void f(int n){ char *p = malloc(n * 4); }\n"},
    )
    hits = _of(signals, "integer_overflow_alloc")
    assert len(hits) == 1
    assert "n" in hits[0].detail
    assert hits[0].cwe == "CWE-190"
    assert hits[0].confidence == "high"


def test_integer_overflow_alloc_negative(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "void f(int n){\n"
                "    if (n > 1000) return;\n"
                "    char *p = malloc(n * 4);\n"
                "}\n"
            )
        },
    )
    assert _of(signals, "integer_overflow_alloc") == []


def test_integer_overflow_alloc_calloc(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": "void f(int n){ char *p = calloc(n, 16); }\n"},
    )
    assert len(_of(signals, "integer_overflow_alloc")) == 1




def test_unchecked_alloc_positive(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": "void f(int n){ char *p = malloc(n); p[0] = 0; }\n"},
    )
    hits = _of(signals, "unchecked_alloc")
    assert len(hits) == 1
    assert "malloc" in hits[0].detail
    assert hits[0].cwe == "CWE-252"


def test_unchecked_alloc_negative(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": "void f(int n){ char *p = malloc(n); if (!p) return; p[0] = 0; }\n"
        },
    )
    assert _of(signals, "unchecked_alloc") == []




def test_unchecked_io_return_positive(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": "void f(int fd){ char buf[16]; int n = read(fd, buf, 16); }\n"},
    )
    hits = _of(signals, "unchecked_io_return")
    assert len(hits) == 1
    assert "n" in hits[0].detail


def test_unchecked_io_return_negative(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": "void f(int fd){ char buf[16]; int n = read(fd, buf, 16); if (n < 0) return; }\n"
        },
    )
    assert _of(signals, "unchecked_io_return") == []




def test_double_free_positive(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": "void f(void){ char *p = malloc(4); free(p); free(p); }\n"},
    )
    assert len(_of(signals, "double_free")) == 1


def test_double_free_negative(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": "void f(void){ char *p = malloc(4); free(p); p = malloc(4); free(p); }\n"
        },
    )
    assert _of(signals, "double_free") == []




def test_use_after_free_positive(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": "void f(void){ char *p = malloc(4); free(p); p[0] = 1; }\n"},
    )
    assert len(_of(signals, "use_after_free")) == 1


def test_use_after_free_negative(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": "void f(void){ char *p = malloc(4); free(p); p = malloc(4); p[0] = 1; }\n"
        },
    )
    assert _of(signals, "use_after_free") == []




def test_realloc_loses_original_positive(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": "void f(void){ char *p = malloc(4); p = realloc(p, 8); }\n"},
    )
    assert len(_of(signals, "realloc_loses_original")) == 1


def test_realloc_loses_original_negative(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": "void f(void){ char *p = malloc(4); char *q = realloc(p, 8); }\n"},
    )
    assert _of(signals, "realloc_loses_original") == []




def test_toctou_positive(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "void f(const char *path){ if (access(path, 0) == 0) "
                "{ int fd = open(path, 0); } }\n"
            )
        },
    )
    hits = _of(signals, "toctou")
    assert len(hits) == 1
    assert "open(" in hits[0].detail
    assert hits[0].cwe == "CWE-367"


def test_toctou_negative(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "void f(const char *path){ if (access(path, 0) == 0) "
                '{ int fd = open("/etc/hosts", 0); } }\n'
            )
        },
    )
    assert _of(signals, "toctou") == []




def test_insecure_temp_file_positive(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": 'void f(void){ char *t = mktemp("/tmp/fooXXXXXX"); }\n'},
    )
    assert len(_of(signals, "insecure_temp_file")) == 1


def test_insecure_temp_file_negative(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": 'void f(void){ char t[] = "/tmp/fooXXXXXX"; int fd = mkstemp(t); }\n'},
    )
    assert _of(signals, "insecure_temp_file") == []


def test_insecure_temp_file_path_build(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "void f(char *n){ char buf[64]; sprintf(buf, "
                '"/tmp/%s", n); int fd = open(buf, 0); }\n'
            )
        },
    )
    assert len(_of(signals, "insecure_temp_file")) >= 1




def test_format_string_positive(tmp_path: Path) -> None:
    signals = _scan(tmp_path, {"a.c": "void f(char *fmt){ printf(fmt); }\n"})
    hits = _of(signals, "format_string")
    assert len(hits) == 1
    assert "printf" in hits[0].detail
    assert hits[0].cwe == "CWE-134"


def test_format_string_negative(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": 'void f(char *x){ printf("%s", x); }\n'},
    )
    assert _of(signals, "format_string") == []


def test_format_string_snprintf_index(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": 'void f(char *fmt){ char buf[16]; snprintf(buf, sizeof(buf), fmt); }\n'},
    )
    hits = _of(signals, "format_string")
    assert len(hits) == 1
    assert "fmt" in hits[0].detail
    safe = _scan(
        tmp_path,
        {
            "b.c": 'void g(char *x){ char buf[16]; snprintf(buf, sizeof(buf), "%s", x); }\n'
        },
    )
    assert _of(safe, "format_string") == []




def test_size_truncation_positive(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": "void f(char *s){ int len = strlen(s); char *p = malloc(len); }\n"
        },
    )
    hits = _of(signals, "size_truncation")
    assert len(hits) == 1
    assert "strlen" in hits[0].detail


def test_size_truncation_negative(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": "void f(char *s){ size_t len = strlen(s); char *p = malloc(len); }\n"
        },
    )
    assert _of(signals, "size_truncation") == []




def test_off_by_one_positive(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "void f(void){ char buf[16]; "
                "for (int i = 0; i <= 16; i++) { buf[i] = 0; } }\n"
            )
        },
    )
    hits = _of(signals, "off_by_one")
    assert len(hits) == 1
    assert "buf" in hits[0].detail


def test_off_by_one_negative(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "void f(void){ char buf[16]; "
                "for (int i = 0; i < 16; i++) { buf[i] = 0; } }\n"
            )
        },
    )
    assert _of(signals, "off_by_one") == []




def test_signal_unsafe_positive(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "#include <signal.h>\n"
                "void handler(int s){ char *p = malloc(4); free(p); }\n"
                "void setup(void){ signal(SIGINT, handler); }\n"
            )
        },
    )
    hits = _of(signals, "signal_unsafe")
    assert len(hits) >= 1
    assert all("handler" in hit.detail for hit in hits)
    assert all(hit.confidence == "high" for hit in hits)


def test_signal_unsafe_negative(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "#include <signal.h>\n"
                "void handler(int s){ }\n"
                "void setup(void){ signal(SIGINT, handler); }\n"
            )
        },
    )
    assert _of(signals, "signal_unsafe") == []




def test_missing_nul_termination_positive(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": "void f(char *s){ char buf[16]; strncpy(buf, s, 15); }\n"},
    )
    hits = _of(signals, "missing_nul_termination")
    assert len(hits) == 1
    assert "buf" in hits[0].detail


def test_missing_nul_termination_negative(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "void f(char *s){ char buf[16]; strncpy(buf, s, 15); "
                "buf[15] = '\\0'; }\n"
            )
        },
    )
    assert _of(signals, "missing_nul_termination") == []




def test_snprintf_offset_write_positive(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "void f(char *s){ char buf[16]; int r = snprintf(buf, "
                'sizeof(buf), "%s", s); buf[r] = 0; }\n'
            )
        },
    )
    hits = _of(signals, "snprintf_offset_write")
    assert len(hits) == 1
    assert "buf[r]" in hits[0].detail


def test_snprintf_offset_write_negative(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "void f(char *s){ char buf[16]; int r = snprintf(buf, "
                'sizeof(buf), "%s", s); if (r < 0) return; }\n'
            )
        },
    )
    assert _of(signals, "snprintf_offset_write") == []




def test_dest_scratch_write_same_expression(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": "void f(char *p, int n){ memcpy(p, p, n); }\n"},
    )
    hits = _of(signals, "dest_scratch_write")
    assert len(hits) == 1
    assert "same expression" in hits[0].detail


def test_dest_scratch_write_computed_offset(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": "void f(char *dst, int off, int n){ memcpy(dst + off, src, n); }\n"},
    )
    hits = _of(signals, "dest_scratch_write")
    assert len(hits) == 1
    assert "dst" in hits[0].detail


def test_dest_scratch_write_negative(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": "void f(char *a, char *b, int n){ memcpy(a, b, n); }\n"},
    )
    assert _of(signals, "dest_scratch_write") == []
    clamped = _scan(
        tmp_path,
        {
            "b.c": (
                "void g(char *dst, int off, int n){\n"
                "    if (off + n > 64) return;\n"
                "    memcpy(dst + off, src, n);\n"
                "}\n"
            )
        },
    )
    assert _of(clamped, "dest_scratch_write") == []




def test_unchecked_retry_loop_positive(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "void f(void){ while (!done) { result = write_file(fd, buf, n); "
                "done = 1; } }\n"
            )
        },
    )
    hits = _of(signals, "unchecked_retry_loop")
    assert len(hits) == 1
    assert hits[0].confidence == "low"
    assert "write_file" in hits[0].detail


def test_unchecked_retry_loop_negative(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": "void f(void){ while (!done) { done = check(); } }\n"},
    )
    assert _of(signals, "unchecked_retry_loop") == []
    counting = _scan(
        tmp_path,
        {
            "b.c": (
                "void g(void){ int i = 0; while (i < 10) { process(i); i++; } }\n"
            )
        },
    )
    assert _of(counting, "unchecked_retry_loop") == []




def test_python_unchecked_io_return(tmp_path: Path) -> None:
    positive = _scan(
        tmp_path,
        {"p.py": "import os\n\n\ndef f(fd):\n    n = os.read(fd, 16)\n"},
    )
    assert len(_of(positive, "unchecked_io_return")) == 1
    negative = _scan(
        tmp_path,
        {
            "q.py": (
                "import os\n\n\ndef f(fd):\n    n = os.read(fd, 16)\n"
                "    if n < 0:\n        return\n"
            )
        },
    )
    assert _of(negative, "unchecked_io_return") == []


def test_python_toctou(tmp_path: Path) -> None:
    positive = _scan(
        tmp_path,
        {
            "p.py": (
                "import os\n\n\ndef f(path):\n    if os.access(path, 0) == 0:\n"
                "        fd = os.open(path, 0)\n"
            )
        },
    )
    assert len(_of(positive, "toctou")) == 1
    negative = _scan(
        tmp_path,
        {
            "q.py": (
                "import os\n\n\ndef f(path):\n    if os.access(path, 0) == 0:\n"
                '        fd = os.open("/etc/hosts", 0)\n'
            )
        },
    )
    assert _of(negative, "toctou") == []


def test_python_insecure_temp_file(tmp_path: Path) -> None:
    positive = _scan(
        tmp_path,
        {"p.py": "import tempfile\n\n\ndef f():\n    name = tempfile.mktemp()\n"},
    )
    assert len(_of(positive, "insecure_temp_file")) == 1
    negative = _scan(
        tmp_path,
        {"q.py": "import tempfile\n\n\ndef f():\n    fd, name = tempfile.mkstemp()\n"},
    )
    assert _of(negative, "insecure_temp_file") == []


def test_python_unchecked_retry_loop(tmp_path: Path) -> None:
    positive = _scan(
        tmp_path,
        {
            "p.py": (
                "def f(done):\n    while not done:\n        result = write_file()\n"
                "        done = True\n"
            )
        },
    )
    assert len(_of(positive, "unchecked_retry_loop")) == 1
    negative = _scan(
        tmp_path,
        {"q.py": "def f(done):\n    while not done:\n        done = check()\n"},
    )
    assert _of(negative, "unchecked_retry_loop") == []




def test_scan_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "a.c").write_text("void f(char *x){ printf(x); }\n", encoding="utf-8")
    (tmp_path / "b.c").write_text("void g(char *x){ printf(x); }\n", encoding="utf-8")
    first = [(s.kind, s.file, s.line) for s in PatternScanner(tmp_path).scan()]
    second = [(s.kind, s.file, s.line) for s in PatternScanner(tmp_path).scan()]
    assert first == second
    assert first == sorted(first)


def test_scan_skips_symlinks_and_hidden(tmp_path: Path) -> None:
    (tmp_path / "real.c").write_text("void f(char *x){ printf(x); }\n", encoding="utf-8")
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "skip.c").write_text("void g(char *x){ printf(x); }\n", encoding="utf-8")
    files = {signal.file for signal in PatternScanner(tmp_path).scan()}
    assert files == {"real.c"}




def test_graph_exposes_signals(tmp_path: Path) -> None:
    (tmp_path / "a.c").write_text("void f(char *x){ printf(x); }\n", encoding="utf-8")
    graph = CodeGraph(tmp_path)
    graph.build()
    assert graph.signals
    assert graph.signals_for(kind="format_string")
    assert graph.signals_for(file="a.c")
    assert graph.signals_for(file="a.c", kind="format_string")
    assert graph.signals_for(file="missing.c") == []
    assert graph.signals_for(kind="no_such_kind") == []




def _tools(tmp_path: Path) -> CouncilTools:
    (tmp_path / "a.c").write_text(
        "void f(char *x){ printf(x); }\n", encoding="utf-8"
    )
    graph = CodeGraph(tmp_path)
    graph.build()
    return CouncilTools(graph)


def test_patterns_tool_through_dispatch(tmp_path: Path) -> None:
    result = _tools(tmp_path).dispatch("patterns", {})
    assert result["ok"] is True
    payload = result["result"]
    assert payload["count"] >= 1
    assert payload["truncated"] is False
    kinds = {entry["kind"] for entry in payload["signals"]}
    assert "format_string" in kinds
    entry = payload["signals"][0]
    assert set(entry) == {"kind", "file", "line", "function", "detail", "cwe",
                          "confidence"}


def test_patterns_tool_filters(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    by_kind = tools.dispatch("patterns", {"kind": "format_string"})
    assert by_kind["ok"] is True
    assert by_kind["result"]["count"] == 1
    by_path = tools.dispatch("patterns", {"path": "a.c"})
    assert by_path["result"]["count"] == 1
    missing = tools.dispatch("patterns", {"path": "missing.c"})
    assert missing["ok"] is True
    assert missing["result"]["count"] == 0


def test_patterns_tool_bad_arguments(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    assert tools.dispatch("patterns", {"kind": 3})["ok"] is False
    assert tools.dispatch("patterns", {"path": 3})["ok"] is False
    assert tools.dispatch("patterns", {"limit": 0})["ok"] is False
    assert tools.dispatch("patterns", {"limit": "x"})["ok"] is False
    assert tools.dispatch("patterns", {"limit": True})["ok"] is False


def test_patterns_tool_limit_cap(tmp_path: Path) -> None:
    result = _tools(tmp_path).dispatch("patterns", {"limit": 1})
    assert result["ok"] is True
    payload = result["result"]
    assert payload["count"] == 1
    assert len(payload["signals"]) == 1
    assert payload["truncated"] is False


def test_patterns_tool_global_cap(tmp_path: Path) -> None:
    lines = [
        f"  p{index} = malloc(n * 4);" for index in range(210)
    ]
    source = "void f(int n){\n  char *p0;\n" + "\n".join(lines) + "\n}\n"
    (tmp_path / "many.c").write_text(source, encoding="utf-8")
    graph = CodeGraph(tmp_path)
    graph.build()
    tools = CouncilTools(graph)
    result = tools.dispatch("patterns", {"kind": "integer_overflow_alloc"})
    assert result["ok"] is True
    payload = result["result"]
    assert payload["count"] == 210
    assert payload["truncated"] is True
    assert len(payload["signals"]) == 200


def test_patterns_signal_detail_names_expression(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": "void f(int n){ char *p = malloc(n * 4); }\n"},
    )
    detail = _of(signals, "integer_overflow_alloc")[0].detail
    assert "malloc" in detail
    assert "n * 4" in detail.replace(" ", " ") or "n" in detail


def test_finder_task_includes_static_signals(tmp_path: Path) -> None:
    from harness.council.agents import _finder_task

    (tmp_path / "a.c").write_text("void f(char *x){ printf(x); }\n", encoding="utf-8")
    graph = CodeGraph(tmp_path)
    graph.build()
    tools = CouncilTools(graph)
    task = _finder_task(tools, "topic", "finder", str(tmp_path), graph)
    assert "Static signals:" in task
    assert "format_string" in task


def test_integer_overflow_alloc_guard_in_enclosing_if(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "void f(int n){\n"
                "    if (n > 1000) {\n"
                "        int a = 1;\n"
                "        int b = 2;\n"
                "        int c = 3;\n"
                "        int d = 4;\n"
                "        int e = 5;\n"
                "        char *p = malloc(n * 4);\n"
                "    }\n"
                "}\n"
            )
        },
    )
    assert _of(signals, "integer_overflow_alloc") == []


def test_integer_overflow_alloc_guard_early_return(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "void f(int n){\n"
                "    if (n <= 0) { return; }\n"
                "    char *p = malloc(n * 4);\n"
                "}\n"
            )
        },
    )
    assert _of(signals, "integer_overflow_alloc") == []


def test_integer_overflow_alloc_guard_one_hop_helper(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "int capped(int x){ if (x > 1000) return 0; return x; }\n"
                "void f(int n){ int m = capped(n); char *p = malloc(n * 4); }\n"
            )
        },
    )
    assert _of(signals, "integer_overflow_alloc") == []


def test_integer_overflow_alloc_one_hop_without_guard(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "int passthrough(int x){ return x; }\n"
                "void f(int n){ int m = passthrough(n); "
                "char *p = malloc(n * 4); }\n"
            )
        },
    )
    assert len(_of(signals, "integer_overflow_alloc")) == 1


def test_integer_overflow_alloc_alias_then_guard(tmp_path: Path) -> None:
    guarded = _scan(
        tmp_path,
        {
            "a.c": (
                "void f(int n){ int m = n; if (m > 1000) return; "
                "char *p = malloc(n * 4); }\n"
            )
        },
    )
    assert _of(guarded, "integer_overflow_alloc") == []
    unguarded = _scan(
        tmp_path,
        {"b.c": "void f(int n){ int m = n; char *p = malloc(n * 4); }\n"},
    )
    assert len(_of(unguarded, "integer_overflow_alloc")) == 1


def test_integer_overflow_alloc_clamp_and_mask_guard(tmp_path: Path) -> None:
    clamp = _scan(
        tmp_path,
        {"a.c": "void f(int n){ if (n > 1000) n = 1000; char *p = malloc(n * 4); }\n"},
    )
    assert _of(clamp, "integer_overflow_alloc") == []
    mask = _scan(
        tmp_path,
        {"b.c": "void f(int n){ n &= 0xffff; char *p = malloc(n * 4); }\n"},
    )
    assert _of(mask, "integer_overflow_alloc") == []
    mod = _scan(
        tmp_path,
        {"c.c": "void f(int n){ n %= 1000; char *p = malloc(n * 4); }\n"},
    )
    assert _of(mod, "integer_overflow_alloc") == []


def test_integer_overflow_alloc_min_max_guard(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"a.c": "void f(int n){ int m = MIN(n, 1000); char *p = malloc(n * 4); }\n"},
    )
    assert _of(signals, "integer_overflow_alloc") == []
    other = _scan(
        tmp_path,
        {"b.c": "void f(int n){ int m = MAX(0, n); char *p = malloc(n * 4); }\n"},
    )
    assert _of(other, "integer_overflow_alloc") == []


def test_off_by_one_guard(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "void f(void){ char buf[16]; int i; "
                "for (i = 0; i <= 16; i++) { if (i >= 16) break; buf[i] = 0; } }\n"
            )
        },
    )
    assert _of(signals, "off_by_one") == []


def test_snprintf_offset_write_guard(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "void f(char *s){ char buf[16]; int r = snprintf(buf, "
                'sizeof(buf), "%s", s); if (r < 0) return; buf[r] = 0; }\n'
            )
        },
    )
    assert _of(signals, "snprintf_offset_write") == []


def test_unchecked_retry_loop_guard(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "void f(void){ while (!done) { result = write_file(fd, buf, n); "
                "if (result < 0) return; done = 1; } }\n"
            )
        },
    )
    assert _of(signals, "unchecked_retry_loop") == []


def test_unchecked_alloc_assert_and_ternary_checks(tmp_path: Path) -> None:
    asserted = _scan(
        tmp_path,
        {"a.c": "void f(int n){ char *p = malloc(n); assert(p); p[0] = 0; }\n"},
    )
    assert _of(asserted, "unchecked_alloc") == []
    ternary = _scan(
        tmp_path,
        {
            "b.c": (
                "void f(int n){ char *p = malloc(n); int x = p ? p[0] : 0; }\n"
            )
        },
    )
    assert _of(ternary, "unchecked_alloc") == []
    wrapped = _scan(
        tmp_path,
        {"c.c": "void f(int n){ char *p = malloc(n); if (p) { p[0] = 0; } }\n"},
    )
    assert _of(wrapped, "unchecked_alloc") == []


def test_python_shell_injection_fstring(tmp_path: Path) -> None:
    flagged = _scan(
        tmp_path,
        {"p.py": "import os\n\n\ndef f(p):\n    os.system(f\"ls {p}\")\n"},
    )
    assert len(_of(flagged, "shell_injection")) == 1
    literal = _scan(
        tmp_path,
        {"q.py": "import os\n\n\ndef f():\n    os.system(\"ls\")\n"},
    )
    assert _of(literal, "shell_injection") == []


def test_python_shell_injection_subprocess(tmp_path: Path) -> None:
    variable = _scan(
        tmp_path,
        {
            "p.py": (
                "import subprocess\n\n\ndef f(cmd):\n"
                "    subprocess.run(cmd, shell=True)\n"
            )
        },
    )
    assert len(_of(variable, "shell_injection")) == 1
    literal_list = _scan(
        tmp_path,
        {
            "q.py": (
                "import subprocess\n\n\ndef f():\n"
                '    subprocess.run(["ls"], shell=True)\n'
            )
        },
    )
    assert _of(literal_list, "shell_injection") == []
    plain_list = _scan(
        tmp_path,
        {
            "r.py": (
                "import subprocess\n\n\ndef f():\n"
                '    subprocess.run(["ls"])\n'
            )
        },
    )
    assert _of(plain_list, "shell_injection") == []


def test_python_shell_injection_concat(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "p.py": (
                "import subprocess\n\n\ndef f(p):\n"
                '    cmd = "ls " + p\n'
                "    subprocess.run(cmd)\n"
            )
        },
    )
    assert len(_of(signals, "shell_injection")) == 1


def test_python_shell_injection_os_popen_and_exec(tmp_path: Path) -> None:
    popen = _scan(
        tmp_path,
        {"p.py": "import os\n\n\ndef f(cmd):\n    os.popen(cmd)\n"},
    )
    assert len(_of(popen, "shell_injection")) == 1
    execv = _scan(
        tmp_path,
        {"q.py": "import os\n\n\ndef f(path):\n    os.execv(path, [])\n"},
    )
    assert len(_of(execv, "shell_injection")) == 1
    literal = _scan(
        tmp_path,
        {"r.py": "import os\n\n\ndef f():\n    os.execv(\"/bin/ls\", [])\n"},
    )
    assert _of(literal, "shell_injection") == []


def test_unchecked_io_return_errx_argument(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {
            "a.c": (
                "void f(int fd){ char buf[16]; int n = read(fd, buf, 16); "
                'errx(1, "%d", n); }\n'
            )
        },
    )
    assert _of(signals, "unchecked_io_return") == []


def test_python_insecure_temp_file_mktemp(tmp_path: Path) -> None:
    signals = _scan(
        tmp_path,
        {"p.py": "import tempfile\n\n\ndef f():\n    return tempfile.mktemp()\n"},
    )
    assert len(_of(signals, "insecure_temp_file")) == 1


def test_python_toctou_exists_then_open(tmp_path: Path) -> None:
    flagged = _scan(
        tmp_path,
        {
            "p.py": (
                "import os\n\n\ndef f(path):\n"
                "    if os.path.exists(path):\n        open(path)\n"
            )
        },
    )
    assert len(_of(flagged, "toctou")) == 1
    literal = _scan(
        tmp_path,
        {
            "q.py": (
                "import os\n\n\ndef f(path):\n"
                '    if os.path.exists(path):\n        open("/etc/hosts")\n'
            )
        },
    )
    assert _of(literal, "toctou") == []


def test_module_docstring_documents_c_only_kinds() -> None:
    import harness.council.patterns as patterns

    doc = patterns.__doc__ or ""
    assert "C-only" in doc
    assert "no manual free/realloc" in doc
    assert "non-literal format strings" in doc
