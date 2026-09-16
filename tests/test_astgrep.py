from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from harness.council.astgrep import AstGrep

C_SOURCE = """#include <stdio.h>

int add(int a, int b) {
    int total = a + b;
    return total;
}

void greet(void) {
    printf("hello %s\\n", "world");
    printf("%d\\n", 42);
}
"""

PY_SOURCE = """def add(a, b):
    total = a + b
    return total

value = add(1, 2)
"""

MULTILINE_C = """int main(void) {
    int value = 7;
    return value;
}
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.c").write_text(C_SOURCE)
    (source / "b.py").write_text(PY_SOURCE)
    (tmp_path / "notes.txt").write_text("printf( ignored\n")
    return tmp_path


@pytest.fixture
def grep(repo: Path) -> AstGrep:
    return AstGrep(repo)


def test_literal_pattern_in_c(grep: AstGrep) -> None:
    results = grep.search("printf(", lang="c")

    assert len(results) == 2
    assert {match.file for match in results} == {"src/a.c"}
    assert all(match.captures == {} for match in results)
    assert [match.line for match in results] == [9, 10]
    assert all(match.text == "printf(" for match in results)


def test_metavariable_captures_name(grep: AstGrep) -> None:
    results = grep.search("int $X = $A + $B;", lang="c")

    assert len(results) == 1
    match = results[0]
    assert match.captures == {"X": "total", "A": "a", "B": "b"}
    assert match.text == "int total = a + b;"
    assert match.file == "src/a.c"


def test_metavariable_in_python(grep: AstGrep) -> None:
    results = grep.search("$X = add(1, 2)", lang="python")

    assert len(results) == 1
    assert results[0].captures == {"X": "value"}
    assert results[0].text == "value = add(1, 2)"


def test_variadic_capture(grep: AstGrep) -> None:
    results = grep.search("printf($$$ARGS)", lang="c")

    assert len(results) == 2
    assert "hello" in results[0].captures["ARGS"]
    assert "42" in results[1].captures["ARGS"]


def test_language_filter(grep: AstGrep) -> None:
    assert grep.search("printf(", lang="python") == []
    assert grep.search("printf(", lang="py") == []
    assert len(grep.search("int $X = $A + $B;", lang="cpp")) == 1
    assert len(grep.search("$X = add(1, 2)", lang="py")) == 1


def test_no_lang_searches_all_files(grep: AstGrep) -> None:
    files = {match.file for match in grep.search("add")}

    assert files == {"src/a.c", "src/b.py"}


def test_subpath_filter(grep: AstGrep) -> None:
    assert len(grep.search("printf(", path="src/a.c")) == 2
    assert grep.search("printf(", path="src/b.py") == []
    assert len(grep.search("printf(", path="src")) == 2
    assert grep.search("printf(", path="missing") == []


def test_no_match_returns_empty(grep: AstGrep) -> None:
    assert grep.search("zzz_missing_symbol") == []


def test_untranslatable_metavariable_is_not_a_false_empty(repo: Path, grep: AstGrep) -> None:
    (repo / "src" / "literal.c").write_text("/* $X + */\nint x = 0;\n")

    assert grep.search("$X +", lang="c") == []

    response = grep.dispatch("sg_search", {"pattern": "$X +", "lang": "c"})

    assert response == {"ok": False, "error": "pattern not supported"}


def test_limit_is_honored(grep: AstGrep) -> None:
    assert len(grep.search("printf(", lang="c", limit=1)) == 1
    assert grep.search("printf(", lang="c", limit=0) == []


def test_deterministic_ordering(tmp_path: Path) -> None:
    (tmp_path / "one.c").write_text("int alpha = 1;\nint alpha = 2;\n")
    (tmp_path / "two.c").write_text("int alpha = 3;\n")
    grep = AstGrep(tmp_path)

    first = grep.search("alpha", lang="c")
    second = grep.search("alpha", lang="c")

    assert first == second
    keys = [(match.file, match.line, match.column) for match in first]
    assert keys == sorted(keys)
    assert [match.file for match in first] == ["one.c", "one.c", "two.c"]


def test_line_and_column_on_multiline_file(tmp_path: Path) -> None:
    (tmp_path / "m.c").write_text(MULTILINE_C)
    grep = AstGrep(tmp_path)

    results = grep.search("int $X = 7;", lang="c")

    assert len(results) == 1
    match = results[0]
    assert match.line == 2
    assert match.column == 5
    assert match.end_line == 2
    assert match.end_column == 19
    assert match.text == "int value = 7;"
    assert match.captures == {"X": "value"}


def test_dispatch_success(grep: AstGrep) -> None:
    response = grep.dispatch("sg_search", {"pattern": "printf(", "lang": "c"})

    assert response["ok"] is True
    assert len(response["result"]) == 2
    assert response["result"][0]["file"] == "src/a.c"
    assert set(response["result"][0]) == {
        "file",
        "line",
        "column",
        "end_line",
        "end_column",
        "text",
        "captures",
    }


def test_dispatch_bad_args(grep: AstGrep) -> None:
    assert grep.dispatch("sg_search", {})["ok"] is False
    assert grep.dispatch("sg_search", "not-a-dict")["ok"] is False
    assert grep.dispatch("sg_search", {"pattern": 3})["ok"] is False
    assert grep.dispatch("sg_search", {"pattern": "printf(", "lang": 5})["ok"] is False
    assert grep.dispatch("sg_search", {"pattern": "printf(", "path": 5})["ok"] is False


def test_dispatch_unknown_tool(grep: AstGrep) -> None:
    response = grep.dispatch("nope", {"pattern": "x"})

    assert response["ok"] is False
    assert "error" in response


def test_specs_shape(grep: AstGrep) -> None:
    specs = grep.specs()

    assert len(specs) == 1
    function = specs[0]["function"]
    assert function["name"] == "sg_search"
    parameters = function["parameters"]
    assert parameters["required"] == ["pattern"]
    assert set(parameters["properties"]) == {"pattern", "lang", "path"}
    assert parameters["properties"]["lang"]["enum"] == ["c", "python"]


def test_skips_large_and_unsupported_files(tmp_path: Path) -> None:
    (tmp_path / "big.c").write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    (tmp_path / "small.c").write_text("int marker = 1;\n")
    (tmp_path / "other.txt").write_text("marker elsewhere\n")
    grep = AstGrep(tmp_path)

    results = grep.search("marker")

    assert [match.file for match in results] == ["small.c"]


def test_symlinked_files_are_skipped(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    secret = outside / "secret.c"
    secret.write_text("int leaked_marker = 1;\n")
    source = tmp_path / "src"
    source.mkdir()
    (source / "real.c").write_text("int real_marker = 1;\n")
    try:
        (source / "leak.c").symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")
    grep = AstGrep(tmp_path)

    assert grep.search("leaked_marker") == []
    assert [match.file for match in grep.search("real_marker")] == ["src/real.c"]


def test_limit_stops_early_on_large_repeated_file(tmp_path: Path) -> None:
    (tmp_path / "big.c").write_bytes(b"a" * (2 * 1024 * 1024))
    grep = AstGrep(tmp_path)

    started = time.monotonic()
    results = grep.search("a", lang="c", limit=10)
    elapsed = time.monotonic() - started

    assert len(results) == 10
    assert all(match.text == "a" for match in results)
    assert elapsed < 5.0


def test_single_metavariable_requires_one_argument(tmp_path: Path) -> None:
    (tmp_path / "calls.c").write_text(
        "void f(void) {\n"
        '    printf("a", "b");\n'
        '    printf("solo");\n'
        "}\n"
    )
    grep = AstGrep(tmp_path)

    results = grep.search("printf($X)", lang="c")

    assert len(results) == 1
    assert results[0].captures == {"X": '"solo"'}


def test_declaration_pattern_requires_no_initializer(tmp_path: Path) -> None:
    (tmp_path / "decls.c").write_text(
        "int x = 1;\nint y;\nchar *p = 0;\nchar *q;\n"
    )
    grep = AstGrep(tmp_path)

    results = grep.search("$T $X;", lang="c")

    assert [match.captures["X"] for match in results] == ["y", "q"]
    assert all("=" not in match.text for match in results)


def test_initialized_declaration_captures_plain_name(tmp_path: Path) -> None:
    (tmp_path / "decls.c").write_text("char *p = 0;\n")
    grep = AstGrep(tmp_path)

    results = grep.search("$T $X = $Y;", lang="c")

    assert len(results) == 1
    assert results[0].captures == {"T": "char", "X": "p", "Y": "0"}


def test_variadic_capture_excludes_delimiters(grep: AstGrep) -> None:
    results = grep.search("printf($$$ARGS)", lang="c")

    assert [match.captures["ARGS"] for match in results] == [
        '"hello %s\\n", "world"',
        '"%d\\n", 42',
    ]


def test_call_nested_in_initializer_is_found(tmp_path: Path) -> None:
    (tmp_path / "n.c").write_text(
        "void f(int fd){\n"
        "    char buf[16];\n"
        "    size_t n = strlen(input);\n"
        "    ssize_t r = read(fd, buf, n);\n"
        "}\n"
    )
    grep = AstGrep(tmp_path)

    calls = grep.search("strlen($A)", lang="c")

    assert len(calls) == 1
    assert calls[0].text == "strlen(input)"
    assert calls[0].captures == {"A": "input"}

    reads = grep.search("read($$$ARGS)", lang="c")

    assert len(reads) == 1
    assert reads[0].text == "read(fd, buf, n)"
    assert reads[0].captures == {"ARGS": "fd, buf, n"}


def test_query_cache_stays_bounded(tmp_path: Path) -> None:
    from harness.council.astgrep import MAX_QUERY_CACHE

    (tmp_path / "x.c").write_text("int x = 0;\n")
    grep = AstGrep(tmp_path)

    for index in range(MAX_QUERY_CACHE + 50):
        grep.search(f"call_{index}($A)", lang="c")

    assert len(grep._query_cache) <= MAX_QUERY_CACHE


def test_oversized_pattern_is_rejected(grep: AstGrep) -> None:
    response = grep.dispatch("sg_search", {"pattern": "a" * 5000})

    assert response["ok"] is False
    assert "error" in response
    assert grep.search("a" * 5000) == []


def test_relative_never_returns_absolute_path(tmp_path: Path) -> None:
    grep = AstGrep(tmp_path)
    outside = Path("/etc") / "definitely_outside_astgrep"

    result = grep._relative(outside, tmp_path.resolve())

    assert result == "<outside>"
    assert not os.path.isabs(result)
