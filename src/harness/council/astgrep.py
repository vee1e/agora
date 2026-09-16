"""Tree-sitter powered structural search over a checkout.

Patterns may use ast-grep style metavariables: ``$NAME`` matches one node,
``$$$NAME`` captures the surrounding list. Metavariable-free patterns are
matched as literal substrings; patterns with metavariables must translate into
a tree-sitter query or the search is reported as unsupported. Read-only, no
shell.
"""
from __future__ import annotations

import bisect
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import tree_sitter_c
import tree_sitter_python
from tree_sitter import Language, Node, Parser, Query, QueryCursor

MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_FILES = 5000
DEFAULT_LIMIT = 200
MAX_PATTERN_LENGTH = 4096
MAX_SCAN_BYTES = 64 * 1024 * 1024
MAX_QUERY_CACHE = 1024
TOOL_NAME = "sg_search"
OUTSIDE_PLACEHOLDER = "<outside>"

_EXT_LANG = {".c": "c", ".h": "c", ".py": "python"}
_LANG_ALIASES = {"c": "c", "cpp": "c", "py": "python", "python": "python"}
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
    }
)
_META_RE = re.compile(r"\$\$\$([A-Za-z_][A-Za-z0-9_]*)|\$([A-Za-z_][A-Za-z0-9_]*)")
_MATCH_CAPTURE_PREFIX = "__sg_match_"
_INTERNAL_CAPTURE_PREFIX = "__sg"
_WRAP_FUNCTION = "__sg_pattern"
_DECLARATOR_FIELD = "declarator"


@dataclass
class Match:
    file: str
    line: int
    column: int
    end_line: int
    end_column: int
    text: str
    captures: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Translation:
    text: str
    variadic: frozenset[str]


class _SourceMap:
    def __init__(self, data: bytes) -> None:
        starts = [0]
        for index, byte in enumerate(data):
            if byte == 0x0A:
                starts.append(index + 1)
        self._starts = starts

    def position(self, offset: int) -> tuple[int, int]:
        index = bisect.bisect_right(self._starts, offset) - 1
        if index < 0:
            index = 0
        return index + 1, offset - self._starts[index] + 1


def _escape_query_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _substitute(pattern: str) -> tuple[str, dict[str, tuple[str, bool]]]:
    metas: dict[str, tuple[str, bool]] = {}

    def replace(match: re.Match[str]) -> str:
        name = match.group(1) if match.group(1) is not None else match.group(2)
        sentinel = f"__sg_meta_{len(metas)}__"
        metas[sentinel] = (name, match.group(1) is not None)
        return sentinel

    return _META_RE.sub(replace, pattern), metas


def _has_error(node: Node) -> bool:
    if node.is_error or node.is_missing:
        return True
    return any(_has_error(child) for child in node.children)


def _meta_leaf(
    node: Node, metas: dict[str, tuple[str, bool]]
) -> tuple[str, bool] | None:
    if node.is_named and node.child_count == 0:
        text = node.text.decode("utf-8", "replace")
        if text in metas:
            return metas[text]
    return None


def _meta_capture(name: str, field: str | None) -> str:
    if field == _DECLARATOR_FIELD:
        return (
            f"[(identifier) @{name} "
            f"(field_identifier) @{name} "
            f"(pointer_declarator declarator: (identifier) @{name}) "
            f"(array_declarator declarator: (identifier) @{name}) "
            f"(function_declarator declarator: (identifier) @{name})]"
        )
    return f"(_) @{name}"


def _emit(node: Node, metas: dict[str, tuple[str, bool]]) -> str:
    meta = _meta_leaf(node, metas)
    if meta is not None:
        name, variadic = meta
        if variadic:
            return ""
        return f"(_) @{name}"
    text = node.text.decode("utf-8", "replace")
    if not node.is_named:
        return f'"{_escape_query_string(text)}"'

    variadic_names: list[str] = []
    parts: list[str] = []
    for index, child in enumerate(node.children):
        child_meta = _meta_leaf(child, metas)
        if child_meta is not None:
            name, variadic = child_meta
            if variadic:
                variadic_names.append(name)
                continue
            parts.append(_meta_capture(name, node.field_name_for_child(index)))
            continue
        parts.append(_emit(child, metas))

    if node.child_count == 0:
        literal = f"__sglit_{id(node)}"
        return (
            f"({node.type}) @{literal} "
            f'(#eq? @{literal} "{_escape_query_string(text)}")'
        )

    joiner = " " if variadic_names else " . "
    inner = joiner.join(part for part in parts if part)
    query = f"({node.type}{' ' + inner if inner else ''})"
    for name in variadic_names:
        query += f" @{name}"
    return query


def _wrapper(lang: str, body: str) -> str:
    if lang == "python":
        indented = "\n".join(f"    {line}" for line in body.splitlines())
        return f"def {_WRAP_FUNCTION}():\n{indented}\n"
    stripped = body.rstrip()
    if not stripped.endswith((";", "}")):
        body = f"{body};"
    return f"void {_WRAP_FUNCTION}(void) {{\n{body}\n}}"


def _top_nodes(root: Node, wrapped: bool) -> list[Node]:
    if not wrapped:
        return [child for child in root.children if child.is_named]
    for node in root.named_children:
        if node.type == "function_definition":
            for body in node.children:
                if body.type in ("compound_statement", "block"):
                    return [child for child in body.named_children]
    return []


def _unwrap_expression(node: Node) -> Node:
    current = node
    while current.type == "expression_statement" and current.named_child_count == 1:
        child = current.named_children[0]
        if not child.is_named or child.type == "comment":
            break
        current = child
    return current


def _translate(pattern: str, lang: str, language: Language) -> _Translation | None:
    substituted, metas = _substitute(pattern)
    variadic = frozenset(name for name, is_variadic in metas.values() if is_variadic)
    parser = Parser(language)
    for wrapped, source in (
        (False, substituted),
        (True, _wrapper(lang, substituted)),
    ):
        tree = parser.parse(source.encode("utf-8"))
        if _has_error(tree.root_node):
            continue
        nodes = [_unwrap_expression(node) for node in _top_nodes(tree.root_node, wrapped)]
        if not nodes:
            continue
        patterns = [
            f"({_emit(node, metas)}) @{_MATCH_CAPTURE_PREFIX}{index}"
            for index, node in enumerate(nodes)
        ]
        text = " ".join(patterns)
        try:
            Query(language, text)
        except Exception:
            continue
        return _Translation(text=text, variadic=variadic)
    return None


def _capture_text(node: Node, data: bytes, variadic: bool) -> str:
    if variadic and node.child_count >= 2:
        inner = data[node.children[0].end_byte : node.children[-1].start_byte]
        return inner.decode("utf-8", "replace")
    return data[node.start_byte : node.end_byte].decode("utf-8", "replace")


class AstGrep:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._languages = {
            "c": Language(tree_sitter_c.language()),
            "python": Language(tree_sitter_python.language()),
        }
        self._parsers = {
            name: Parser(language) for name, language in self._languages.items()
        }
        self._query_cache: dict[tuple[str, str], tuple[Query | None, frozenset[str]]] = {}

    def specs(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "description": (
                        "Structural code search over the checkout using "
                        "tree-sitter patterns. Supports $NAME metavariables "
                        "and $$$ARGS variadic captures; plain text patterns "
                        "match literally. Read-only."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": (
                                    "Search pattern, e.g. 'int $X = 1;' or "
                                    "'printf($$$ARGS)'."
                                ),
                            },
                            "lang": {
                                "type": "string",
                                "enum": ["c", "python"],
                                "description": "Language filter. Omit for all.",
                            },
                            "path": {
                                "type": "string",
                                "description": "Subpath under the checkout root.",
                            },
                        },
                        "required": ["pattern"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def dispatch(self, name: str, arguments: dict) -> dict:
        try:
            if name != TOOL_NAME:
                return {"ok": False, "error": f"unknown tool: {name}"}
            if not isinstance(arguments, dict):
                return {"ok": False, "error": "arguments must be an object"}
            pattern = arguments.get("pattern")
            if not isinstance(pattern, str) or not pattern:
                return {"ok": False, "error": "pattern is required"}
            if len(pattern) > MAX_PATTERN_LENGTH:
                return {
                    "ok": False,
                    "error": f"pattern too long: {len(pattern)} > {MAX_PATTERN_LENGTH}",
                }
            lang = arguments.get("lang")
            if lang is not None and not isinstance(lang, str):
                return {"ok": False, "error": "lang must be a string"}
            path = arguments.get("path")
            if path is not None and not isinstance(path, str):
                return {"ok": False, "error": "path must be a string"}
            if _META_RE.search(pattern) is not None and not self._is_supported(pattern, lang):
                return {"ok": False, "error": "pattern not supported"}
            matches = self.search(pattern, lang=lang, path=path)
            return {"ok": True, "result": [match.to_dict() for match in matches]}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def search(
        self,
        pattern: str,
        lang: str | None = None,
        path: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[Match]:
        try:
            return self._search(pattern, lang, path, limit)
        except Exception:
            return []

    def _search(
        self,
        pattern: str,
        lang: str | None,
        path: str | None,
        limit: int,
    ) -> list[Match]:
        if not isinstance(pattern, str) or not pattern:
            return []
        if len(pattern) > MAX_PATTERN_LENGTH:
            return []
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            return []
        if lang is None:
            allowed: set[str] | None = None
        else:
            normalized = _LANG_ALIASES.get(lang.lower())
            if normalized is None:
                return []
            allowed = {normalized}
        if _META_RE.search(pattern) is not None and not self._is_supported(pattern, lang):
            return []
        if isinstance(path, str) and not path:
            path = None

        base = self.root.resolve()
        scan_root = base
        single_file: Path | None = None
        if path is not None:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = base / candidate
            candidate = candidate.resolve()
            if candidate != base and base not in candidate.parents:
                return []
            if not candidate.exists():
                return []
            if candidate.is_file():
                single_file = candidate
                scan_root = candidate.parent
            else:
                scan_root = candidate

        literal = _META_RE.search(pattern) is None
        literal_bytes = pattern.encode("utf-8")
        results: list[Match] = []
        scanned_bytes = 0

        for file_path, file_lang in self._iter_files(scan_root, allowed, single_file, base):
            if len(results) >= limit or scanned_bytes >= MAX_SCAN_BYTES:
                break
            try:
                data = file_path.read_bytes()
            except OSError:
                continue
            scanned_bytes += len(data)
            if len(data) > MAX_FILE_BYTES:
                continue
            relative = self._relative(file_path, base)
            source_map = _SourceMap(data)
            if literal:
                file_matches = self._literal_matches(
                    data, literal_bytes, relative, source_map, limit
                )
            else:
                file_matches = self._query_matches(
                    data, pattern, file_lang, relative, source_map, limit
                )
            file_matches.sort(key=_match_key)
            for match in file_matches:
                results.append(match)
                if len(results) >= limit:
                    break
        return results[:limit]

    def _relative(self, file_path: Path, base: Path) -> str:
        try:
            resolved = file_path.resolve()
        except OSError:
            return OUTSIDE_PLACEHOLDER
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError:
            return OUTSIDE_PLACEHOLDER

    def _iter_files(
        self,
        scan_root: Path,
        allowed: set[str] | None,
        single_file: Path | None,
        base: Path,
    ) -> Iterator[tuple[Path, str]]:
        if single_file is not None:
            file_lang = _EXT_LANG.get(single_file.suffix.lower())
            if file_lang is not None and (allowed is None or file_lang in allowed):
                yield single_file, file_lang
            return
        scanned = 0
        for dirpath, dirnames, filenames in os.walk(scan_root):
            dirnames[:] = sorted(
                name
                for name in dirnames
                if name not in _SKIP_DIRS and not name.startswith(".")
            )
            for name in sorted(filenames):
                file_lang = _EXT_LANG.get(Path(name).suffix.lower())
                if file_lang is None or (allowed is not None and file_lang not in allowed):
                    continue
                candidate = Path(dirpath) / name
                if candidate.is_symlink():
                    continue
                try:
                    resolved = candidate.resolve()
                except OSError:
                    continue
                if resolved != base and base not in resolved.parents:
                    continue
                scanned += 1
                if scanned > MAX_FILES:
                    return
                yield candidate, file_lang

    def _literal_matches(
        self,
        data: bytes,
        needle: bytes,
        relative: str,
        source_map: _SourceMap,
        limit: int,
    ) -> list[Match]:
        matches: list[Match] = []
        if not needle or limit <= 0:
            return matches
        start = 0
        while len(matches) < limit:
            index = data.find(needle, start)
            if index < 0:
                break
            end = index + len(needle)
            line, column = source_map.position(index)
            end_line, end_column = source_map.position(end)
            matches.append(
                Match(
                    file=relative,
                    line=line,
                    column=column,
                    end_line=end_line,
                    end_column=end_column,
                    text=data[index:end].decode("utf-8", "replace"),
                    captures={},
                )
            )
            start = end
        return matches

    def _query_matches(
        self,
        data: bytes,
        pattern: str,
        file_lang: str,
        relative: str,
        source_map: _SourceMap,
        limit: int,
    ) -> list[Match]:
        query, variadic = self._query_for(pattern, file_lang)
        if query is None:
            return []
        parser = self._parsers[file_lang]
        tree = parser.parse(data)
        cursor = QueryCursor(query)
        matches: list[Match] = []
        for _, captures in cursor.matches(tree.root_node):
            if len(matches) >= limit:
                break
            if not captures:
                continue
            roots = [
                nodes[0]
                for key, nodes in captures.items()
                if key.startswith(_MATCH_CAPTURE_PREFIX) and nodes
            ]
            if not roots:
                continue
            node = roots[0]
            user_captures = {
                key: _capture_text(nodes[0], data, key in variadic)
                for key, nodes in captures.items()
                if not key.startswith(_INTERNAL_CAPTURE_PREFIX) and nodes
            }
            line, column = source_map.position(node.start_byte)
            end_line, end_column = source_map.position(node.end_byte)
            matches.append(
                Match(
                    file=relative,
                    line=line,
                    column=column,
                    end_line=end_line,
                    end_column=end_column,
                    text=data[node.start_byte:node.end_byte].decode("utf-8", "replace"),
                    captures=user_captures,
                )
            )
        return matches

    def _is_supported(self, pattern: str, lang: str | None) -> bool:
        if lang is None:
            candidates = list(self._languages)
        else:
            normalized = _LANG_ALIASES.get(lang.lower())
            if normalized is None:
                return False
            candidates = [normalized]
        return any(self._query_for(pattern, name)[0] is not None for name in candidates)

    def _query_for(self, pattern: str, lang: str) -> tuple[Query | None, frozenset[str]]:
        key = (pattern, lang)
        if key in self._query_cache:
            return self._query_cache[key]
        if len(self._query_cache) >= MAX_QUERY_CACHE:
            self._query_cache.clear()
        translation = _translate(pattern, lang, self._languages[lang])
        query: Query | None = None
        variadic: frozenset[str] = frozenset()
        if translation is not None:
            variadic = translation.variadic
            try:
                query = Query(self._languages[lang], translation.text)
            except Exception:
                query = None
        self._query_cache[key] = (query, variadic)
        return query, variadic


def _match_key(match: Match) -> tuple[str, int, int, int, int]:
    return (match.file, match.line, match.column, match.end_line, match.end_column)
