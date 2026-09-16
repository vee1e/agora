"""Tree-sitter based call/taint graph over a code checkout.

Read-only. Parses C, C++ headers and Python, extracts functions, calls,
allocations, buffer sizes, taint sources and sinks. Never imports or executes
any analysed file.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from tree_sitter import Language, Node, Parser

import tree_sitter_c
import tree_sitter_python

if TYPE_CHECKING:
    from .patterns import Signal

MAX_FILE_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_FILES = 5000
MAX_PATH_EXPANSIONS = 100_000
MAX_DECLARATOR_DEPTH = 10_000
MAX_PY_SCOPE_DEPTH = 200
C_EXTENSIONS = {".c", ".h"}
PY_EXTENSIONS = {".py"}

PY_SCOPE_CONTAINERS = {
    "block",
    "module",
    "if_statement",
    "elif_clause",
    "else_clause",
    "try_statement",
    "except_clause",
    "finally_clause",
    "with_statement",
    "for_statement",
    "while_statement",
    "match_statement",
    "case_clause",
}

ALLOC_NAMES = {"malloc", "calloc", "realloc", "alloca", "strdup", "strndup"}
PY_ALLOC_NAMES = {"bytearray", "bytes", "list", "dict", "set"}

TAINT_SOURCE_NAMES = {
    "read": "read",
    "recv": "recv",
    "recvfrom": "recv",
    "recvmsg": "recv",
    "fgets": "fgets",
    "gets": "fgets",
    "getline": "read",
    "getenv": "getenv",
    "scanf": "read",
    "fscanf": "read",
    "sscanf": "read",
    "getchar": "read",
    "fread": "read",
}
PY_TAINT_SOURCE_NAMES = {
    "input": "input",
    "sys.argv": "argv",
    "os.environ": "getenv",
    "os.getenv": "getenv",
    "environ": "getenv",
}

FORMAT_SINKS = {"printf", "fprintf", "sprintf", "snprintf", "syslog", "dprintf"}
STRING_SINKS = {"strcpy", "strcat", "strncpy", "strncat", "memcpy", "memmove", "sprintf"}
EXEC_SINKS = {"system", "popen", "execl", "execle", "execlp", "execv", "execve", "execvp", "execvpe"}
TAINT_SOURCE_DEST_ARGS: dict[str, int] = {
    "read": 1,
    "recv": 1,
    "recvfrom": 1,
    "recvmsg": 1,
    "fgets": 0,
    "gets": 0,
    "getline": 0,
    "scanf": 0,
}
COPY_SINK_DEST_ARG0 = {"strcpy", "strcat", "memcpy", "sprintf", "snprintf"}
PY_EXEC_SINKS = {"eval", "exec", "os.system", "os.popen", "subprocess.call", "subprocess.run", "subprocess.Popen"}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "dist",
    "build",
    ".idea",
    ".vscode",
}


@dataclass
class Alloc:
    name: str
    line: int
    size_expr: str | None = None


@dataclass
class BufferSize:
    name: str
    line: int
    size_expr: str | None = None
    kind: str = "array"


@dataclass
class TaintSource:
    name: str
    line: int
    kind: str


@dataclass
class Sink:
    name: str
    line: int
    arg: str


@dataclass
class Function:
    name: str
    file: str
    line: int
    end_line: int
    params: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    call_lines: list[tuple[str, int]] = field(default_factory=list)
    allocs: list[Alloc] = field(default_factory=list)
    buffers: list[BufferSize] = field(default_factory=list)
    sources: list[TaintSource] = field(default_factory=list)
    sinks: list[Sink] = field(default_factory=list)


class CodeGraph:
    """Call graph and light taint facts for one checkout."""

    def __init__(self, root: Path, max_files: int = DEFAULT_MAX_FILES) -> None:
        self._root = Path(root)
        self._max_files = max_files
        self._files: list[str] = []
        self._functions: dict[str, list[Function]] = {}
        self._skipped: list[str] = []
        self._built = False
        self._signals: list[Signal] = []
        self._c_lang = Language(tree_sitter_c.language())
        self._py_lang = Language(tree_sitter_python.language())
        self._c_parser = Parser(self._c_lang)
        self._py_parser = Parser(self._py_lang)
        self._chain_cache: dict[tuple[str, int, int, str], dict[str, str]] = {}
        self._reach_cache: dict[str, set[str]] = {}

    # ----- public API -----

    def build(self) -> None:
        self._files = []
        self._functions = {}
        self._skipped = []
        self._reach_cache = {}
        self._signals = []
        count = 0
        for path in self._iter_files():
            if count >= self._max_files:
                self._skipped.append(str(path))
                continue
            rel = self._relative(path)
            if self._parse_file(path, rel):
                self._files.append(rel)
                count += 1
        from .patterns import PatternScanner

        self._signals = PatternScanner(self._root).scan()
        self._built = True

    @property
    def files(self) -> list[str]:
        return list(self._files)

    @property
    def functions(self) -> dict[str, list[Function]]:
        return self._functions

    @property
    def skipped(self) -> list[str]:
        return list(self._skipped)

    @property
    def signals(self) -> list[Signal]:
        return list(self._signals)

    def signals_for(
        self, file: str | None = None, kind: str | None = None
    ) -> list[Signal]:
        out: list[Signal] = []
        for signal in self._signals:
            if file is not None and signal.file != file:
                continue
            if kind is not None and signal.kind != kind:
                continue
            out.append(signal)
        return out

    def callers_of(self, fn: str) -> list[tuple[Function, int]]:
        out: list[tuple[Function, int]] = []
        for _name, funcs in self._functions.items():
            for caller in funcs:
                for callee, line in caller.call_lines:
                    if callee == fn:
                        out.append((caller, line))
        out.sort(key=lambda item: (item[0].file, item[1], item[0].line, item[0].name))
        return out

    def _sink_callers(self, sink: str) -> set[str]:
        terminal: set[str] = set()
        for name, funcs in self._functions.items():
            for func in funcs:
                if sink in func.calls:
                    terminal.add(name)
        return terminal

    def call_path(
        self,
        source: str,
        sink: str,
        max_depth: int = 8,
        max_paths: int = 20,
    ) -> list[list[str]]:
        starts = self._functions.get(source, [])
        if not starts:
            return []
        terminal = self._sink_callers(sink)
        if not terminal and not self._functions.get(sink):
            return []
        reachable = self._can_reach(sink)
        if source not in reachable:
            return []
        found: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        budget = MAX_PATH_EXPANSIONS

        def record(chain: list[str]) -> None:
            key = tuple(chain)
            if key not in seen:
                seen.add(key)
                found.append(list(chain))

        def dfs(chain: list[str], depth: int) -> None:
            nonlocal budget
            if budget <= 0 or len(found) >= max_paths:
                return
            current = chain[-1]
            if current == sink:
                if len(chain) > 1:
                    record(chain)
                return
            if current in terminal:
                if depth < max_depth:
                    record(chain + [sink])
                return
            if depth >= max_depth:
                return
            for callee in self._sorted_calls(current):
                if budget <= 0 or len(found) >= max_paths:
                    return
                budget -= 1
                if callee in chain:
                    continue
                if callee not in reachable:
                    continue
                if not self._functions.get(callee):
                    continue
                chain.append(callee)
                dfs(chain, depth + 1)
                chain.pop()

        for fn in starts:
            if budget <= 0:
                break
            dfs([fn.name], 0)
        return found

    def _can_reach(self, sink: str) -> set[str]:
        cached = self._reach_cache.get(sink)
        if cached is not None:
            return cached
        reverse: dict[str, set[str]] = {}
        for funcs in self._functions.values():
            for func in funcs:
                for callee in func.calls:
                    reverse.setdefault(callee, set()).add(func.name)
        reachable: set[str] = {sink}
        stack = [sink]
        while stack:
            current = stack.pop()
            for caller in reverse.get(current, ()):
                if caller not in reachable:
                    reachable.add(caller)
                    stack.append(caller)
        self._reach_cache[sink] = reachable
        return reachable

    def alloc_of(self, fn: str, ptr: str | None = None) -> list[Alloc]:
        out: list[Alloc] = []
        for func in self._functions.get(fn, []):
            for alloc in func.allocs:
                if ptr is None or ptr in alloc.name:
                    out.append(alloc)
        return out

    def size_expr(self, fn: str, buf: str) -> list[BufferSize]:
        out: list[BufferSize] = []
        for func in self._functions.get(fn, []):
            for buffer in func.buffers:
                if buf in buffer.name:
                    out.append(buffer)
            for alloc in func.allocs:
                if buf in alloc.name:
                    out.append(
                        BufferSize(
                            name=alloc.name,
                            line=alloc.line,
                            size_expr=alloc.size_expr,
                            kind="heap",
                        )
                    )
        return out

    def tainted(self, fn: str, expr: str) -> bool:
        funcs = self._functions.get(fn, [])
        if not funcs:
            return False
        target = expr.strip()
        for func in funcs:
            if self._expr_tainted(func, target):
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self._root),
            "files": list(self._files),
            "skipped": list(self._skipped),
            "functions": {
                name: [
                    {
                        "name": f.name,
                        "file": f.file,
                        "line": f.line,
                        "end_line": f.end_line,
                        "params": list(f.params),
                        "calls": list(f.calls),
                        "allocs": [
                            {"name": a.name, "line": a.line, "size_expr": a.size_expr}
                            for a in f.allocs
                        ],
                        "buffers": [
                            {"name": b.name, "line": b.line, "size_expr": b.size_expr}
                            for b in f.buffers
                        ],
                        "sources": [
                            {"name": s.name, "line": s.line, "kind": s.kind}
                            for s in f.sources
                        ],
                        "sinks": [
                            {"name": s.name, "line": s.line, "arg": s.arg}
                            for s in f.sinks
                        ],
                    }
                    for f in funcs
                ]
                for name, funcs in self._functions.items()
            },
        }

    # ----- file walking -----

    def _iter_files(self) -> Iterable[Path]:
        if not self._root.exists():
            return
        root = self._root.resolve()
        for dirpath, dirnames, filenames in os.walk(self._root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS)
            for filename in sorted(filenames):
                if filename.startswith("."):
                    continue
                path = Path(dirpath) / filename
                ext = path.suffix.lower()
                if ext not in C_EXTENSIONS and ext not in PY_EXTENSIONS:
                    continue
                if path.is_symlink():
                    self._skipped.append(self._relative(path))
                    continue
                try:
                    resolved = path.resolve()
                except OSError:
                    continue
                if not self._is_within(resolved, root):
                    self._skipped.append(self._relative(path))
                    continue
                try:
                    if resolved.stat().st_size > MAX_FILE_BYTES:
                        self._skipped.append(self._relative(path))
                        continue
                except OSError:
                    continue
                yield path

    def _is_within(self, path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self._root.resolve()))
        except (OSError, ValueError):
            return str(path)

    # ----- parsing -----

    def _parse_file(self, path: Path, rel: str) -> bool:
        try:
            data = path.read_bytes()
        except OSError:
            return False
        ext = path.suffix.lower()
        try:
            if ext in C_EXTENSIONS:
                tree = self._c_parser.parse(data)
                self._collect_c(data, tree.root_node, rel)
            else:
                tree = self._py_parser.parse(data)
                self._collect_python(data, tree.root_node, rel)
        except (RecursionError, Exception):
            self._discard_file(rel)
            return False
        return True

    def _discard_file(self, rel: str) -> None:
        for name in list(self._functions):
            kept = [func for func in self._functions[name] if func.file != rel]
            if kept:
                self._functions[name] = kept
            else:
                del self._functions[name]

    def _add_function(self, func: Function) -> None:
        self._functions.setdefault(func.name, []).append(func)

    def _text(self, data: bytes, node: Node | None) -> str:
        if node is None:
            return ""
        return data[node.start_byte : node.end_byte].decode("utf-8", "replace")

    # ----- C -----

    def _collect_c(self, data: bytes, root: Node, rel: str) -> None:
        for node in self._walk(root):
            if node.type == "function_definition":
                self._collect_c_function(data, node, rel)

    def _collect_c_function(self, data: bytes, node: Node, rel: str) -> None:
        declarator = node.child_by_field_name("declarator")
        name = self._c_declarator_name(declarator)
        if not name:
            return
        params = self._c_params(declarator)
        body = node.child_by_field_name("body")
        func = Function(
            name=name,
            file=rel,
            line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            params=params,
            sources=[
                TaintSource(name=p, line=node.start_point[0] + 1, kind="param")
                for p in params
            ],
        )
        if body is not None:
            self._scan_c_body(data, body, func)
        self._add_function(func)

    def _c_declarator_name(self, node: Node | None) -> str | None:
        current = node
        depth = 0
        while current is not None and depth < MAX_DECLARATOR_DEPTH:
            depth += 1
            node_type = current.type
            if node_type == "identifier":
                return self._text_raw(current)
            if node_type == "array_declarator":
                current = current.child_by_field_name("declarator")
                continue
            if node_type in (
                "pointer_declarator",
                "function_declarator",
                "parenthesized_declarator",
                "init_declarator",
            ):
                inner = current.child_by_field_name("declarator")
                if inner is None:
                    inner = next((child for child in current.children if child.type != "*"), None)
                current = inner
                continue
            return None
        return None

    def _text_raw(self, node: Node) -> str:
        return node.text.decode("utf-8", "replace") if node.text is not None else ""

    def _c_params(self, declarator: Node | None) -> list[str]:
        params: list[str] = []
        if declarator is None:
            return params
        plist = None
        for node in self._walk(declarator):
            if node.type == "parameter_list":
                plist = node
                break
        if plist is None:
            return params
        for param in plist.named_children:
            if param.type != "parameter_declaration":
                continue
            inner = param.child_by_field_name("declarator")
            pname = self._c_declarator_name(inner)
            if pname:
                params.append(pname)
        return params

    def _scan_c_body(self, data: bytes, body: Node, func: Function) -> None:
        for node in self._walk(body):
            if node.type == "call_expression":
                target = self._c_assignment_target(data, node)
                self._handle_c_call(data, node, func, target)
            elif node.type == "declaration":
                self._handle_c_declaration(data, node, func)
            elif node.type == "subscript_expression":
                self._handle_c_subscript(data, node, func)

    def _c_assignment_target(self, data: bytes, call: Node) -> str | None:
        parent = call.parent
        if parent is None:
            return None
        if parent.type == "init_declarator":
            return self._c_declarator_name(parent.child_by_field_name("declarator"))
        if parent.type == "assignment_expression":
            left = parent.child_by_field_name("left")
            return self._text(data, left) if left is not None else None
        return None

    def _handle_c_subscript(self, data: bytes, node: Node, func: Function) -> None:
        arg = node.child_by_field_name("argument")
        if arg is None:
            return
        base = self._text(data, arg)
        if base in func.params and base in ("argv", "envp"):
            line = node.start_point[0] + 1
            key = (base, line)
            if key not in {(s.name, s.line) for s in func.sources}:
                func.sources.append(TaintSource(name=base, line=line, kind="argv"))

    def _callee_name(self, data: bytes, func_node: Node | None) -> str | None:
        if func_node is None:
            return None
        text = self._text(data, func_node)
        if func_node.type == "identifier":
            return text.split(".")[-1].strip()
        return text.strip() or None

    def _arg_nodes(self, args: Node | None) -> list[Node]:
        if args is None:
            return []
        return [node for node in args.named_children if node.type != "comment"]

    def _dest_name(self, text: str) -> str:
        value = text.strip()
        while value.startswith(("&", "*", "(")):
            value = value[1:].strip()
        match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", value)
        return match.group(0) if match else ""

    def _handle_c_call(
        self, data: bytes, node: Node, func: Function, target: str | None = None
    ) -> None:
        fn_node = node.child_by_field_name("function")
        name = self._callee_name(data, fn_node)
        if not name:
            return
        line = node.start_point[0] + 1
        args = node.child_by_field_name("arguments")
        arg_nodes = self._arg_nodes(args)
        if name not in func.calls:
            func.calls.append(name)
        func.call_lines.append((name, line))
        base = name.split("::")[-1]
        if base in ALLOC_NAMES:
            size = None
            if base in ("calloc", "realloc") and len(arg_nodes) >= 2:
                size = self._text(data, arg_nodes[1])
            elif arg_nodes:
                size = self._text(data, arg_nodes[0])
            func.allocs.append(Alloc(name=target or base, line=line, size_expr=size))
        if base in TAINT_SOURCE_NAMES:
            func.sources.append(TaintSource(name=base, line=line, kind=TAINT_SOURCE_NAMES[base]))
            dest_index = TAINT_SOURCE_DEST_ARGS.get(base)
            if dest_index is not None and dest_index < len(arg_nodes):
                dest = self._dest_name(self._text(data, arg_nodes[dest_index]))
                if dest:
                    func.sources.append(TaintSource(name=dest, line=line, kind="dest"))
        if base in COPY_SINK_DEST_ARG0 and arg_nodes:
            dest = self._dest_name(self._text(data, arg_nodes[0]))
            if dest:
                func.sources.append(TaintSource(name=dest, line=line, kind="dest"))
        if base in STRING_SINKS or base in EXEC_SINKS:
            arg = self._text(data, arg_nodes[0]) if arg_nodes else ""
            func.sinks.append(Sink(name=base, line=line, arg=arg))
        elif base in FORMAT_SINKS:
            if base == "fprintf" and len(arg_nodes) >= 2:
                fmt = arg_nodes[1]
                first = arg_nodes[1]
            elif base in ("sprintf", "snprintf"):
                fmt = arg_nodes[1] if len(arg_nodes) >= 2 else (arg_nodes[0] if arg_nodes else None)
                first = fmt
            else:
                fmt = arg_nodes[0] if arg_nodes else None
                first = fmt
            if fmt is not None and fmt.type != "string_literal":
                arg = self._text(data, first)
                func.sinks.append(Sink(name=base, line=line, arg=arg))

    def _handle_c_declaration(self, data: bytes, node: Node, func: Function) -> None:
        for declarator in self._walk(node):
            if declarator.type == "array_declarator":
                name = self._c_declarator_name(declarator.child_by_field_name("declarator"))
                if not name:
                    continue
                size_node = declarator.child_by_field_name("size")
                size = self._text(data, size_node) if size_node is not None else None
                func.buffers.append(
                    BufferSize(name=name, line=declarator.start_point[0] + 1, size_expr=size)
                )

    # ----- Python -----

    def _collect_python(self, data: bytes, root: Node, rel: str) -> None:
        self._collect_py_scope(data, root, rel, 0)

    def _collect_py_scope(self, data: bytes, scope: Node, rel: str, depth: int) -> None:
        if depth > MAX_PY_SCOPE_DEPTH:
            return
        for node in scope.named_children:
            node_type = node.type
            if node_type == "function_definition":
                self._collect_py_function(data, node, rel, depth + 1)
            elif node_type in ("class_definition", "decorated_definition"):
                self._collect_py_scope(data, node, rel, depth + 1)
            elif node_type in PY_SCOPE_CONTAINERS:
                self._collect_py_scope(data, node, rel, depth + 1)

    def _collect_py_function(self, data: bytes, node: Node, rel: str, depth: int) -> None:
        name_node = node.child_by_field_name("name")
        name = self._text(data, name_node) if name_node is not None else None
        if not name:
            return
        params_node = node.child_by_field_name("parameters")
        params = self._py_params(data, params_node)
        body = node.child_by_field_name("body")
        func = Function(
            name=name,
            file=rel,
            line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            params=params,
            sources=[
                TaintSource(name=p, line=node.start_point[0] + 1, kind="param")
                for p in params
            ],
        )
        if body is not None:
            self._scan_py_body(data, body, func)
        self._add_function(func)
        if body is not None:
            self._collect_py_scope(data, body, rel, depth + 1)

    def _py_params(self, data: bytes, node: Node | None) -> list[str]:
        params: list[str] = []
        if node is None:
            return params
        for child in node.named_children:
            if child.type == "identifier":
                params.append(self._text(data, child))
            elif child.type in ("default_parameter", "typed_parameter", "typed_default_parameter"):
                inner = child.child_by_field_name("name") or (
                    child.named_children[0] if child.named_children else None
                )
                if inner is not None:
                    params.append(self._text(data, inner))
            elif child.type in ("list_splat_pattern", "dictionary_splat_pattern"):
                for sub in child.named_children:
                    if sub.type == "identifier":
                        params.append(self._text(data, sub))
        return params

    def _scan_py_body(self, data: bytes, body: Node, func: Function) -> None:
        seen_sources: set[tuple[str, int]] = set()
        for node in self._walk(body):
            if node.type == "call":
                self._handle_py_call(data, node, func, seen_sources)
            elif node.type == "assignment":
                self._handle_py_assignment(data, node, func)
            elif node.type in ("subscript", "attribute"):
                if self._is_assignment_target(node):
                    continue
                self._add_py_source(data, node, self._py_callee_name(data, node), func, seen_sources)

    def _add_py_source(
        self,
        data: bytes,
        node: Node,
        name: str | None,
        func: Function,
        seen: set[tuple[str, int]],
    ) -> None:
        kind = PY_TAINT_SOURCE_NAMES.get(name or "")
        if kind is None:
            return
        line = node.start_point[0] + 1
        key = (name or "", line)
        if key in seen:
            return
        seen.add(key)
        func.sources.append(TaintSource(name=name or "", line=line, kind=kind))

    def _is_assignment_target(self, node: Node) -> bool:
        parent = node.parent
        if parent is None:
            return False
        if parent.type in ("assignment", "augmented_assignment"):
            left = parent.child_by_field_name("left")
            return left is not None and left.id == node.id
        return False

    def _handle_py_call(
        self, data: bytes, node: Node, func: Function, seen: set[tuple[str, int]]
    ) -> None:
        fn_node = node.child_by_field_name("function")
        name = self._py_callee_name(data, fn_node)
        if not name:
            return
        line = node.start_point[0] + 1
        args = node.child_by_field_name("arguments")
        arg_nodes = args.named_children if args is not None else []
        if name not in func.calls:
            func.calls.append(name)
        func.call_lines.append((name, line))
        base = name
        if base in PY_ALLOC_NAMES:
            size = self._text(data, arg_nodes[0]) if arg_nodes else None
            target = self._py_assignment_target(data, node)
            func.allocs.append(Alloc(name=target or base, line=line, size_expr=size))
        self._add_py_source(data, node, name, func, seen)
        if name in PY_EXEC_SINKS:
            arg = self._text(data, arg_nodes[0]) if arg_nodes else ""
            func.sinks.append(Sink(name=name, line=line, arg=arg))

    def _py_assignment_target(self, data: bytes, call: Node) -> str | None:
        parent = call.parent
        if parent is None:
            return None
        if parent.type == "assignment":
            left = parent.child_by_field_name("left")
            if left is not None and left.type == "identifier":
                return self._text(data, left)
        return None

    def _py_callee_name(self, data: bytes, node: Node | None) -> str | None:
        if node is None:
            return None
        if node.type == "identifier":
            return self._text(data, node)
        if node.type == "attribute":
            obj = node.child_by_field_name("object")
            attr = node.child_by_field_name("attribute")
            if obj is not None and attr is not None:
                base = self._py_callee_name(data, obj)
                leaf = self._text(data, attr)
                return f"{base}.{leaf}" if base else leaf
        if node.type == "subscript":
            value = node.child_by_field_name("value")
            return self._py_callee_name(data, value)
        return self._text(data, node)

    def _handle_py_assignment(self, data: bytes, node: Node, func: Function) -> None:
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if left is None or right is None:
            return
        if left.type != "identifier":
            return
        if right.type == "binary_operator":
            op = self._text(data, right.child_by_field_name("operator"))
            if op == "*":
                size_node = right.child_by_field_name("right")
                size = self._text(data, size_node) if size_node is not None else None
                func.buffers.append(
                    BufferSize(
                        name=self._text(data, left),
                        line=node.start_point[0] + 1,
                        size_expr=size,
                    )
                )

    # ----- taint -----

    def _expr_tainted(self, func: Function, expr: str) -> bool:
        source_names = {s.name for s in func.sources}
        argv_names = self._argv_names(func)
        if self._text_has_source(expr, source_names, argv_names):
            return True
        chain = self._assignment_chain(func)
        tainted_vars = set(self._seed_tainted_vars(chain, source_names, argv_names))
        changed = True
        while changed:
            changed = False
            for target, rhs in chain.items():
                if target in tainted_vars:
                    continue
                if self._rhs_references(rhs, tainted_vars):
                    tainted_vars.add(target)
                    changed = True
        return self._rhs_references(expr, tainted_vars)

    def _argv_names(self, func: Function) -> set[str]:
        names: set[str] = set()
        for param in func.params:
            if param:
                names.add(param)
        return names

    def _text_has_source(self, text: str, source_names: set[str], argv_names: set[str]) -> bool:
        if not text:
            return False
        for src in source_names:
            if re.search(rf"(?<![\w.]){re.escape(src)}(?![\w])", text):
                return True
            if src == "sys.argv" and "sys.argv" in text:
                return True
            if src == "os.environ" and "os.environ" in text:
                return True
            if src == "os.getenv" and "os.getenv" in text:
                return True
        for name in argv_names:
            if re.search(rf"(?<![\w.]){re.escape(name)}\b", text):
                return True
        return False

    def _rhs_references(self, text: str, names: set[str]) -> bool:
        for name in names:
            if re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", text):
                return True
        return False

    def _assignment_chain(self, func: Function) -> dict[str, str]:
        chain: dict[str, str] = {}
        key = (func.file, func.line, func.end_line, func.name)
        cached = self._chain_cache.get(key)
        if cached is not None:
            return cached
        path = self._root / func.file
        try:
            data = path.read_bytes()
        except OSError:
            self._chain_cache[key] = chain
            return chain
        is_c = path.suffix.lower() in C_EXTENSIONS
        parser = self._c_parser if is_c else self._py_parser
        tree = parser.parse(data)
        node = self._find_function_node(tree.root_node, func, is_c)
        if node is None:
            self._chain_cache[key] = chain
            return chain
        body = node.child_by_field_name("body")
        if body is None:
            self._chain_cache[key] = chain
            return chain
        if is_c:
            chain = self._c_assignment_chain(data, body)
        else:
            chain = self._py_assignment_chain(data, body)
        self._chain_cache[key] = chain
        return chain

    _chain_cache: dict[tuple[str, int, int, str], dict[str, str]] = {}

    def _find_function_node(self, root: Node, func: Function, is_c: bool) -> Node | None:
        target_line = func.line
        for node in self._walk(root):
            if node.type != "function_definition":
                continue
            if node.start_point[0] + 1 != target_line:
                continue
            if is_c:
                name = self._c_declarator_name(node.child_by_field_name("declarator"))
            else:
                name_node = node.child_by_field_name("name")
                name = self._text_raw(name_node) if name_node is not None else None
            if name == func.name:
                return node
        return None

    def _c_assignment_chain(self, data: bytes, body: Node) -> dict[str, str]:
        chain: dict[str, str] = {}
        for node in self._walk(body):
            if node.type == "init_declarator":
                decl = node.child_by_field_name("declarator")
                value = node.child_by_field_name("value")
                if decl is None or value is None:
                    continue
                name = self._c_declarator_name(decl)
                if not name:
                    continue
                chain[name] = self._text(data, value)
            elif node.type == "assignment_expression":
                left = node.child_by_field_name("left")
                right = node.child_by_field_name("right")
                if left is None or right is None:
                    continue
                name = self._text(data, left).strip()
                if name:
                    chain[name] = self._text(data, right)
        return chain

    def _py_assignment_chain(self, data: bytes, body: Node) -> dict[str, str]:
        chain: dict[str, str] = {}
        for node in self._walk(body):
            if node.type != "assignment":
                continue
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is None or right is None:
                continue
            if left.type == "identifier":
                chain[self._text(data, left)] = self._text(data, right)
            elif left.type == "pattern_list":
                for sub in left.named_children:
                    if sub.type == "identifier":
                        chain[self._text(data, sub)] = self._text(data, right)
            elif left.type in ("attribute", "subscript"):
                chain[self._text(data, left)] = self._text(data, right)
        return chain

    def _seed_tainted_vars(
        self, chain: dict[str, str], source_names: set[str], argv_names: set[str]
    ) -> list[str]:
        seeds: list[str] = []
        for target, rhs in chain.items():
            if self._text_has_source(rhs, source_names, argv_names):
                seeds.append(target)
        return seeds

    # ----- helpers -----

    def _walk(self, node: Node) -> Iterable[Node]:
        stack = [node]
        while stack:
            current = stack.pop()
            yield current
            for i in range(current.child_count - 1, -1, -1):
                stack.append(current.child(i))

    def _sorted_calls(self, name: str) -> list[str]:
        called: set[str] = set()
        for func in self._functions.get(name, []):
            for callee in func.calls:
                called.add(callee)
        return sorted(called)
