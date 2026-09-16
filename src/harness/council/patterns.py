"""Static pattern detectors for bug classes that have no traditional sink.

Read-only over a checkout. Tree-sitter only. Never imports or executes any
analysed file. Emits one :class:`Signal` per occurrence with the enclosing
function name (``"<module>"`` at file scope).

C-only kinds (``integer_overflow_alloc``, ``unchecked_alloc``, ``double_free``,
``use_after_free``, ``realloc_loses_original``, ``format_string``,
``size_truncation``, ``off_by_one``, ``missing_nul_termination``,
``snprintf_offset_write``, ``dest_scratch_write``, ``signal_unsafe``) are not
emitted for Python because Python has no manual free/realloc, no fixed-width
stack buffers, no integer wraparound on ints, no NUL-terminated buffers, and no
non-literal format strings.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tree_sitter import Language, Node, Parser

import tree_sitter_c
import tree_sitter_python

from .graph import (
    C_EXTENSIONS,
    DEFAULT_MAX_FILES,
    MAX_FILE_BYTES,
    PY_EXTENSIONS,
    SKIP_DIRS,
)

MODULE_FUNCTION = "<module>"
HIGH = "high"
LOW = "low"

ALLOC_FUNCS = {"malloc", "calloc", "realloc", "xmalloc"}
ALLOC_RESULT_FUNCS = {"malloc", "calloc", "realloc", "strdup"}
IO_RETURN_FUNCS = {"read", "recv", "recvfrom", "fread", "getline", "fgets"}
IO_RETURN_PY = {
    "os.read",
    "os.pread",
    "recv",
    "recvfrom",
    "socket.recv",
    "socket.recvfrom",
}
COPY_FUNCS = {"memcpy", "memmove", "strcpy", "strncpy", "strcat"}
SIZE_FUNCS = {
    "memcpy",
    "memmove",
    "memset",
    "read",
    "write",
    "recv",
    "recvfrom",
    "send",
    "malloc",
    "calloc",
    "realloc",
    "alloca",
    "strncpy",
    "strncat",
    "snprintf",
    "fread",
    "fwrite",
}
CHECK_PATH_FUNCS = {"access", "stat", "lstat", "faccessat"}
OPEN_PATH_FUNCS = {"open", "fopen", "unlink", "rename", "chmod", "chown"}
PY_CHECK_PATH_FUNCS = {
    "os.access",
    "os.stat",
    "os.lstat",
    "os.path.exists",
    "os.path.isfile",
}
PY_OPEN_PATH_FUNCS = {
    "os.open",
    "os.unlink",
    "os.rename",
    "os.chmod",
    "os.chown",
    "open",
}
PY_SHELL_OS_FUNCS = {"os.system", "os.popen"}
PY_SHELL_SUBPROCESS_FUNCS = {
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
}
UNSAFE_FUNCS = {
    "malloc",
    "calloc",
    "realloc",
    "free",
    "printf",
    "sprintf",
    "strcpy",
    "strlen",
    "syslog",
    "exit",
}
TEMPFILE_FUNCS = {"mktemp", "tmpnam", "tempnam"}
TRUNC_TYPE_TOKENS = {"int", "short", "char"}
TRUNC_FUNC_VALUES = {"strlen"}

COMPARISON_OPS = {"<", "<=", ">", ">=", "==", "!="}
LIMIT_MACROS = {"SIZE_MAX", "INT_MAX", "UINT_MAX", "LLONG_MAX", "PATH_MAX"}
GUARD_EXIT_FUNCS = {"exit", "abort", "err", "errx", "die", "_exit"}
CLAMP_FUNCS = {"MIN", "MAX", "min", "max"}
MASK_OPS = {"&=", "%="}
CALL_TYPES = ("call_expression", "call")
_SKIP_ARG_TYPES = {"comment"}

FORMAT_INDEX: dict[str, int] = {
    "printf": 0,
    "fprintf": 1,
    "syslog": 1,
    "snprintf": 2,
    "fscanf": 1,
}
FORMAT_FUNCS = set(FORMAT_INDEX)


@dataclass(frozen=True)
class Signal:
    kind: str
    file: str
    line: int
    function: str
    detail: str
    cwe: str
    confidence: str


@dataclass
class _Scope:
    data: bytes
    rel: str
    func: str
    is_c: bool
    params: list[str]
    nodes: list[Node]
    func_node: Node | None = None

    @property
    def lang(self) -> str:
        return "c" if self.is_c else "py"


@dataclass
class _FunctionRecord:
    file: str
    name: str
    unsafe: list[tuple[int, str]]


class PatternScanner:
    """Heuristic bug-pattern scan over one checkout."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._c_lang = Language(tree_sitter_c.language())
        self._py_lang = Language(tree_sitter_python.language())
        self._c_parser = Parser(self._c_lang)
        self._py_parser = Parser(self._py_lang)

    def scan(self) -> list[Signal]:
        signals: list[Signal] = []
        handler_names: set[str] = set()
        records: list[_FunctionRecord] = []
        for path in self._iter_files():
            try:
                data = path.read_bytes()
            except OSError:
                continue
            rel = self._relative(path)
            try:
                scopes, is_c = self._build_scopes(data, rel)
            except RecursionError:
                continue
            except Exception:
                continue
            for scope in scopes:
                signals.extend(self._scan_scope(scope))
            if is_c:
                handler_names |= self._signal_handlers(scopes)
                for scope in scopes:
                    if scope.func == MODULE_FUNCTION:
                        continue
                    unsafe = self._unsafe_calls(scope)
                    if unsafe:
                        records.append(_FunctionRecord(scope.rel, scope.func, unsafe))
        for record in records:
            if record.name not in handler_names:
                continue
            for line, callee in record.unsafe:
                signals.append(
                    self._signal(
                        record.file,
                        "signal_unsafe",
                        line,
                        record.name,
                        f"signal handler {record.name} calls {callee}",
                        "CWE-364",
                        HIGH,
                    )
                )
        return self._dedupe(signals)


    def _iter_files(self) -> Iterable[Path]:
        if not self._root.exists():
            return
        root = self._root.resolve()
        count = 0
        for dirpath, dirnames, filenames in os.walk(self._root):
            dirnames[:] = sorted(
                d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS
            )
            for filename in sorted(filenames):
                if filename.startswith("."):
                    continue
                path = Path(dirpath) / filename
                ext = path.suffix.lower()
                if ext not in C_EXTENSIONS and ext not in PY_EXTENSIONS:
                    continue
                if path.is_symlink():
                    continue
                try:
                    resolved = path.resolve()
                except OSError:
                    continue
                if not self._is_within(resolved, root):
                    continue
                try:
                    if resolved.stat().st_size > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                if count >= DEFAULT_MAX_FILES:
                    continue
                count += 1
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


    def _build_scopes(self, data: bytes, rel: str) -> tuple[list[_Scope], bool]:
        is_c = not rel.lower().endswith(".py")
        parser = self._c_parser if is_c else self._py_parser
        root = parser.parse(data).root_node
        module = _Scope(data, rel, MODULE_FUNCTION, is_c, [], [], root)
        scopes: list[_Scope] = [module]
        by_id: dict[int, _Scope] = {}
        for node in self._walk(root):
            if node.type != "function_definition":
                continue
            if is_c:
                name = self._c_function_name(node)
                params = self._c_params(node)
            else:
                name_node = node.child_by_field_name("name")
                name = self._text(data, name_node) if name_node is not None else ""
                params = self._py_params(data, node)
            if not name:
                continue
            scope = _Scope(data, rel, name, is_c, params, [], node)
            scopes.append(scope)
            by_id[node.id] = scope
        stack: list[tuple[Node, _Scope]] = [(root, module)]
        while stack:
            node, scope = stack.pop()
            owner = scope
            if node.type == "function_definition" and node.id in by_id:
                owner = by_id[node.id]
            owner.nodes.append(node)
            for index in range(node.child_count - 1, -1, -1):
                stack.append((node.child(index), owner))
        for scope in scopes:
            scope.nodes.sort(key=lambda item: (item.start_byte, -item.end_byte))
        return scopes, is_c

    def _c_function_name(self, node: Node) -> str:
        current = node.child_by_field_name("declarator")
        while current is not None:
            if current.type == "identifier":
                return self._text_raw(current)
            if current.type == "array_declarator":
                current = current.child_by_field_name("declarator")
                continue
            if current.type in (
                "pointer_declarator",
                "function_declarator",
                "parenthesized_declarator",
                "init_declarator",
            ):
                inner = current.child_by_field_name("declarator")
                if inner is None:
                    inner = next(
                        (child for child in current.children if child.type != "*"),
                        None,
                    )
                current = inner
                continue
            return ""
        return ""

    def _c_params(self, node: Node) -> list[str]:
        params: list[str] = []
        declarator = node.child_by_field_name("declarator")
        if declarator is None:
            return params
        plist = None
        for sub in self._walk(declarator):
            if sub.type == "parameter_list":
                plist = sub
                break
        if plist is None:
            return params
        for param in plist.named_children:
            if param.type != "parameter_declaration":
                continue
            name = self._c_function_name(param)
            if name:
                params.append(name)
        return params

    def _py_params(self, data: bytes, node: Node) -> list[str]:
        params: list[str] = []
        node_params = node.child_by_field_name("parameters")
        if node_params is None:
            return params
        for child in node_params.named_children:
            if child.type == "identifier":
                params.append(self._text(data, child))
                continue
            if child.type in ("typed_parameter", "default_parameter",
                              "typed_default_parameter", "list_splat_pattern",
                              "dictionary_splat_pattern"):
                inner = child.child_by_field_name("name")
                if inner is None:
                    inner = child.named_children[0] if child.named_children else None
                if inner is not None:
                    text = self._text(data, inner)
                    if text:
                        params.append(text)
        return params

    def _scan_scope(self, scope: _Scope) -> list[Signal]:
        out: list[Signal] = []
        if scope.is_c:
            out.extend(self._integer_overflow_alloc(scope))
            out.extend(self._unchecked_alloc(scope))
        out.extend(self._unchecked_io_return(scope))
        if scope.is_c:
            out.extend(self._free_use(scope))
            out.extend(self._realloc_loses_original(scope))
        out.extend(self._toctou(scope))
        out.extend(self._insecure_temp_file(scope))
        if not scope.is_c:
            out.extend(self._shell_injection(scope))
        if scope.is_c:
            out.extend(self._format_string(scope))
            out.extend(self._size_truncation(scope))
            out.extend(self._off_by_one(scope))
            out.extend(self._missing_nul_termination(scope))
            out.extend(self._snprintf_offset_write(scope))
            out.extend(self._dest_scratch_write(scope))
        out.extend(self._unchecked_retry_loop(scope))
        return out


    def _integer_overflow_alloc(self, scope: _Scope) -> list[Signal]:
        out: list[Signal] = []
        for call in self._calls(scope):
            leaf = self._leaf(self._callee(scope, call))
            if leaf not in ALLOC_FUNCS:
                continue
            args = self._args(scope, call)
            line = call.start_point[0] + 1
            if leaf == "calloc":
                if len(args) >= 2 and (
                    args[0].type != "number_literal"
                    or args[1].type != "number_literal"
                ):
                    operands = set(
                        self._words(self._text(scope, args[0]))
                        + self._words(self._text(scope, args[1]))
                    )
                    if not self._operands_guarded(scope, operands):
                        out.append(
                            self._signal(
                                scope.rel,
                                "integer_overflow_alloc",
                                line,
                                scope.func,
                                f"calloc({self._text(scope, args[0])}, "
                                f"{self._text(scope, args[1])}) multiplies "
                                f"non-literal factors with no range check",
                                "CWE-190",
                                HIGH,
                            )
                        )
                continue
            if leaf == "realloc":
                size_node = args[1] if len(args) >= 2 else None
            else:
                size_node = args[0] if args else None
            if size_node is None or size_node.type != "binary_expression":
                continue
            op = self._text(scope, size_node.child_by_field_name("operator"))
            if op not in ("*", "+"):
                continue
            left = size_node.child_by_field_name("left")
            right = size_node.child_by_field_name("right")
            if left is None or right is None:
                continue
            if left.type == "number_literal" and right.type == "number_literal":
                continue
            operands = set(
                self._words(self._text(scope, left))
                + self._words(self._text(scope, right))
            )
            if not operands or self._operands_guarded(scope, operands):
                continue
            out.append(
                self._signal(
                    scope.rel,
                    "integer_overflow_alloc",
                    line,
                    scope.func,
                    f"{leaf}({self._text(scope, size_node)}) size uses "
                    f"{', '.join(sorted(operands))} with no range check",
                    "CWE-190",
                    HIGH,
                )
            )
        return out

    def _operands_guarded(self, scope: _Scope, operands: set[str]) -> bool:
        node = scope.func_node
        if node is None:
            return False
        return any(
            self._guarded(node, scope.data, name, scope.lang)
            for name in operands
            if name
        )

    def _guarded(
        self,
        func_node: Node,
        data: bytes,
        operand: str,
        lang: str,
        exclude: set[int] | None = None,
    ) -> bool:
        operand = self._base(operand)
        if not operand:
            return False
        names = self._alias_group(func_node, data, operand, lang)
        if self._guard_present(func_node, data, names, lang, exclude):
            return True
        return self._one_hop_guard(func_node, data, names, lang)

    def _guard_present(
        self,
        body: Node,
        data: bytes,
        names: set[str],
        lang: str,
        exclude: set[int] | None,
    ) -> bool:
        names = {name for name in names if name}
        if not names:
            return False
        for _owner, cond in self._condition_nodes(body, lang):
            if exclude and cond.id in exclude:
                continue
            if self._condition_guards(cond, data, names, lang):
                return True
        if self._clamp_or_mask_present(body, data, names, lang):
            return True
        if self._early_exit_present(body, data, names, lang):
            return True
        return False

    def _condition_nodes(self, body: Node, lang: str) -> list[tuple[Node, Node]]:
        fields = {
            "if_statement": "condition",
            "while_statement": "condition",
            "conditional_expression": "condition",
        }
        if lang == "c":
            fields["do_statement"] = "condition"
        out: list[tuple[Node, Node]] = []
        for node in self._walk(body):
            field = fields.get(node.type)
            if field is None:
                continue
            cond = node.child_by_field_name(field)
            if cond is not None:
                out.append((node, cond))
        return out

    def _condition_guards(
        self, cond: Node, data: bytes, names: set[str], lang: str
    ) -> bool:
        for left, _op, right in self._comparisons_in(cond, data, lang):
            left_text = self._text(data, left)
            right_text = self._text(data, right)
            if (
                self._refers_to_any(left_text, names)
                and not self._refers_to_any(right_text, names)
                and self._is_limit(data, right)
            ):
                return True
            if (
                self._refers_to_any(right_text, names)
                and not self._refers_to_any(left_text, names)
                and self._is_limit(data, left)
            ):
                return True
        return False

    def _comparisons_in(
        self, node: Node, data: bytes, lang: str
    ) -> list[tuple[Node, str, Node]]:
        out: list[tuple[Node, str, Node]] = []
        for sub in self._walk(node):
            if lang == "c":
                if sub.type != "binary_expression":
                    continue
                op = self._text(data, sub.child_by_field_name("operator")).strip()
                if op not in COMPARISON_OPS:
                    continue
                left = sub.child_by_field_name("left")
                right = sub.child_by_field_name("right")
                if left is not None and right is not None:
                    out.append((left, op, right))
            else:
                if sub.type != "comparison_operator":
                    continue
                operands = list(sub.named_children)
                ops = [child for child in sub.children if child.type in COMPARISON_OPS]
                for index, op in enumerate(ops):
                    if index + 1 >= len(operands):
                        break
                    out.append(
                        (
                            operands[index],
                            self._text(data, op).strip(),
                            operands[index + 1],
                        )
                    )
        return out

    def _is_limit(self, data: bytes, node: Node) -> bool:
        if node is None:
            return False
        text = self._text(data, node).strip()
        if not text:
            return False
        if node.type in ("number_literal", "integer", "float"):
            return True
        if node.type == "sizeof_expression" or text.startswith("sizeof"):
            return True
        if text in LIMIT_MACROS or re.fullmatch(r"MAX_\w+", text):
            return True
        if re.search(r"\d", text):
            return True
        return bool(self._words(text))

    def _clamp_or_mask_present(
        self, body: Node, data: bytes, names: set[str], lang: str
    ) -> bool:
        for node in self._walk(body):
            if node.type in CALL_TYPES:
                callee = self._callee_data(data, node)
                if self._leaf(callee) in CLAMP_FUNCS:
                    for arg in self._args_data(data, node):
                        if self._refers_to_any(self._text(data, arg), names):
                            return True
            if node.type == "if_statement":
                cond = node.child_by_field_name("condition")
                if cond is not None and self._refers_to_any(
                    self._text(data, cond), names
                ):
                    cons = node.child_by_field_name("consequence")
                    if cons is not None and self._assigns_name(cons, data, names, lang):
                        return True
            if lang == "c" and node.type == "assignment_expression":
                op = self._text(
                    data, node.child_by_field_name("operator")
                ).strip()
                left = node.child_by_field_name("left")
                if (
                    op in MASK_OPS
                    and left is not None
                    and self._refers_to_any(self._text(data, left), names)
                ):
                    return True
                if op == "=" and self._masked_value(
                    node.child_by_field_name("right"), data, names, lang
                ):
                    return True
            if lang == "py" and node.type == "augmented_assignment":
                op = self._text(
                    data, node.child_by_field_name("operator")
                ).strip()
                left = node.child_by_field_name("left")
                if (
                    op in MASK_OPS
                    and left is not None
                    and self._refers_to_any(self._text(data, left), names)
                ):
                    return True
        return False

    def _masked_value(
        self, node: Node | None, data: bytes, names: set[str], lang: str
    ) -> bool:
        if node is None:
            return False
        if node.type != "binary_expression":
            return False
        op = self._text(data, node.child_by_field_name("operator")).strip()
        if op not in ("&", "%"):
            return False
        left = node.child_by_field_name("left")
        return left is not None and self._refers_to_any(self._text(data, left), names)

    def _assigns_name(
        self, body: Node, data: bytes, names: set[str], lang: str
    ) -> bool:
        types = ("assignment_expression",) if lang == "c" else (
            "assignment",
            "augmented_assignment",
        )
        for node in self._walk(body):
            if node.type not in types:
                continue
            left = node.child_by_field_name("left")
            if left is not None and left.type == "identifier" and self._refers_to_any(
                self._text(data, left), names
            ):
                return True
        return False

    def _early_exit_present(
        self, body: Node, data: bytes, names: set[str], lang: str
    ) -> bool:
        for node in self._walk(body):
            if node.type != "if_statement":
                continue
            cond = node.child_by_field_name("condition")
            if cond is None or not self._refers_to_any(self._text(data, cond), names):
                continue
            cons = node.child_by_field_name("consequence")
            if cons is not None and self._contains_exit(cons, data, lang):
                return True
        return False

    def _contains_exit(self, node: Node, data: bytes, lang: str) -> bool:
        for sub in self._walk(node):
            if sub.type in (
                "return_statement",
                "break_statement",
                "continue_statement",
                "goto_statement",
            ):
                return True
            if sub.type in CALL_TYPES:
                callee = self._callee_data(data, sub)
                if self._leaf(callee) in GUARD_EXIT_FUNCS or callee in (
                    "sys.exit",
                    "os._exit",
                ):
                    return True
        return False

    def _one_hop_guard(
        self, func_node: Node, data: bytes, names: set[str], lang: str
    ) -> bool:
        functions = self._file_functions(func_node, data, lang)
        if not functions:
            return False
        for call in self._walk(func_node):
            if call.type not in CALL_TYPES:
                continue
            callee = self._callee_data(data, call)
            key = self._leaf(callee) if lang == "c" else callee
            target = next(
                (entry for entry in functions if entry[1] and entry[1] == key), None
            )
            if target is None:
                continue
            _fnode, _name, params, body = target
            if body is None:
                continue
            for index, arg in enumerate(self._args_data(data, call)):
                if index >= len(params):
                    break
                if not self._refers_to_any(self._text(data, arg), names):
                    continue
                param = self._base(params[index])
                if param and self._guard_present(body, data, {param}, lang, None):
                    return True
        return False

    def _file_functions(
        self, func_node: Node, data: bytes, lang: str
    ) -> list[tuple[Node, str, list[str], Node | None]]:
        root = func_node
        while root.parent is not None:
            root = root.parent
        out: list[tuple[Node, str, list[str], Node | None]] = []
        for node in self._walk(root):
            if node.type != "function_definition":
                continue
            if lang == "c":
                name = self._c_function_name(node)
                params = self._c_params(node)
            else:
                name_node = node.child_by_field_name("name")
                name = self._text(data, name_node) if name_node is not None else ""
                params = self._py_params(data, node)
            body = node.child_by_field_name("body")
            out.append((node, name, params, body))
        return out

    def _alias_group(
        self, func_node: Node, data: bytes, operand: str, lang: str
    ) -> set[str]:
        parent: dict[str, str] = {}

        def find(name: str) -> str:
            root = name
            while parent.get(root, root) != root:
                root = parent[root]
            while parent.get(name, name) != name:
                parent[name], name = root, parent[name]
            return root

        def union(first: str, second: str) -> None:
            if first == second:
                return
            root_a, root_b = find(first), find(second)
            if root_a != root_b:
                parent[root_a] = root_b

        for node in self._walk(func_node):
            if lang == "c":
                if node.type == "init_declarator":
                    decl = node.child_by_field_name("declarator")
                    val = node.child_by_field_name("value")
                    if val is None or val.type != "identifier":
                        continue
                    left = self._base(self._text(data, decl))
                    right = self._base(self._text(data, val))
                    if left and right:
                        union(left, right)
                elif node.type == "assignment_expression":
                    if self._text(
                        data, node.child_by_field_name("operator")
                    ).strip() != "=":
                        continue
                    first = node.child_by_field_name("left")
                    second = node.child_by_field_name("right")
                    if (
                        first is not None
                        and first.type == "identifier"
                        and second is not None
                        and second.type == "identifier"
                    ):
                        union(
                            self._base(self._text(data, first)),
                            self._base(self._text(data, second)),
                        )
            elif node.type == "assignment":
                first = node.child_by_field_name("left")
                second = node.child_by_field_name("right")
                if (
                    first is not None
                    and first.type == "identifier"
                    and second is not None
                    and second.type == "identifier"
                ):
                    union(
                        self._base(self._text(data, first)),
                        self._base(self._text(data, second)),
                    )
        names = {operand}
        target = find(operand)
        for name in parent:
            if find(name) == target:
                names.add(name)
        return names

    def _refers_to_any(self, text: str, names: set[str]) -> bool:
        return any(self._references(text, name) for name in names)


    def _unchecked_alloc(self, scope: _Scope) -> list[Signal]:
        out: list[Signal] = []
        for call in self._calls(scope):
            leaf = self._leaf(self._callee(scope, call))
            if leaf not in ALLOC_RESULT_FUNCS:
                continue
            target = self._assignment_target(scope, call)
            if target is None:
                if self._is_call_argument(call):
                    out.append(
                        self._signal(
                            scope.rel,
                            "unchecked_alloc",
                            call.start_point[0] + 1,
                            scope.func,
                            f"{leaf} result passed directly to a call with no "
                            f"NULL check",
                            "CWE-252",
                            HIGH,
                        )
                    )
                continue
            alloc_line = call.start_point[0] + 1
            deref = self._first_deref(scope, target, call.end_byte)
            if deref is None:
                continue
            check_line, check_byte = self._first_null_check(
                scope, target, call.end_byte
            )
            if check_byte is not None and check_byte < deref[1]:
                continue
            out.append(
                self._signal(
                    scope.rel,
                    "unchecked_alloc",
                    alloc_line,
                    scope.func,
                    f"{leaf} result {target} dereferenced at line {deref[0]} with "
                    f"no NULL check",
                    "CWE-252",
                    HIGH,
                )
            )
        return out

    def _first_null_check(
        self, scope: _Scope, name: str, after_byte: int
    ) -> tuple[int, int | None]:
        best_line: int | None = None
        best_byte: int | None = None
        for node in scope.nodes:
            if node.start_byte <= after_byte:
                continue
            if node.type == "if_statement":
                candidate = node.child_by_field_name("condition")
                if not self._is_null_check(scope, candidate, name):
                    continue
            elif node.type == "conditional_expression":
                candidate = node.child_by_field_name("condition")
                if not self._is_null_check(scope, candidate, name):
                    continue
            elif node.type == "binary_expression":
                if not self._is_null_check(scope, node, name):
                    continue
            elif node.type == "call_expression":
                if not self._is_assert_check(scope, node, name):
                    continue
            else:
                continue
            line = node.start_point[0] + 1
            if best_line is None or line < best_line:
                best_line = line
                best_byte = node.start_byte
        return best_line or 0, best_byte

    def _is_assert_check(self, scope: _Scope, node: Node, name: str) -> bool:
        if self._leaf(self._callee(scope, node)) != "assert":
            return False
        return any(
            self._references(self._text(scope, arg), name)
            for arg in self._args(scope, node)
        )

    def _is_null_check(self, scope: _Scope, node: Node | None, name: str) -> bool:
        if node is None:
            return False
        if node.type == "parenthesized_expression":
            inner = node.named_children[0] if node.named_children else None
            return self._is_null_check(scope, inner, name)
        if node.type == "identifier":
            return self._same_var(self._text(scope, node), name)
        if node.type == "unary_expression":
            return self._text(scope, node).strip() == f"!{name}"
        if node.type != "binary_expression":
            return False
        op = self._text(scope, node.child_by_field_name("operator"))
        if op not in ("==", "!="):
            return False
        left = self._text(scope, node.child_by_field_name("left"))
        right = self._text(scope, node.child_by_field_name("right"))
        return (
            self._same_var(left, name) and right.strip() in ("NULL", "0")
        ) or (
            self._same_var(right, name) and left.strip() in ("NULL", "0")
        )

    def _same_var(self, text: str, name: str) -> bool:
        return bool(re.fullmatch(rf"\s*{re.escape(name)}\s*", text))


    def _unchecked_io_return(self, scope: _Scope) -> list[Signal]:
        out: list[Signal] = []
        for call in self._calls(scope):
            callee = self._callee(scope, call)
            leaf = self._leaf(callee)
            if scope.is_c:
                if leaf not in IO_RETURN_FUNCS:
                    continue
            elif callee not in IO_RETURN_PY and leaf not in {
                "fread",
                "getline",
                "fgets",
            }:
                continue
            target = self._assignment_target(scope, call)
            if target is None:
                continue
            if self._expression_compared(scope, target):
                continue
            out.append(
                self._signal(
                    scope.rel,
                    "unchecked_io_return",
                    call.start_point[0] + 1,
                    scope.func,
                    f"{callee} result assigned to {target} and never compared",
                    "CWE-252",
                    HIGH,
                )
            )
        return out

    def _expression_compared(self, scope: _Scope, name: str) -> bool:
        for node in scope.nodes:
            if node.type == "comparison_operator":
                if self._references(self._text(scope, node), name):
                    return True
                continue
            if node.type in CALL_TYPES:
                callee = self._callee(scope, node)
                if self._leaf(callee) in {"perror", "err", "errx", "die"} and any(
                    self._references(self._text(scope, arg), name)
                    for arg in self._args(scope, node)
                ):
                    return True
                continue
            if node.type != "binary_expression":
                continue
            op = self._text(scope, node.child_by_field_name("operator"))
            if op not in ("<", "<=", ">", ">=", "==", "!="):
                continue
            left = self._text(scope, node.child_by_field_name("left"))
            right = self._text(scope, node.child_by_field_name("right"))
            if self._references(left, name) or self._references(right, name):
                return True
        return False

    def _references(self, text: str, name: str) -> bool:
        return bool(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text))


    def _free_use(self, scope: _Scope) -> list[Signal]:
        out: list[Signal] = []
        events: list[tuple[int, int, str, str]] = []
        for call in self._calls(scope):
            if self._leaf(self._callee(scope, call)) != "free":
                continue
            args = self._args(scope, call)
            if not args:
                continue
            name = self._lvalue(self._text(scope, args[0]))
            if name:
                events.append((call.start_point[0] + 1, call.start_byte, "free", name))
        for node in scope.nodes:
            if node.type == "assignment_expression":
                left = self._text(scope, node.child_by_field_name("left"))
                name = self._base(left)
                if name and self._same_var(left, name):
                    events.append(
                        (node.start_point[0] + 1, node.start_byte, "assign", name)
                    )
            elif node.type == "init_declarator":
                name = self._base(
                    self._text(scope, node.child_by_field_name("declarator"))
                )
                if name:
                    events.append(
                        (node.start_point[0] + 1, node.start_byte, "assign", name)
                    )
        for node in scope.nodes:
            if node.type not in (
                "subscript_expression",
                "unary_expression",
                "field_expression",
                "call_expression",
            ):
                continue
            if node.type == "call_expression" and self._leaf(
                self._callee(scope, node)
            ) == "free":
                continue
            name = self._deref_base(scope, node)
            if name:
                events.append((node.start_point[0] + 1, node.start_byte, "deref", name))
        events.sort()
        last_free: dict[str, int] = {}
        for line, _byte, kind, name in events:
            if kind == "free":
                if name in last_free:
                    out.append(
                        self._signal(
                            scope.rel,
                            "double_free",
                            line,
                            scope.func,
                            f"free({name}) called again after free on line "
                            f"{last_free[name]} with no reassignment",
                            "CWE-415",
                            HIGH,
                        )
                    )
                last_free[name] = line
            elif kind == "assign":
                last_free.pop(name, None)
            elif kind == "deref" and name in last_free:
                out.append(
                    self._signal(
                        scope.rel,
                        "use_after_free",
                        line,
                        scope.func,
                        f"{name} dereferenced after free on line "
                        f"{last_free[name]} with no reassignment",
                        "CWE-416",
                        HIGH,
                    )
                )
                last_free.pop(name, None)
        return out

    def _first_deref(
        self, scope: _Scope, name: str, after_byte: int
    ) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        for node in scope.nodes:
            if node.start_byte <= after_byte:
                continue
            if node.type not in (
                "subscript_expression",
                "unary_expression",
                "field_expression",
                "call_expression",
            ):
                continue
            if node.type == "call_expression" and self._leaf(
                self._callee(scope, node)
            ) in ("free", "realloc", "assert"):
                continue
            if self._deref_base(scope, node) == name:
                line = node.start_point[0] + 1
                if best is None or line < best[0]:
                    best = (line, node.start_byte)
        return best

    def _deref_base(self, scope: _Scope, node: Node) -> str:
        data = scope.data
        if node.type == "subscript_expression":
            base = node.child_by_field_name("argument")
            if base is None:
                base = node.child_by_field_name("value")
            return self._base(self._text(data, base))
        if node.type == "unary_expression":
            op = self._text(data, node.child_by_field_name("operator"))
            if op != "*":
                return ""
            return self._base(self._text(data, node.child_by_field_name("argument")))
        if node.type == "field_expression":
            return self._base(self._text(data, node.child_by_field_name("argument")))
        if node.type == "call_expression":
            for arg in self._args(scope, node):
                text = self._text(data, arg)
                base = self._base(text)
                if base and self._same_var(text, base):
                    return base
        return ""


    def _realloc_loses_original(self, scope: _Scope) -> list[Signal]:
        out: list[Signal] = []
        for call in self._calls(scope):
            if self._leaf(self._callee(scope, call)) != "realloc":
                continue
            args = self._args(scope, call)
            if not args:
                continue
            source = self._base(self._text(scope, args[0]))
            target = self._assignment_target(scope, call)
            if source and target and source == target:
                out.append(
                    self._signal(
                        scope.rel,
                        "realloc_loses_original",
                        call.start_point[0] + 1,
                        scope.func,
                        f"{target} = realloc({source}, ...) overwrites the "
                        f"original pointer",
                        "CWE-401",
                        HIGH,
                    )
                )
        return out


    def _toctou(self, scope: _Scope) -> list[Signal]:
        checks: list[tuple[int, int, str, str]] = []
        opens: list[tuple[int, int, str, str]] = []
        for call in self._calls(scope):
            callee = self._callee(scope, call)
            leaf = self._leaf(callee)
            args = self._args(scope, call)
            if not args:
                continue
            line = call.start_point[0] + 1
            if scope.is_c:
                if leaf in CHECK_PATH_FUNCS:
                    index = 1 if leaf == "faccessat" else 0
                    if index < len(args):
                        path = self._text(scope, args[index]).strip()
                        if path:
                            checks.append((call.start_byte, call.end_byte, path,
                                           callee))
                if leaf in OPEN_PATH_FUNCS:
                    path = self._text(scope, args[0]).strip()
                    if path:
                        opens.append((call.start_byte, line, path, callee))
            else:
                if callee in PY_CHECK_PATH_FUNCS:
                    path = self._text(scope, args[0]).strip()
                    if path:
                        checks.append((call.start_byte, call.end_byte, path, callee))
                if callee in PY_OPEN_PATH_FUNCS:
                    path = self._text(scope, args[0]).strip()
                    if path:
                        opens.append((call.start_byte, line, path, callee))
        out: list[Signal] = []
        seen: set[tuple[str, int]] = set()
        for check_start, check_end, path, check_name in checks:
            for open_start, open_line, open_path, open_name in opens:
                if open_start <= check_end or open_path != path:
                    continue
                key = (path, open_start)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    self._signal(
                        scope.rel,
                        "toctou",
                        open_line,
                        scope.func,
                        f"{check_name}({path}) checked then {open_name}({open_path}) "
                        f"reopens the same path",
                        "CWE-367",
                        HIGH,
                    )
                )
        return out


    def _insecure_temp_file(self, scope: _Scope) -> list[Signal]:
        out: list[Signal] = []
        builders: dict[str, int] = {}
        for call in self._calls(scope):
            callee = self._callee(scope, call)
            leaf = self._leaf(callee)
            if leaf in TEMPFILE_FUNCS or callee in ("tempfile.mktemp",):
                out.append(
                    self._signal(
                        scope.rel,
                        "insecure_temp_file",
                        call.start_point[0] + 1,
                        scope.func,
                        f"{callee} creates a predictable temporary name",
                        "CWE-377",
                        HIGH,
                    )
                )
                continue
            if leaf not in ("sprintf", "snprintf", "strcpy", "strcat"):
                continue
            args = self._args(scope, call)
            if not args:
                continue
            joined = self._text(scope, call)
            if "/tmp/" not in joined and "TMPDIR" not in joined:
                continue
            dest = self._base(self._text(scope, args[0]))
            if dest:
                builders[dest] = call.start_point[0] + 1
        for call in self._calls(scope):
            leaf = self._leaf(self._callee(scope, call))
            if leaf not in ("open", "fopen", "unlink", "rename", "chmod"):
                continue
            args = self._args(scope, call)
            if not args:
                continue
            dest = self._base(self._text(scope, args[0]))
            if dest and dest in builders:
                out.append(
                    self._signal(
                        scope.rel,
                        "insecure_temp_file",
                        call.start_point[0] + 1,
                        scope.func,
                        f"{leaf}({dest}) opens a path built from /tmp in {dest} "
                        f"at line {builders[dest]}",
                        "CWE-377",
                        HIGH,
                    )
                )
        return out

    def _shell_injection(self, scope: _Scope) -> list[Signal]:
        out: list[Signal] = []
        for call in self._calls(scope):
            callee = self._callee(scope, call)
            leaf = self._leaf(callee)
            args = self._args(scope, call)
            if not args:
                continue
            if callee in PY_SHELL_OS_FUNCS or ("." not in callee and leaf in (
                "system",
                "popen",
            )):
                if self._py_nonliteral_string(scope, args[0]):
                    out.append(self._shell_signal(scope, call, callee))
                continue
            if callee.startswith("os.exec"):
                if self._py_nonliteral_string(scope, args[0]):
                    out.append(self._shell_signal(scope, call, callee))
                continue
            if callee in PY_SHELL_SUBPROCESS_FUNCS:
                command = args[0]
                if self._py_nonliteral_string(scope, command):
                    out.append(self._shell_signal(scope, call, callee))
                    continue
                if self._has_shell_true(scope, call) and not self._py_literal_list(
                    command
                ):
                    out.append(self._shell_signal(scope, call, callee))
        return out

    def _shell_signal(self, scope: _Scope, call: Node, callee: str) -> Signal:
        return self._signal(
            scope.rel,
            "shell_injection",
            call.start_point[0] + 1,
            scope.func,
            f"{callee} builds a command from a non-literal expression",
            "CWE-78",
            HIGH,
        )

    def _has_shell_true(self, scope: _Scope, call: Node) -> bool:
        arguments = call.child_by_field_name("arguments")
        if arguments is None:
            return False
        for child in arguments.named_children:
            if child.type != "keyword_argument":
                continue
            name = self._text(scope, child.child_by_field_name("name"))
            value = child.child_by_field_name("value")
            if (
                name == "shell"
                and value is not None
                and self._text(scope, value).strip() in ("True", "1")
            ):
                return True
        return False

    def _py_literal_list(self, node: Node) -> bool:
        return node.type in ("list", "tuple", "set")

    def _py_nonliteral_string(self, scope: _Scope, node: Node | None) -> bool:
        if node is None:
            return False
        if node.type == "identifier":
            return True
        if node.type == "string":
            return any(child.type == "interpolation" for child in node.named_children)
        if node.type == "concatenated_string":
            return any(
                self._py_nonliteral_string(scope, child)
                for child in node.named_children
                if child.type == "string"
            )
        if node.type == "binary_operator":
            op = self._text(scope, node.child_by_field_name("operator")).strip()
            return op in ("+", "%")
        if node.type in CALL_TYPES:
            return self._leaf(self._callee(scope, node)) in ("format", "join")
        return False


    def _format_string(self, scope: _Scope) -> list[Signal]:
        out: list[Signal] = []
        for call in self._calls(scope):
            leaf = self._leaf(self._callee(scope, call))
            if leaf not in FORMAT_FUNCS:
                continue
            index = FORMAT_INDEX[leaf]
            args = self._args(scope, call)
            if len(args) <= index:
                continue
            fmt = args[index]
            if self._is_string_literal(fmt):
                continue
            out.append(
                self._signal(
                    scope.rel,
                    "format_string",
                    call.start_point[0] + 1,
                    scope.func,
                    f"{leaf} format argument {self._text(scope, fmt)} is not a "
                    f"string literal",
                    "CWE-134",
                    HIGH,
                )
            )
        return out

    def _is_string_literal(self, node: Node) -> bool:
        if node.type == "string_literal":
            return True
        if node.type == "parenthesized_expression":
            inner = node.named_children[0] if node.named_children else None
            return inner is not None and self._is_string_literal(inner)
        return False


    def _size_truncation(self, scope: _Scope) -> list[Signal]:
        out: list[Signal] = []
        for node in scope.nodes:
            if node.type != "declaration":
                continue
            for init in self._walk(node):
                if init.type != "init_declarator":
                    continue
                declarator = init.child_by_field_name("declarator")
                name = self._base(self._text(scope, declarator))
                if not name:
                    continue
                decl_text = self._text(scope, node)
                cut = decl_text.find(name)
                type_text = decl_text[:cut].strip() if cut > 0 else ""
                if not self._is_truncating_type(type_text):
                    continue
                value = init.child_by_field_name("value")
                if value is None or not self._is_size_value(scope, value):
                    continue
                if not self._used_as_size(scope, name, node.end_byte):
                    continue
                out.append(
                    self._signal(
                        scope.rel,
                        "size_truncation",
                        node.start_point[0] + 1,
                        scope.func,
                        f"{type_text} {name} holds {self._text(scope, value)} and "
                        f"is used as a size or index",
                        "CWE-197",
                        HIGH,
                    )
                )
        return out

    def _is_truncating_type(self, type_text: str) -> bool:
        if "size_t" in type_text or "long" in type_text:
            return False
        words = self._words(type_text)
        if not words:
            return False
        allowed = TRUNC_TYPE_TOKENS | {"unsigned", "signed", "const"}
        if not all(word in allowed for word in words):
            return False
        return any(word in TRUNC_TYPE_TOKENS for word in words)

    def _is_size_value(self, scope: _Scope, value: Node) -> bool:
        if value.type == "sizeof_expression":
            return True
        if value.type == "call_expression" and self._leaf(
            self._callee(scope, value)
        ) in TRUNC_FUNC_VALUES:
            return True
        return "size_t" in self._text(scope, value)

    def _used_as_size(self, scope: _Scope, name: str, after_byte: int) -> bool:
        for node in scope.nodes:
            if node.start_byte < after_byte:
                continue
            if node.type == "subscript_expression":
                index = node.child_by_field_name("index")
                if self._references(self._text(scope, index), name):
                    return True
            elif node.type == "call_expression":
                leaf = self._leaf(self._callee(scope, node))
                if leaf not in SIZE_FUNCS:
                    continue
                for arg in self._args(scope, node):
                    if self._references(self._text(scope, arg), name):
                        return True
        return False


    def _off_by_one(self, scope: _Scope) -> list[Signal]:
        out: list[Signal] = []
        dims = self._array_dims(scope)
        for node in scope.nodes:
            if node.type != "for_statement":
                continue
            condition = node.child_by_field_name("condition")
            if condition is None or condition.type != "binary_expression":
                continue
            op = self._text(scope, condition.child_by_field_name("operator"))
            if op != "<=":
                continue
            left = condition.child_by_field_name("left")
            right = condition.child_by_field_name("right")
            if left is None or right is None or left.type != "identifier":
                continue
            loop_var = self._text(scope, left)
            bound = self._text(scope, right).strip()
            if not bound:
                continue
            body = node.child_by_field_name("body")
            if body is None:
                continue
            for sub in self._walk(body):
                if sub.type != "subscript_expression":
                    continue
                index = sub.child_by_field_name("index")
                if self._text(scope, index).strip() != loop_var:
                    continue
                base = self._base(
                    self._text(scope, sub.child_by_field_name("argument"))
                )
                dim = dims.get(base)
                if dim is None or dim.strip() != bound:
                    continue
                if scope.func_node is not None and self._guarded(
                    scope.func_node, scope.data, loop_var, "c"
                ):
                    continue
                out.append(
                    self._signal(
                        scope.rel,
                        "off_by_one",
                        node.start_point[0] + 1,
                        scope.func,
                        f"loop {loop_var} <= {bound} indexes {base}[{loop_var}] "
                        f"declared {base}[{dim}]",
                        "CWE-193",
                        HIGH,
                    )
                )
        return out

    def _array_dims(self, scope: _Scope) -> dict[str, str]:
        dims: dict[str, str] = {}
        for node in scope.nodes:
            if node.type != "array_declarator":
                continue
            name = self._base(
                self._text(scope, node.child_by_field_name("declarator"))
            )
            size = node.child_by_field_name("size")
            if name and size is not None:
                dims[name] = self._text(scope, size)
        return dims


    def _signal_handlers(self, scopes: list[_Scope]) -> set[str]:
        handlers: set[str] = set()
        for scope in scopes:
            for call in self._calls(scope):
                leaf = self._leaf(self._callee(scope, call))
                if leaf == "signal":
                    args = self._args(scope, call)
                    if len(args) >= 2:
                        name = self._base(self._text(scope, args[1]))
                        if name:
                            handlers.add(name)
                elif leaf == "sigaction":
                    text = self._text(scope, call)
                    for match in re.finditer(
                        r"sa_handler\s*=\s*&?\s*([A-Za-z_]\w*)", text
                    ):
                        handlers.add(match.group(1))
        return handlers

    def _unsafe_calls(self, scope: _Scope) -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        for call in self._calls(scope):
            leaf = self._leaf(self._callee(scope, call))
            if leaf in UNSAFE_FUNCS:
                out.append((call.start_point[0] + 1, leaf))
        return out


    def _missing_nul_termination(self, scope: _Scope) -> list[Signal]:
        out: list[Signal] = []
        for call in self._calls(scope):
            if self._leaf(self._callee(scope, call)) != "strncpy":
                continue
            args = self._args(scope, call)
            if not args:
                continue
            dest = self._base(self._text(scope, args[0]))
            if not dest or self._has_nul_termination(scope, dest):
                continue
            out.append(
                self._signal(
                    scope.rel,
                    "missing_nul_termination",
                    call.start_point[0] + 1,
                    scope.func,
                    f"strncpy({dest}, ...) has no terminating NUL assignment in "
                    f"the function",
                    "CWE-170",
                    HIGH,
                )
            )
        return out

    def _has_nul_termination(self, scope: _Scope, name: str) -> bool:
        nul = chr(92) + "0"
        for node in scope.nodes:
            if node.type != "assignment_expression":
                continue
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is None or right is None or left.type != "subscript_expression":
                continue
            base = self._base(self._text(scope, left.child_by_field_name("argument")))
            if base != name:
                continue
            right_text = self._text(scope, right).strip()
            if right.type == "char_literal" and nul in right_text:
                return True
            if right.type == "number_literal" and right_text in ("0", "0x0"):
                return True
        return False


    def _snprintf_offset_write(self, scope: _Scope) -> list[Signal]:
        out: list[Signal] = []
        returns: list[tuple[str, int]] = []
        for call in self._calls(scope):
            if self._leaf(self._callee(scope, call)) not in ("snprintf", "sprintf"):
                continue
            target = self._assignment_target(scope, call)
            if target:
                returns.append((target, call.end_byte))
        for name, after_byte in returns:
            if scope.func_node is not None and self._guarded(
                scope.func_node, scope.data, name, "c"
            ):
                continue
            for node in scope.nodes:
                if node.start_byte <= after_byte:
                    continue
                if node.type == "assignment_expression":
                    left = node.child_by_field_name("left")
                    if left is None or left.type != "subscript_expression":
                        continue
                    index = self._text(scope, left.child_by_field_name("index"))
                    if self._references(index, name):
                        out.append(
                            self._signal(
                                scope.rel,
                                "snprintf_offset_write",
                                node.start_point[0] + 1,
                                scope.func,
                                f"snprintf return {name} used as subscript "
                                f"{self._text(scope, left)}",
                                "CWE-787",
                                HIGH,
                            )
                        )
                elif node.type == "call_expression":
                    for arg in self._args(scope, node):
                        text = self._text(scope, arg)
                        pattern = (
                            rf"\b{re.escape(name)}\b\s*[-+]"
                            rf"|[-+]\s*\b{re.escape(name)}\b"
                        )
                        if re.search(pattern, text):
                            out.append(
                                self._signal(
                                    scope.rel,
                                    "snprintf_offset_write",
                                    node.start_point[0] + 1,
                                    scope.func,
                                    f"snprintf return {name} used as an offset in "
                                    f"{self._text(scope, node)}",
                                    "CWE-787",
                                    HIGH,
                                )
                            )
                            break
        return out


    def _dest_scratch_write(self, scope: _Scope) -> list[Signal]:
        out: list[Signal] = []
        for call in self._calls(scope):
            leaf = self._leaf(self._callee(scope, call))
            if leaf not in COPY_FUNCS and leaf not in ("sprintf", "snprintf"):
                continue
            args = self._args(scope, call)
            if not args:
                continue
            if leaf in COPY_FUNCS and len(args) >= 2:
                dest_text = self._normalize(self._text(scope, args[0]))
                src_text = self._normalize(self._text(scope, args[1]))
                if dest_text and dest_text == src_text:
                    out.append(
                        self._signal(
                            scope.rel,
                            "dest_scratch_write",
                            call.start_point[0] + 1,
                            scope.func,
                            f"{leaf}({self._text(scope, args[0])}, "
                            f"{self._text(scope, args[1])}, ...) source and "
                            f"destination are the same expression",
                            "CWE-787",
                            HIGH,
                        )
                    )
                    continue
            dest = self._text(scope, args[0])
            base = self._base(dest)
            if not base or base not in scope.params:
                continue
            raw = dest.strip()
            if not re.match(rf"&?\s*{re.escape(base)}\s*(\[|\+)", raw):
                continue
            names = {word for word in self._words(dest)}
            if scope.func_node is not None and any(
                self._guarded(scope.func_node, scope.data, name, "c")
                for name in names
            ):
                continue
            out.append(
                self._signal(
                    scope.rel,
                    "dest_scratch_write",
                    call.start_point[0] + 1,
                    scope.func,
                    f"{leaf} writes caller buffer {base} at {raw} with no clamp "
                    f"on the offset",
                    "CWE-787",
                    HIGH,
                )
            )
        return out

    def _unchecked_retry_loop(self, scope: _Scope) -> list[Signal]:
        out: list[Signal] = []
        loop_types = ("while_statement", "for_statement", "do_statement")
        call_type = "call_expression" if scope.is_c else "call"
        assign_type = "assignment_expression" if scope.is_c else "assignment"
        for node in scope.nodes:
            if node.type not in loop_types:
                continue
            condition = node.child_by_field_name("condition")
            body = node.child_by_field_name("body")
            if condition is None or body is None:
                continue
            ids = set(self._words(self._text(scope, condition)))
            if not ids:
                continue
            body_nodes = list(self._walk(body))
            assignment = None
            for sub in body_nodes:
                if sub.type != assign_type:
                    continue
                left = sub.child_by_field_name("left")
                right = sub.child_by_field_name("right")
                if left is None or right is None or left.type != "identifier":
                    continue
                if scope.is_c:
                    op = self._text(scope, sub.child_by_field_name("operator"))
                    if op != "=":
                        continue
                name = self._text(scope, left)
                if name not in ids:
                    continue
                if right.type == call_type:
                    continue
                assignment = sub
                break
            if assignment is None:
                continue
            retried: Node | None = None
            for sub in body_nodes:
                if sub.start_byte >= assignment.start_byte:
                    break
                if sub.type == call_type:
                    retried = sub
            if retried is None:
                continue
            guarded_names = set(ids)
            call_target = self._assignment_target(scope, retried)
            if call_target:
                guarded_names.add(call_target)
            if scope.func_node is not None and any(
                self._guarded(
                    scope.func_node,
                    scope.data,
                    name,
                    scope.lang,
                    {condition.id},
                )
                for name in guarded_names
            ):
                continue
            out.append(
                self._signal(
                    scope.rel,
                    "unchecked_retry_loop",
                    retried.start_point[0] + 1,
                    scope.func,
                    f"retry loop calls {self._callee(scope, retried)} then "
                    f"updates {self._text(scope, assignment)} without "
                    f"re-checking the condition",
                    "CWE-362",
                    LOW,
                )
            )
        return out


    def _calls(self, scope: _Scope) -> list[Node]:
        call_type = "call_expression" if scope.is_c else "call"
        return [node for node in scope.nodes if node.type == call_type]

    def _callee(self, scope: _Scope, call: Node) -> str:
        return self._callee_data(scope.data, call)

    def _callee_data(self, data: bytes, call: Node) -> str:
        function = call.child_by_field_name("function")
        return self._text(data, function)

    def _args(self, scope: _Scope, call: Node) -> list[Node]:
        return self._args_data(scope.data, call)

    def _args_data(self, data: bytes, call: Node) -> list[Node]:
        arguments = call.child_by_field_name("arguments")
        if arguments is None:
            return []
        out: list[Node] = []
        for child in arguments.named_children:
            if child.type in _SKIP_ARG_TYPES:
                continue
            if child.type == "keyword_argument":
                value = child.child_by_field_name("value")
                if value is not None:
                    out.append(value)
                continue
            out.append(child)
        return out

    def _assignment_target(self, scope: _Scope, call: Node) -> str | None:
        parent = call.parent
        if parent is None:
            return None
        if scope.is_c:
            if parent.type == "init_declarator":
                return self._base(
                    self._text(scope, parent.child_by_field_name("declarator"))
                )
            if parent.type == "assignment_expression":
                left = parent.child_by_field_name("left")
                return self._base(self._text(scope, left))
            return None
        if parent.type == "assignment":
            left = parent.child_by_field_name("left")
            if left is not None and left.type == "identifier":
                return self._text(scope, left)
        return None

    def _is_call_argument(self, call: Node) -> bool:
        parent = call.parent
        while parent is not None:
            if parent.type in ("argument_list", "arguments"):
                return True
            if parent.type in (
                "init_declarator",
                "assignment_expression",
                "assignment",
                "expression_statement",
                "return_statement",
            ):
                return False
            parent = parent.parent
        return False

    def _leaf(self, name: str) -> str:
        value = name.strip()
        if "::" in value:
            value = value.split("::")[-1]
        if "." in value:
            value = value.split(".")[-1]
        return value.strip()

    def _base(self, text: str) -> str:
        value = text.strip()
        while value.startswith(("&", "*", "(", "!")):
            value = value[1:].strip()
        match = re.match(r"[A-Za-z_]\w*", value)
        return match.group(0) if match else ""

    def _lvalue(self, text: str) -> str:
        value = text.strip()
        while value.startswith("&"):
            value = value[1:].strip()
        return re.sub(r"\s+", "", value)

    def _words(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z_]\w*", text)

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", "", text)

    def _text(self, source: bytes | _Scope, node: Node | None) -> str:
        if node is None:
            return ""
        data = source.data if isinstance(source, _Scope) else source
        return data[node.start_byte : node.end_byte].decode("utf-8", "replace")

    def _text_raw(self, node: Node) -> str:
        if node.text is None:
            return ""
        return node.text.decode("utf-8", "replace")

    def _walk(self, node: Node) -> Iterable[Node]:
        stack = [node]
        while stack:
            current = stack.pop()
            yield current
            for index in range(current.child_count - 1, -1, -1):
                stack.append(current.child(index))

    def _signal(
        self,
        file: str,
        kind: str,
        line: int,
        function: str,
        detail: str,
        cwe: str,
        confidence: str,
    ) -> Signal:
        return Signal(
            kind=kind,
            file=file,
            line=line,
            function=function,
            detail=detail,
            cwe=cwe,
            confidence=confidence,
        )

    def _dedupe(self, signals: list[Signal]) -> list[Signal]:
        seen: set[tuple[str, str, int, str, str, str, str]] = set()
        out: list[Signal] = []
        for signal in signals:
            key = (
                signal.kind,
                signal.file,
                signal.line,
                signal.function,
                signal.detail,
                signal.cwe,
                signal.confidence,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(signal)
        out.sort(
            key=lambda item: (
                item.file,
                item.line,
                item.kind,
                item.function,
                item.detail,
            )
        )
        return out
