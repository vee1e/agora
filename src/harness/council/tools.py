"""OpenAI-style tool wrappers around :class:`CodeGraph`.

Every method returns a JSON-serializable dict. `dispatch` validates input and
never raises out to the caller.
"""
from __future__ import annotations

from typing import Any, Callable

from .graph import Alloc, BufferSize, CodeGraph, Function, Sink, TaintSource
from .patterns import Signal

MAX_RESULT_ITEMS = 200


def _cap(
    items: list[Any], serialize: Callable[[Any], dict[str, Any]]
) -> tuple[list[dict[str, Any]], int, bool]:
    total = len(items)
    capped = [serialize(item) for item in items[:MAX_RESULT_ITEMS]]
    return capped, total, total > MAX_RESULT_ITEMS


def _function_brief(func: Function, line: int | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": func.name,
        "file": func.file,
        "line": func.line,
        "end_line": func.end_line,
    }
    if line is not None:
        data["call_line"] = line
    return data


def _alloc_dict(alloc: Alloc) -> dict[str, Any]:
    return {"name": alloc.name, "line": alloc.line, "size_expr": alloc.size_expr}


def _buffer_dict(buffer: BufferSize) -> dict[str, Any]:
    return {
        "name": buffer.name,
        "line": buffer.line,
        "size_expr": buffer.size_expr,
        "kind": buffer.kind,
    }


def _source_dict(source: TaintSource) -> dict[str, Any]:
    return {"name": source.name, "line": source.line, "kind": source.kind}


def _sink_dict(sink: Sink) -> dict[str, Any]:
    return {"name": sink.name, "line": sink.line, "arg": sink.arg}


def _signal_dict(signal: Signal) -> dict[str, Any]:
    return {
        "kind": signal.kind,
        "file": signal.file,
        "line": signal.line,
        "function": signal.function,
        "detail": signal.detail,
        "cwe": signal.cwe,
        "confidence": signal.confidence,
    }


class CouncilTools:
    """Tool surface the council agents call. Read-only over the checkout."""

    def __init__(self, graph: CodeGraph) -> None:
        self._graph = graph

    def specs(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "callers_of",
                    "description": "List functions that call the named function, with the call line.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fn": {"type": "string", "description": "Function name to look up."}
                        },
                        "required": ["fn"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "call_path",
                    "description": "Find call chains from a source function to a sink function.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "description": "Start function name."},
                            "sink": {"type": "string", "description": "End function name."},
                        },
                        "required": ["source", "sink"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "alloc_of",
                    "description": "List heap allocations, optionally filtered by pointer name and function.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ptr": {"type": "string", "description": "Pointer or variable name substring."},
                            "fn": {"type": "string", "description": "Optional function name to scope the search."},
                        },
                        "required": ["ptr"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "size_expr",
                    "description": "Show the size expression of a buffer, optionally scoped to one function.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "buf": {"type": "string", "description": "Buffer name substring."},
                            "fn": {"type": "string", "description": "Optional function name to scope the search."},
                        },
                        "required": ["buf"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tainted",
                    "description": "Report whether an expression comes from untrusted input in a function.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expr": {"type": "string", "description": "Expression or variable to check."},
                            "fn": {"type": "string", "description": "Optional function name to scope the check."},
                        },
                        "required": ["expr"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "patterns",
                    "description": (
                        "Static signals for bug classes without a traditional sink: integer "
                        "overflow feeding allocation, TOCTOU, double free, use after free, "
                        "unchecked returns, insecure temp files, format strings, size "
                        "truncation, off by one, signal-unsafe calls, missing NUL "
                        "termination, snprintf offset writes, destination-as-scratch "
                        "writes, and retry loops that drop a precondition."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "description": "Optional signal kind filter."},
                            "path": {"type": "string", "description": "Optional file path substring filter."},
                            "limit": {"type": "integer", "description": "Optional cap on returned signals."},
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def dispatch(self, name: str, arguments: dict) -> dict:
        if not isinstance(name, str):
            return {"ok": False, "error": f"unknown tool: {name!r}"}
        handlers: dict[str, Callable[[dict], dict]] = {
            "callers_of": self._dispatch_callers_of,
            "call_path": self._dispatch_call_path,
            "alloc_of": self._dispatch_alloc_of,
            "size_expr": self._dispatch_size_expr,
            "tainted": self._dispatch_tainted,
            "patterns": self._dispatch_patterns,
        }
        if not isinstance(arguments, dict):
            return {"ok": False, "error": "arguments must be an object"}
        try:
            handler = handlers.get(name)
            if handler is None:
                return {"ok": False, "error": f"unknown tool: {name}"}
            return {"ok": True, "result": handler(arguments)}
        except _ToolError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _require_str(self, args: dict, key: str) -> str:
        value = args.get(key)
        if not isinstance(value, str) or not value.strip():
            raise _ToolError(f"missing or invalid '{key}' (expected non-empty string)")
        return value.strip()

    def _optional_str(self, args: dict, key: str) -> str | None:
        value = args.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise _ToolError(f"invalid '{key}' (expected non-empty string)")
        return value.strip()

    def _optional_int(self, args: dict, key: str) -> int | None:
        value = args.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise _ToolError(f"invalid '{key}' (expected positive integer)")
        return value

    def _dispatch_callers_of(self, args: dict) -> dict:
        fn = self._require_str(args, "fn")
        return self.callers_of(fn)

    def _dispatch_call_path(self, args: dict) -> dict:
        source = self._require_str(args, "source")
        sink = self._require_str(args, "sink")
        return self.call_path(source, sink)

    def _dispatch_alloc_of(self, args: dict) -> dict:
        ptr = self._require_str(args, "ptr")
        fn = self._optional_str(args, "fn")
        return self.alloc_of(ptr, fn)

    def _dispatch_size_expr(self, args: dict) -> dict:
        buf = self._require_str(args, "buf")
        fn = self._optional_str(args, "fn")
        return self.size_expr(buf, fn)

    def _dispatch_tainted(self, args: dict) -> dict:
        expr = self._require_str(args, "expr")
        fn = self._optional_str(args, "fn")
        return self.tainted(expr, fn)

    def _dispatch_patterns(self, args: dict) -> dict:
        kind = self._optional_str(args, "kind")
        path = self._optional_str(args, "path")
        limit = self._optional_int(args, "limit")
        return self.patterns(kind, path, limit)

    # ----- tool implementations -----

    def callers_of(self, fn: str) -> dict:
        entries = self._graph.callers_of(fn)
        callers, total, truncated = _cap(entries, lambda pair: _function_brief(pair[0], pair[1]))
        return {
            "fn": fn,
            "count": total,
            "truncated": truncated,
            "callers": callers,
        }

    def call_path(self, source: str, sink: str) -> dict:
        paths = self._graph.call_path(source, sink)
        return {
            "source": source,
            "sink": sink,
            "count": len(paths),
            "truncated": len(paths) > MAX_RESULT_ITEMS,
            "paths": paths[:MAX_RESULT_ITEMS],
        }

    def alloc_of(self, ptr: str, fn: str | None = None) -> dict:
        if fn is not None:
            allocs = self._graph.alloc_of(fn, ptr)
            capped, total, truncated = _cap(allocs, _alloc_dict)
            return {
                "ptr": ptr,
                "fn": fn,
                "count": total,
                "truncated": truncated,
                "allocs": capped,
            }
        results: list[dict[str, Any]] = []
        for name, funcs in self._graph.functions.items():
            for func in funcs:
                for alloc in func.allocs:
                    if ptr in alloc.name:
                        entry = _alloc_dict(alloc)
                        entry["fn"] = func.name
                        entry["file"] = func.file
                        results.append(entry)
        results.sort(key=lambda item: (item["file"], item["line"], item["name"]))
        capped, total, truncated = _cap(results, lambda item: item)
        return {
            "ptr": ptr,
            "fn": None,
            "count": total,
            "truncated": truncated,
            "allocs": capped,
        }

    def size_expr(self, buf: str, fn: str | None = None) -> dict:
        if fn is not None:
            buffers = self._graph.size_expr(fn, buf)
            capped, total, truncated = _cap(buffers, _buffer_dict)
            return {
                "buf": buf,
                "fn": fn,
                "count": total,
                "truncated": truncated,
                "buffers": capped,
            }
        results: list[dict[str, Any]] = []
        for _name, funcs in self._graph.functions.items():
            for func in funcs:
                for buffer in func.buffers:
                    if buf in buffer.name:
                        entry = _buffer_dict(buffer)
                        entry["fn"] = func.name
                        entry["file"] = func.file
                        results.append(entry)
                for alloc in func.allocs:
                    if buf in alloc.name:
                        entry = _buffer_dict(
                            BufferSize(
                                name=alloc.name,
                                line=alloc.line,
                                size_expr=alloc.size_expr,
                                kind="heap",
                            )
                        )
                        entry["fn"] = func.name
                        entry["file"] = func.file
                        results.append(entry)
        results.sort(key=lambda item: (item["file"], item["line"], item["name"]))
        capped, total, truncated = _cap(results, lambda item: item)
        return {
            "buf": buf,
            "fn": None,
            "count": total,
            "truncated": truncated,
            "buffers": capped,
        }

    def tainted(self, expr: str, fn: str | None = None) -> dict:
        if fn is not None:
            result = self._graph.tainted(fn, expr)
            return {
                "expr": expr,
                "fn": fn,
                "tainted": result,
                "functions": [fn] if self._graph.functions.get(fn) else [],
            }
        hits: list[str] = []
        for name, funcs in self._graph.functions.items():
            for func in funcs:
                if self._graph.tainted(func.name, expr):
                    hits.append(func.name)
                    break
        hits.sort()
        return {
            "expr": expr,
            "fn": None,
            "tainted": bool(hits),
            "count": len(hits),
            "truncated": len(hits) > MAX_RESULT_ITEMS,
            "functions": hits[:MAX_RESULT_ITEMS],
        }

    def patterns(
        self,
        kind: str | None = None,
        path: str | None = None,
        limit: int | None = None,
    ) -> dict:
        signals = self._graph.signals_for(file=path, kind=kind)
        cap = MAX_RESULT_ITEMS
        if limit is not None and limit < cap:
            cap = limit
        total = len(signals)
        capped = [_signal_dict(signal) for signal in signals[:cap]]
        return {
            "kind": kind,
            "path": path,
            "count": total,
            "truncated": total > cap,
            "signals": capped,
        }


class _ToolError(Exception):
    """Raised inside dispatch for bad arguments; converted to an error dict."""
