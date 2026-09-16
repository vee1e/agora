"""Combined tool executor for council runs.

Wraps a :class:`CouncilTools` call graph and an :class:`AstGrep` search over the
same checkout behind one ``specs``/``dispatch`` surface. Read-only; no shell.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .astgrep import AstGrep
from .graph import CodeGraph
from .tools import CouncilTools


def _tool_name(spec: Any) -> str | None:
    if not isinstance(spec, dict):
        return None
    function = spec.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if isinstance(name, str) and name:
        return name
    return None


class CombinedTools:
    """Union of council graph tools and structural search, one dispatch table."""

    def __init__(self, council: CouncilTools, astgrep: AstGrep) -> None:
        self._council = council
        self._astgrep = astgrep
        self._handlers: dict[str, Callable[[dict], dict]] = {}
        specs: list[dict] = []
        for backend in (council, astgrep):
            for spec in backend.specs():
                name = _tool_name(spec)
                if name is None or name in self._handlers:
                    continue
                self._handlers[name] = backend.dispatch
                specs.append(spec)
        self._specs = specs

    @property
    def graph(self) -> CodeGraph | None:
        graph = getattr(self._council, "_graph", None)
        if isinstance(graph, CodeGraph):
            return graph
        return None

    def specs(self) -> list[dict]:
        return list(self._specs)

    def dispatch(self, name: str, arguments: dict) -> dict:
        handler = self._handlers.get(name)
        if handler is None:
            return {"ok": False, "error": f"unknown tool: {name}"}
        try:
            return handler(name, arguments)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def build_combined_tools(checkout: str | Path) -> CombinedTools:
    root = Path(checkout)
    graph = CodeGraph(root)
    graph.build()
    return CombinedTools(CouncilTools(graph), AstGrep(root))
