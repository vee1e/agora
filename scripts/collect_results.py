"""Collect council run artifacts into a single JSON for the report."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

RUNS = ["authd", "netstack", "logparse", "pyservice"]
LABELS = {
    "authd": ("authd", "C authentication daemon"),
    "netstack": ("netstack", "C packet handler"),
    "logparse": ("logparse", "C log line parser"),
    "pyservice": ("pyservice", "Python job service"),
}


def _rows(db: Path, sql: str, params: tuple = ()) -> list[dict]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def collect(run_root: Path) -> dict:
    runs = []
    totals = {
        "candidates": 0,
        "survived": 0,
        "killed": 0,
        "exploited": 0,
        "poc_attempts": 0,
        "poc_succeeded": 0,
        "tokens_global": 0,
        "llm_calls": 0,
        "tool_calls": 0,
    }
    for name in RUNS:
        db = run_root / name / "harness.db"
        if not db.exists():
            continue
        findings = _rows(
            db,
            "SELECT id,file,line,sink,call_path,status,detail,created_at FROM findings ORDER BY file,line",
        )
        for f in findings:
            try:
                f["call_path"] = json.loads(f["call_path"] or "[]")
            except (TypeError, ValueError):
                f["call_path"] = []
            f["poc_runs"] = _rows(
                db,
                "SELECT attempt,exit_code,success,stdout,stderr,created_at FROM poc_runs WHERE finding_id=? ORDER BY attempt",
                (f["id"],),
            )
            for r in f["poc_runs"]:
                r["success"] = bool(r["success"])
                r["stdout"] = (r["stdout"] or "")[:4000]
                r["stderr"] = (r["stderr"] or "")[:2000]
        agents = _rows(db, "SELECT id,role,state FROM agents")
        budget = _rows(db, "SELECT scope,key,tokens_used,cap FROM budget")
        posts = _rows(db, "SELECT id,agent_id,topic_id,body FROM posts ORDER BY id")
        status = {}
        for f in findings:
            status[f["status"]] = status.get(f["status"], 0) + 1
        llm_calls = 0
        tool_calls = 0
        log_dir = run_root / name / "logs"
        if (log_dir / "llm.jsonl").exists():
            llm_calls = sum(1 for _ in (log_dir / "llm.jsonl").open())
        if (log_dir / "tools.jsonl").exists():
            tool_calls = sum(1 for _ in (log_dir / "tools.jsonl").open())
        global_tokens = next(
            (b["tokens_used"] for b in budget if b["scope"] == "global"), 0
        )
        attempts = sum(len(f["poc_runs"]) for f in findings)
        succeeded = sum(1 for f in findings for r in f["poc_runs"] if r["success"])
        runs.append(
            {
                "name": name,
                "label": LABELS.get(name, (name, name))[1],
                "findings": findings,
                "agents": agents,
                "budget": budget,
                "posts": posts,
                "status_counts": status,
                "llm_calls": llm_calls,
                "tool_calls": tool_calls,
                "global_tokens": global_tokens,
                "poc_attempts": attempts,
                "poc_succeeded": succeeded,
            }
        )
        totals["candidates"] += len(findings)
        totals["survived"] += status.get("survived", 0)
        totals["killed"] += status.get("killed", 0)
        totals["exploited"] += status.get("exploited", 0)
        totals["poc_attempts"] += attempts
        totals["poc_succeeded"] += succeeded
        totals["tokens_global"] += global_tokens
        totals["llm_calls"] += llm_calls
        totals["tool_calls"] += tool_calls
    return {"runs": runs, "totals": totals}


def main() -> int:
    run_root = Path(sys.argv[1]).expanduser()
    out = Path(sys.argv[2])
    data = collect(run_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    print(f"wrote {out} ({len(data['runs'])} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
