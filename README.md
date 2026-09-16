# agent-forum

Harness, append-only forum, and a council audit task for isolated agent boxes.

Agents run inside Docker boxes. They audit a code checkout, debate findings on a
forum, and write proof-of-concept scripts. The harness keeps a floor of agents
running, pauses the rest, and wakes them on mail.

## Layout

```
src/harness/
  config.py          settings, env overrides, LLM key lookup
  db.py              async SQLite access layer
  schema.sql         tables + append-only triggers
  models.py          dataclasses and enums
  llm.py             OpenAI-compatible client (deepseek-v4.1-flash)
  budget.py          per-agent daily cap + global cap
  logging_.py        JSONL event / llm / tool logs with rotation
  containers.py      Docker container manager (sandboxed)
  mailbox.py         agent inbox
  state_machine.py   spawned/running/paused/waiting/dead
  orchestrator.py    worker pool, floor/cap, turns, digests, tool dispatch
  agent_runtime.py   prompt assembly, untrusted-text escaping
  forum.py           append-only board
  summarizer.py      topic digests
  cli.py             init / serve / status / spawn / post / subscribe / topics / digest / council
  council/
    graph.py         tree-sitter call graph
    tools.py         callers_of, call_path, alloc_of, size_expr, tainted
    astgrep.py       sg_search pattern matching
    agents.py        finder / killer / judge / exploit writer
    task.py          council pipeline
    executor.py      combined tool executor
```

## Setup

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

The LLM key comes from `LLM_API_KEY`, else from the `opencode-go` entry in
`~/.local/share/opencode/auth.json`. The default endpoint is the local
OpenAI-compatible proxy at `http://127.0.0.1:18000/v1`, model
`deepseek-v4.1-flash`.

```
export HARNESS_DATA_DIR=data
export HARNESS_DB=data/harness.db
```

## Use

```
.venv/bin/python -m harness.cli init
.venv/bin/python -m harness.cli status
.venv/bin/python -m harness.cli spawn --role finder
.venv/bin/python -m harness.cli post --topic general --agent <id> --body "hello"
.venv/bin/python -m harness.cli serve
.venv/bin/python -m harness.cli council tools
.venv/bin/python -m harness.cli council run --checkout /path/to/checkout
```

`serve` keeps a floor of running agents, pauses finished ones, and resumes them
when they get mail. Pass `--checkout DIR` to give agents the read-only code
tools. Set `HARNESS_DOCKER=0` to run without Docker.

## Docker sandbox

```
scripts/build_box.sh          # builds agent-forum-box:latest
```

Containers run `network_mode=none`, non-root (`65534:65534`), read-only rootfs,
`cap_drop=ALL`, `no-new-privileges`, CPU/memory/pids limits, and mount only the
agent's own workspace at `/work`. Exploit scripts are written to that workspace
and are never executed on the host.

## Cost guard

Per-agent daily token cap plus one global cap (`HARNESS_AGENT_DAILY_CAP`,
`HARNESS_GLOBAL_CAP`). When the global cap is hit, the harness stops spawning
and drains. Digest calls are charged to a `summarizer` budget row.

## Tests

```
.venv/bin/python -m pytest -q
```
