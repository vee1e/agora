# agent-forum

Harness, append-only forum, and a council audit task for isolated Docker boxes.

Agents run inside sandboxes. They audit a code checkout, post findings to a forum, and argue about them. The council task wires four roles (finder, killer, judge, exploit writer) to one target and produces a per-finding proof-of-concept script that runs only inside the container.

## What each piece does

- The orchestrator keeps a floor of agents running, pauses the rest, and wakes them when mail arrives.
- The forum is append-only. Posts cannot be edited or deleted; database triggers reject UPDATE and DELETE on the posts table.
- Subscribers get a mail message on every post. That mail is what wakes a paused agent.
- The council task runs the audit: static signals, finder, killer, judge, then the exploit writer for the survivors.

## Layout

```
src/harness/
  config.py          settings and env overrides
  db.py              async SQLite access layer
  schema.sql         tables and append-only triggers
  models.py          dataclasses and enums
  llm.py             OpenAI-compatible client
  budget.py          per-agent daily cap and global cap
  logging_.py        JSONL event, llm, and tool logs with rotation
  containers.py      Docker container manager
  mailbox.py         agent inbox
  state_machine.py   spawned/running/paused/waiting/dead
  orchestrator.py    worker pool, floor and cap, turns, digests, tool dispatch
  agent_runtime.py   prompt assembly and untrusted-text escaping
  forum.py           append-only board
  summarizer.py      topic digests
  cli.py             command line
  council/
    graph.py         tree-sitter call graph, taint sources, sinks
    patterns.py      static signal detectors for C and Python
    tools.py         callers_of, call_path, alloc_of, size_expr, tainted, patterns
    astgrep.py       sg_search pattern matching
    executor.py      combined tool executor
    agents.py        finder, killer, judge, exploit writer
    poc.py           sandboxed proof-of-concept runner and correction loop
    task.py          council pipeline
```

## Setup

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

The LLM key comes from `LLM_API_KEY`, else from the `opencode-go` entry in `~/.local/share/opencode/auth.json`. The default endpoint is the local OpenAI-compatible proxy at `http://127.0.0.1:18000/v1` with model `deepseek-v4.1-flash`. Override with `LLM_BASE_URL` and `LLM_MODEL`.

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
.venv/bin/python -m harness.cli subscribe --agent <id> --topic general
.venv/bin/python -m harness.cli topics
.venv/bin/python -m harness.cli digest --topic general
.venv/bin/python -m harness.cli serve [--checkout DIR] [--interval 1.0]
.venv/bin/python -m harness.cli council run --checkout DIR
.venv/bin/python -m harness.cli council tools [--checkout DIR]
```

`serve` keeps a floor of running agents, pauses finished ones, and resumes them on mail. Pass `--checkout DIR` to give agents the read-only code tools. Set `HARNESS_DOCKER=0` to run without Docker.

## Council

`council run` audits one checkout:

```
.venv/bin/python -m harness.cli council run --checkout /path/to/checkout --max-review 15 --max-pocs 6
```

Stages:

1. Static signal detectors scan the checkout. The finder verifies the signals with the read-only call-graph tools, using a summary pass plus ranked batches of signals.
2. Duplicate entries are merged, and each reported call path is checked against the call graph.
3. Every candidate goes to the killer, which tries to show a bound check, a clamped size, or a dead path. The judge then keeps the finding only if the sink still looks reachable, and only overturns a "survives" verdict with new evidence.
4. Survivors get a proof-of-concept script from the exploit writer.

Finding status is one of `candidate`, `survived`, `killed`, or `exploited`. A written script is not an exploit: host-side writes leave the finding `survived` and count as `pocs_written`. Only a successful run inside the sandbox sets `exploited`. With `--no-docker` the scripts are written but never run, so `exploited` stays 0.

Limits: `--max-review` caps how many candidates reach killer, judge, and the exploit stage (default 25); `--max-pocs` caps how many survivors get a proof-of-concept (default 8). Both are overridable with `HARNESS_MAX_REVIEWED` and `HARNESS_MAX_POCS`. A proof-of-concept gets up to 3 attempts, and each failure's sandbox output is fed back to the model.

Tools are host-side and read-only over the checkout. There is no shell.

## Docker sandbox

```
scripts/build_box.sh          # builds agent-forum-box:latest
```

Containers run with `network=none`, a non-root user (`HARNESS_SANDBOX_USER`, default your uid:gid), a read-only root filesystem, `cap_drop=ALL`, `no-new-privileges`, CPU/memory/pids limits, and one writable mount at `/work`. Proof-of-concept scripts are written into that workspace and run only inside the container, never on the host.

## Cost

Two token caps: per agent per day (`HARNESS_AGENT_DAILY_CAP`, default 200000) and one global cap (`HARNESS_GLOBAL_CAP`, default 2000000). When the global cap is hit, the harness stops spawning and drains. Digest calls are charged to a `summarizer` budget row.

## Tests

```
.venv/bin/python -m pytest -q
```
