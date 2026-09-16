# Interfaces

Signatures and contracts for the harness. Update this file when one changes.

## Runtime LLM endpoint

- Base URL `http://127.0.0.1:18000/v1` (OpenAI-compatible), model `deepseek-v4.1-flash`.
- Key comes from `Settings.llm_api_key`: `LLM_API_KEY`, else `OPENCODE_GO_API_KEY`, else the `opencode-go` entry in `~/.local/share/opencode/auth.json`.
- Override with `LLM_BASE_URL` and `LLM_MODEL`.

## Core

### `models.py`

Enums:

- `AgentState`: `spawned`, `running`, `paused`, `waiting`, `dead`.
- `AgentRole`: `finder`, `killer`, `judge`, `exploit`, `summarizer`, `generic`.
- `ContainerStatus`: `created`, `running`, `stopped`, `removed`.
- `FindingStatus`: `candidate`, `survived`, `killed`, `exploited`.
- `TurnState`: `running`, `done`, `failed`, `budget_exceeded`.

Dataclasses include `Agent`, `Turn`, `Container`, `MailMessage`, `BudgetRow`, `Topic`, `Post`, `Digest`, `Finding`, `Usage`.

### `db.py`

`Database(path)`, then `await connect()`, `await init_schema()`, `await close()`. One `aiosqlite` connection, serialized through internal helpers.

Method groups:

- Agents: `create_agent`, `get_agent`, `list_agents`, `count_agents`, `set_agent_state`, `transition_state`, `set_agent_container`, `set_agent_context_path`, `set_agent_topic`, `touch_agent_wake`.
- Turns: `start_turn`, `end_turn`, `get_turn`, `list_turns`, `running_turn`, `close_stale_turns`.
- Containers: `upsert_container`, `set_container_status`, `get_container`, `list_containers`.
- Mail: `send_mail`, `unread_mail`, `drain_mail`, `count_unread`, `mark_mail_read`.
- Budget: `get_budget`, `add_tokens`, `add_tokens_many`.
- Forum: `create_topic`, `topic_by_title`, `get_topic`, `list_topics`, `set_topic_owner`, `set_topic_owner_cas`, `bump_topic`, `set_topic_digest_mark`, `create_post`, `insert_post_limited`, `get_post`, `list_posts`, `list_posts_recent`, `count_posts_since`, `last_post_id`, `subscribe`, `unsubscribe`, `list_subscribers`, `list_subscriptions`, `add_digest`, `add_digest_and_mark`, `latest_digest`.
- Findings and PoCs: `create_finding`, `get_finding`, `list_findings`, `set_finding_status`, `add_poc_run`, `list_poc_runs`.

### `llm.py`

```python
class LLMError(Exception): ...
class LLMBudgetError(LLMError): ...

@dataclass
class LLMResponse:
    text: str
    model: str
    usage: m.Usage          # tokens_in, tokens_out
    finish_reason: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

class LLMClient:
    def __init__(self, settings, logger=None, budget=None, agent_id=None): ...
    async def complete(self, messages, tools=None, max_tokens=None,
                       temperature=None, agent_id=None,
                       response_format=None) -> LLMResponse: ...
    async def aclose(self) -> None: ...
```

- POSTs to `{llm_base_url}/chat/completions`.
- Reads `choices[0].message.content`, `choices[0].message.tool_calls`, and `usage`.
- Calls `budget.check_agent()` before the request and `budget.record()` after.
- Logs each call through `EventLogger.llm_call`.
- `httpx.AsyncClient`, 120s timeout, one retry on 429 and 5xx.

### `budget.py`

```python
class BudgetExceeded(Exception): ...

class BudgetGuard:
    def __init__(self, db, settings, logger=None): ...
    @staticmethod
    def today() -> str: ...                 # UTC date, "YYYY-MM-DD"
    async def agent_used(self, agent_id) -> int: ...
    async def global_used(self) -> int: ...
    async def check_agent(self, agent_id) -> None: ...   # raises BudgetExceeded
    async def check_global(self) -> None: ...            # raises BudgetExceeded
    async def can_spawn(self) -> bool: ...
    async def record(self, agent_id, tokens) -> None: ...
    async def remaining_global(self) -> int: ...
```

Two `budget` rows per day per agent: one `scope="agent"` keyed by agent id, one `scope="global"` keyed `"global"`. `record` updates both.

### `logging_.py`

```python
class EventLogger:
    def __init__(self, log_dir, agent_id=None, max_bytes=DEFAULT_LOG_MAX_BYTES): ...
    def event(self, kind, **fields) -> None: ...
    def llm_call(self, *, agent_id, model, tokens_in, tokens_out,
                 latency_ms, error=None, note=None) -> None: ...
    def tool_call(self, *, agent_id, name, args, result=None,
                  error=None, latency_ms=None) -> None: ...
    def close(self) -> None: ...
```

Appends one JSON object per line to `<log_dir>/llm.jsonl`, `<log_dir>/tools.jsonl`, and `<log_dir>/events.jsonl`. Rotates at `max_bytes`. Never raises on a bad payload.

## Runtime

### `containers.py`

```python
@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str

class ContainerManager:
    def __init__(self, settings, db, logger=None): ...
    def workspace_for(self, agent_id) -> Path: ...   # absolute, resolved
    async def create(self, agent_id, image=None) -> m.Container: ...
    async def start(self, container_id) -> None: ...
    async def stop(self, container_id, timeout=10) -> None: ...
    async def remove(self, container_id) -> None: ...
    async def exec(self, container_id, argv, timeout=60) -> ExecResult: ...
    async def status(self, container_id) -> m.ContainerStatus: ...
    async def aclose(self) -> None: ...
```

- The Docker SDK runs inside `asyncio.to_thread`.
- Hardening: `network_mode=settings.sandbox_network` (`none` by default), `cap_drop=["ALL"]`, `security_opt=["no-new-privileges"]`, `pids_limit=256`, `mem_limit="1g"`, `read_only=True`, a `64m` tmpfs at `/tmp`, and the workspace bind-mounted at `/work`.
- When `settings.docker_enabled` is False: no Docker calls, the container id is `fake-<agent_id>`, `status` returns `RUNNING`, and rows are still written to `containers`.

### `mailbox.py`

```python
class Mailbox:
    def __init__(self, db, logger=None): ...
    async def send(self, agent_id, body, sender_id=None,
                   topic_id=None, post_id=None) -> m.MailMessage: ...
    async def drain(self, agent_id) -> list[m.MailMessage]: ...
    async def has_unread(self, agent_id) -> bool: ...
    async def notify_subscribers(self, post, subscribers) -> list[str]: ...
```

`notify_subscribers` mails every subscriber except the author and returns the recipients.

### `state_machine.py`

```python
class InvalidTransition(Exception): ...

class StateMachine:
    ALLOWED: ClassVar[dict[m.AgentState, set[m.AgentState]]]
    def __init__(self, db, logger=None): ...
    @staticmethod
    def can_transition(src, dst) -> bool: ...
    @staticmethod
    def initial() -> m.AgentState: ...     # SPAWNED
    async def transition(self, agent_id, dst, reason=None) -> m.Agent: ...
```

Allowed: spawned to {running, waiting, dead}; running to {paused, waiting, dead}; paused to {waiting, running, dead}; waiting to {running, paused, dead}; dead to {}.

### `forum.py`

```python
class RateLimited(Exception): ...
class NotTopicOwner(Exception): ...
class InvalidPost(Exception): ...

@dataclass
class PostResult:
    post: m.Post
    woken: list[str]

class Forum:
    def __init__(self, db, mailbox, settings, logger=None): ...
    async def ensure_topic(self, title, owner_id=None) -> m.Topic: ...
    async def subscribe(self, agent_id, topic_id) -> None: ...
    async def unsubscribe(self, agent_id, topic_id) -> None: ...
    async def subscribers(self, topic_id) -> list[str]: ...
    async def owner_id(self, topic_id) -> str | None: ...
    async def post(self, agent_id, topic_id, body,
                   parent_id=None, refs=None) -> PostResult: ...
    async def read(self, topic_id, since_id=0, limit=100) -> list[m.Post]: ...
    async def handoff(self, topic_id, from_agent, to_agent, note="") -> PostResult: ...
```

- `post` validates: agent exists, topic exists, body length 1..`MAX_POST_CHARS` (8000), `parent_id` (when set) points at a post in the same topic, and `refs` is a list of dicts with string keys. Bad input raises `InvalidPost`.
- `post` enforces the rate limit with `db.count_posts_since(agent_id, now-3600)` against `settings.posts_per_agent_per_hour`; at or over the limit it raises `RateLimited`.
- `post` is the only write path. It inserts, bumps the topic, mails every subscriber except the author, and returns `PostResult`. No state is kept outside the database.
- `handoff` raises `NotTopicOwner` unless `from_agent` is the current owner, then moves ownership and posts a short handoff message.
- `ensure_topic` is idempotent by title.

### `summarizer.py`

```python
DIGEST_SYSTEM_PROMPT: str

async def summarize_topic(db, llm, topic_id, every_n=25, force=False,
                          logger=None) -> dict | None: ...
```

Reads posts after `topics.last_digest_post_id`. With fewer than `every_n` new posts and not `force`, returns `None`. Otherwise asks the model for a plain-text digest (hard cap 1500 chars), stores it with `db.add_digest`, and advances the mark. Returns `{"topic_id", "from_post_id", "to_post_id", "body"}`. No tools and no container.

### `orchestrator.py` and `agent_runtime.py`

```python
class Orchestrator:
    def __init__(self, settings, db, logger, containers, mailbox, budget,
                 state_machine, llm): ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def tick(self) -> None: ...          # one reconcile pass; used by tests
    async def run_forever(self, interval=1.0) -> None: ...
    async def spawn_agent(self, role) -> m.Agent: ...
    async def request_wake(self, agent_id) -> None: ...
    async def run_turn(self, agent_id) -> None: ...

async def build_orchestrator(settings, checkout=None) -> Orchestrator: ...
```

Rules:

- Floor `settings.floor`, cap `settings.hard_cap`. Spawn only when running is below the floor and `budget.can_spawn()`.
- Agents beyond `hard_cap` go to `WAITING` and enter when a slot frees.
- `run_turn` logs a `turns` row, calls the model until `settings.turn_token_budget` is spent or the model stops (bounded by `MAX_LLM_CALLS = 8`), writes context to `settings.workspace_dir/<agent_id>/context.json`, stops the container, and sets the agent `PAUSED`.
- Wake on mail: `request_wake` moves a paused or waiting agent to running if a slot is free, else waiting.
- A pool of `settings.worker_pool_size` workers pulls from an asyncio queue.
- `agent_runtime` assembles prompts (`build_messages`), escapes untrusted text, and exposes `tool_specs(tools=None)`, which returns `tools.specs()` when a tool executor is supplied, else `[]`.

### `cli.py`

Commands: `init`, `status`, `spawn --role R`, `post --topic T --body B --agent A`, `subscribe --agent A --topic T`, `topics`, `digest --topic T`, `serve [--interval S] [--checkout DIR]`, `council run --checkout DIR [--topic T] [--no-docker] [--max-review N] [--max-pocs N]`, `council tools [--checkout DIR]`. Runs as `python -m harness.cli`.

## Council

### `council/graph.py`

```python
@dataclass
class Alloc:
    name: str; line: int; size_expr: str | None

@dataclass
class BufferSize:
    name: str; line: int; size_expr: str | None

@dataclass
class TaintSource:
    name: str; line: int; kind: str

@dataclass
class Sink:
    name: str; line: int; arg: str

@dataclass
class Function:
    name: str; file: str; line: int; end_line: int
    params: list[str]
    calls: list[str]
    allocs: list[Alloc]
    buffers: list[BufferSize]
    sources: list[TaintSource]
    sinks: list[Sink]

class CodeGraph:
    def __init__(self, root: Path): ...
    def build(self) -> None: ...
    @property
    def files(self) -> list[str]: ...
    @property
    def functions(self) -> dict[str, list[Function]]: ...
    @property
    def signals(self) -> list[Signal]: ...
    def callers_of(self, fn) -> list[tuple[Function, int]]: ...
    def call_path(self, source, sink, max_depth=8, max_paths=20) -> list[list[str]]: ...
    def alloc_of(self, fn, ptr=None) -> list[Alloc]: ...
    def size_expr(self, fn, buf) -> list[BufferSize]: ...
    def tainted(self, fn, expr) -> bool: ...
    def to_dict(self) -> dict: ...
```

Parses `.c` and `.h` with tree-sitter C and `.py` with tree-sitter Python. `signals` holds the static signals found during `build`. Never executes checkout code.

### `council/patterns.py`

Static detectors. `Signal` is a frozen dataclass `{kind, file, line, function, detail, cwe, confidence}`. Kinds include `unchecked_alloc`, `realloc_loses_original`, `double_free`, `use_after_free`, `integer_overflow_alloc`, `size_truncation`, `off_by_one`, `format_string`, `missing_nul_termination`, `unchecked_io_return`, `signal_unsafe`, `dest_scratch_write`, `snprintf_offset_write`, `toctou`, `insecure_temp_file`, `unchecked_retry_loop`, and `shell_injection`. Some kinds are C-only; `shell_injection` covers Python. Guard detection scans the whole enclosing function: an `if`, an early return, `goto`, `continue`, a clamp, a `MIN`/`MAX`, a mask, or one hop into a same-file helper that checks the parameter.

### `council/tools.py`

```python
class CouncilTools:
    def __init__(self, graph: CodeGraph): ...
    def specs(self) -> list[dict]: ...
    def dispatch(self, name, arguments) -> dict: ...
    def callers_of(self, fn) -> dict: ...
    def call_path(self, source, sink) -> dict: ...
    def alloc_of(self, ptr, fn=None) -> dict: ...
    def size_expr(self, buf, fn=None) -> dict: ...
    def tainted(self, expr, fn=None) -> dict: ...
    def patterns(self, kind=None, path=None, limit=100) -> dict: ...
```

Tool names: `callers_of`, `call_path`, `alloc_of`, `size_expr`, `tainted`, `patterns`. `dispatch` validates arguments and returns `{"ok": bool, ...}`. No shell.

### `council/astgrep.py`

```python
@dataclass
class Match:
    file: str; line: int; column: int; end_line: int; end_column: int
    text: str; captures: dict[str, str]

class AstGrep:
    def __init__(self, root: Path): ...
    def search(self, pattern, lang=None, path=None, limit=200) -> list[Match]: ...
    def specs(self) -> list[dict]: ...
    def dispatch(self, name, arguments) -> dict: ...
```

Tool name: `sg_search`. Patterns support `$VAR` metavariables, converted to tree-sitter query captures, with a plain-literal fallback. Read-only.

### `council/agents.py`

```python
FINDER_PROMPT: str
FINDER_TRIAGE_PROMPT: str
KILLER_PROMPT: str
JUDGE_PROMPT: str
EXPLOIT_PROMPT: str

@dataclass
class CouncilResult:
    topic_id: str
    candidates: int
    survived: int
    killed: int
    exploited: int
    precision: float
    findings: list[m.Finding]
    notes: list[str]

async def run_tool_agent(llm, tools, system_prompt, task_prompt, agent_id,
                         max_calls=8, temperature=None, logger=None) -> dict: ...

async def finder_audit(llm, tools, db, topic_id, agent_id, checkout,
                       logger=None) -> list[m.Finding]: ...

async def killer_refute(llm, tools, db, finding, agent_id,
                        logger=None) -> KillerVerdict: ...

async def judge_review(llm, tools, db, finding, agent_id,
                       killer_reason=None, logger=None) -> m.FindingStatus: ...

async def write_exploit(llm, tools, db, finding, workspace, agent_id,
                        logger=None) -> str | None: ...
```

- The finder runs a summary pass, then up to 8 triage batches of 12 ranked signals each (cap 96 signals). Entries are deduped and validated: the file must be in the graph, the sink must be a known sink or a matching signal kind, and each call path edge is checked against the graph (an unconfirmed path is kept and marked).
- The killer tries to break a finding (bound check, clamped size, dead path). The judge keeps it only if the path still looks reachable, and only overturns a "survives" verdict with new evidence.
- `write_exploit` writes a script into the sandbox workspace and never runs it on the host. It refuses an invalid id, an oversized body, or a symlinked path.
- `precision = survived / max(1, survived + killed)`.
- The result also carries `reviewed`, `pending`, `pocs_run`, `pocs_succeeded`, `pocs_written`, `pocs_capped`, and `sandbox_ran`, used by the CLI summary.

### `council/poc.py`

```python
SUCCESS_MARKER = "POC_SUCCESS"
POC_MAX_ATTEMPTS = 3

@dataclass
class PocAttempt:
    attempt: int; script: str; exit_code: int
    stdout: str; stderr: str; success: bool; duration_ms: int = 0

@dataclass
class PocResult:
    finding_id: str; success: bool
    attempts: list[PocAttempt] = field(default_factory=list)
    script_path: str | None = None
    reason: str | None = None

def stage_checkout(checkout, workspace) -> Path: ...

async def run_and_correct_poc(llm, db, containers, container_id, finding,
                              checkout, workspace, agent_id,
                              max_attempts=POC_MAX_ATTEMPTS,
                              logger=None) -> PocResult: ...
```

The checkout is staged read-only at `/work/checkout`. The model writes a script, the sandbox runs `python3 /work/<name>.py`, and on failure the output is fed back for a rewrite, up to `max_attempts`. A run succeeds only when the exit code is 0 and `POC_SUCCESS` is on stdout. Nothing runs on the host.

### `council/task.py`

```python
MAX_REVIEWED_DEFAULT = 25
MAX_POCS_DEFAULT = 8

async def run_council(llm, tools, db, forum, topic_id, checkout,
                      finder_id, killer_id, judge_id, exploit_id,
                      logger=None, containers=None,
                      max_reviewed=None, max_pocs=None) -> CouncilResult: ...
```

Pipeline: subscribe the four agents, run the finder, announce the candidates, then run killer and judge per candidate up to `max_reviewed` (default 25, env `HARNESS_MAX_REVIEWED`). Survivors beyond that cap stay `candidate`. For the first `max_pocs` survivors (default 8, env `HARNESS_MAX_POCS`), if Docker is enabled, one container is created for the whole pass, each PoC is run inside it, and a success sets the finding `exploited`, then the container is removed. If Docker is disabled or `--no-docker` is set, the scripts are written host-side, the finding stays `survived`, and the count is reported as `pocs_written`.

### `council/executor.py`

```python
class CombinedTools:
    def __init__(self, council: CouncilTools, astgrep: AstGrep): ...
    @property
    def graph(self) -> CodeGraph | None: ...
    def specs(self) -> list[dict]: ...
    def dispatch(self, name, arguments) -> dict: ...

def build_combined_tools(checkout) -> CombinedTools: ...
```

Unions the council tools and `sg_search` behind one `specs`/`dispatch` surface and builds the code graph once.
