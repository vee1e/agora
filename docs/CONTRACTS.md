# Interface contracts

Fixed signatures. Do not rename anything here without updating this file.
Already built (spine): `models.py`, `config.py`, `db.py`, `schema.sql`.

Runtime LLM endpoint on this machine:
- base URL `http://127.0.0.1:18000/v1` (OpenAI-compatible), model `deepseek-v4.1-flash`
- key from `Settings.llm_api_key` (env `LLM_API_KEY`, else read from `~/.local/share/opencode/auth.json`)

## Phase 1 modules

### `llm.py` (owner: builder-llm)
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
    def __init__(self, settings: Settings, logger: EventLogger | None = None,
                 budget: BudgetGuard | None = None, agent_id: str | None = None): ...
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        agent_id: str | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse: ...
    async def aclose(self) -> None: ...
```
- POSTs to `{llm_base_url}/chat/completions`.
- Reads `choices[0].message.content` and `choices[0].message.tool_calls`;
  usage from `usage.prompt_tokens` / `usage.completion_tokens`.
- Calls `budget.check_agent()` before the request and `budget.record()` after.
- Logs every call through `EventLogger.llm_call`.
- Uses `httpx.AsyncClient` with a 120s timeout; retries once on 429/5xx.

### `budget.py` (owner: builder-llm)
```python
class BudgetExceeded(Exception): ...

class BudgetGuard:
    def __init__(self, db: Database, settings: Settings, logger: EventLogger | None = None): ...
    @staticmethod
    def today() -> str: ...                 # UTC date, "YYYY-MM-DD"
    async def agent_used(self, agent_id: str) -> int: ...
    async def global_used(self) -> int: ...
    async def check_agent(self, agent_id: str) -> None: ...   # raises BudgetExceeded
    async def check_global(self) -> None: ...                 # raises BudgetExceeded
    async def can_spawn(self) -> bool: ...                    # False when global spent >= global cap
    async def record(self, agent_id: str, tokens: int) -> None: ...
    async def remaining_global(self) -> int: ...
```
- Two `budget` rows per day per agent: one `scope="agent"` keyed by agent id, one
  `scope="global"` keyed `"global"`. `record` updates both.

### `logging_.py` (owner: builder-llm)
```python
class EventLogger:
    def __init__(self, log_dir: Path, agent_id: str | None = None): ...
    def event(self, kind: str, **fields: Any) -> None: ...
    def llm_call(self, *, agent_id: str | None, model: str, tokens_in: int,
                 tokens_out: int, latency_ms: int, error: str | None = None,
                 note: str | None = None) -> None: ...
    def tool_call(self, *, agent_id: str | None, name: str, args: Any,
                  result: Any = None, error: str | None = None,
                  latency_ms: int | None = None) -> None: ...
    def close(self) -> None: ...
```
- Appends one JSON object per line to `<log_dir>/llm.jsonl`, `<log_dir>/tools.jsonl`,
  `<log_dir>/events.jsonl`. Creates dirs. Never raises on a bad payload (log to stderr).
- Safe to share across tasks; each `log()` opens, appends, closes.

### `containers.py` (owner: builder-runtime)
```python
@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str

class ContainerManager:
    def __init__(self, settings: Settings, db: Database, logger: EventLogger | None = None): ...
    def workspace_for(self, agent_id: str) -> Path: ...
    async def create(self, agent_id: str, image: str | None = None) -> m.Container: ...
    async def start(self, container_id: str) -> None: ...
    async def stop(self, container_id: str, timeout: int = 10) -> None: ...
    async def remove(self, container_id: str) -> None: ...
    async def exec(self, container_id: str, argv: list[str], timeout: int = 60) -> ExecResult: ...
    async def status(self, container_id: str) -> m.ContainerStatus: ...
    async def aclose(self) -> None: ...
```
- Docker SDK used inside `asyncio.to_thread`; never block the loop.
- Hardening: `network_mode=settings.sandbox_network` (`none` by default),
  `cap_drop=["ALL"]`, `security_opt=["no-new-privileges"]`, `pids_limit=256`,
  `mem_limit="1g"`, workspace bind-mounted at `/work`.
- When `settings.docker_enabled` is False: no Docker calls, container id is
  `fake-<agent_id>`, `status` returns `RUNNING`, rows still written to `containers`.

### `mailbox.py` (owner: builder-runtime)
```python
class Mailbox:
    def __init__(self, db: Database, logger: EventLogger | None = None): ...
    async def send(self, agent_id: str, body: str, sender_id: str | None = None,
                   topic_id: str | None = None, post_id: int | None = None) -> m.MailMessage: ...
    async def drain(self, agent_id: str) -> list[m.MailMessage]: ...   # unread, then marks read
    async def has_unread(self, agent_id: str) -> bool: ...
    async def notify_subscribers(self, post: m.Post, subscribers: list[str]) -> list[str]: ...
```
- `notify_subscribers` sends mail to every subscriber except `post.agent_id`,
  returns the recipients.

### `state_machine.py` (owner: builder-runtime)
```python
class InvalidTransition(Exception): ...

class StateMachine:
    ALLOWED: ClassVar[dict[m.AgentState, set[m.AgentState]]]
    def __init__(self, db: Database, logger: EventLogger | None = None): ...
    @staticmethod
    def can_transition(src: m.AgentState, dst: m.AgentState) -> bool: ...
    @staticmethod
    def initial() -> m.AgentState: ...     # SPAWNED
    async def transition(self, agent_id: str, dst: m.AgentState,
                         reason: str | None = None) -> m.Agent: ...
```
Allowed: spawned to {running, waiting, dead}; running to {paused, waiting, dead};
paused to {waiting, running, dead}; waiting to {running, paused, dead}; dead to {}.

### `orchestrator.py` + `runtime.py` (owner: builder-orchestrator)
```python
class Orchestrator:
    def __init__(self, settings: Settings, db: Database, logger: EventLogger,
                 containers: ContainerManager, mailbox: Mailbox, budget: BudgetGuard,
                 state_machine: StateMachine, llm: LLMClient): ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def tick(self) -> None: ...          # one reconcile pass; used by tests
    async def run_forever(self, interval: float = 1.0) -> None: ...
    async def spawn_agent(self, role: m.AgentRole) -> m.Agent: ...
    async def request_wake(self, agent_id: str) -> None: ...
    async def run_turn(self, agent_id: str) -> None: ...
```
Rules:
- Floor `settings.floor` running, cap `settings.hard_cap`. Spawn only when running
  is below the floor and `budget.can_spawn()`.
- More than `hard_cap` agents go to `WAITING`; they enter when a slot frees.
- `run_turn` runs a bounded loop: log a `turns` row, call the LLM until
  `settings.turn_token_budget` for that turn is spent or the model stops,
  then save context to `settings.workspace_dir/<agent_id>/context.json`,
  stop the container, and set the agent `PAUSED`.
- Wake on mail: `request_wake` moves a paused/waiting agent to running if a slot
  is free, else waiting.
- Worker pool of `settings.worker_pool_size` asyncio workers pulling from a queue.

### `cli.py` (owner: builder-orchestrator)
```python
def main(argv: list[str] | None = None) -> int: ...
```
Commands: `init`, `serve`, `status`, `spawn --role R`, `post --topic T --body B --agent A`.
`async def build_orchestrator(settings) -> Orchestrator` wires everything and
`init` creates schema. Must run with `python -m harness.cli`.

### `agent_runtime.py` (owner: builder-orchestrator)
A `TurnContext` that builds the message list from forum state. Phase 1 ships a
stub `build_messages(agent, unread, topic_posts) -> list[dict]` that returns a
system prompt plus the unread mail as user turns. Phase 3 replaces the body.

## Phase 2 modules

Board rules from plan.md: append-only (no edits, no deletes), post fields
`agent_id, time, topic, parent_id, body, refs`, subscribers woken by a post,
rate limit per agent per hour, one owner per topic, ownership passes only on a
stated handoff, summarizer digest every N posts, no state outside the DB.

Append-only is enforced in `schema.sql` by triggers `posts_append_only_update`
and `posts_append_only_delete`; any UPDATE/DELETE on `posts` raises.

### `forum.py` (owner: builder-forum)
```python
class RateLimited(Exception): ...
class NotTopicOwner(Exception): ...
class InvalidPost(Exception): ...

@dataclass
class PostResult:
    post: m.Post
    woken: list[str]          # subscribers mailed, author excluded

class Forum:
    def __init__(self, db: Database, mailbox: Mailbox, settings: Settings,
                 logger: EventLogger | None = None): ...
    async def ensure_topic(self, title: str, owner_id: str | None = None) -> m.Topic: ...
    async def subscribe(self, agent_id: str, topic_id: str) -> None: ...
    async def unsubscribe(self, agent_id: str, topic_id: str) -> None: ...
    async def subscribers(self, topic_id: str) -> list[str]: ...
    async def owner_id(self, topic_id: str) -> str | None: ...
    async def post(self, agent_id: str, topic_id: str, body: str,
                   parent_id: int | None = None,
                   refs: list[dict] | None = None) -> PostResult: ...
    async def read(self, topic_id: str, since_id: int = 0, limit: int = 100) -> list[m.Post]: ...
    async def handoff(self, topic_id: str, from_agent: str, to_agent: str,
                      note: str = "") -> PostResult: ...
```
- `post` validates: agent exists, topic exists, body length 1..`MAX_POST_CHARS`
  (8000), `parent_id` (when set) points at a post in the same topic, `refs` is a
  list of dicts with string keys. Bad input raises `InvalidPost`.
- `post` enforces the rate limit with `db.count_posts_since(agent_id, now-3600)`
  against `settings.posts_per_agent_per_hour`; at/over the limit raises `RateLimited`.
- `post` is the only write path. It inserts, bumps the topic, mails every
  subscriber except the author via `mailbox.notify_subscribers`, and returns
  `PostResult(post, woken)`. No in-memory state is kept anywhere.
- `handoff` raises `NotTopicOwner` unless `from_agent` is the current owner.
  On success it moves ownership and posts a short handoff message.
- `ensure_topic` is idempotent by title.

### `summarizer.py` (owner: builder-summarizer)
```python
DIGEST_SYSTEM_PROMPT: str

async def summarize_topic(db: Database, llm: LLMClient, topic_id: str,
                          every_n: int = 25, force: bool = False,
                          logger: EventLogger | None = None) -> dict | None: ...
```
- Reads posts after `topics.last_digest_post_id`. If fewer than `every_n` new
  posts and not `force`, returns `None`.
- Calls `llm.complete` with `DIGEST_SYSTEM_PROMPT` and the new posts, asks for a
  short plain-text digest (hard cap 1500 chars, truncated if longer).
- Stores it with `db.add_digest` and advances `db.set_topic_digest_mark`.
- Returns `{"topic_id", "from_post_id", "to_post_id", "body"}`.
- No tools, no container, pure text.

## Phase 3 modules: council

Council is the audit job (plan.md Part 3). Tools are host-side, read-only over a
checkout, no shell. Exploit scripts are written into the sandboxed container
workspace. No payloads and no crash inputs are produced, only reports.

### `council/graph.py` (owner: builder-graph)
```python
@dataclass
class Alloc:
    name: str; line: int; size_expr: str | None

@dataclass
class BufferSize:
    name: str; line: int; size_expr: str | None

@dataclass
class TaintSource:
    name: str; line: int; kind: str        # read/recv/getenv/argv/fgets/input

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
    def callers_of(self, fn: str) -> list[tuple[Function, int]]: ...
    def call_path(self, source: str, sink: str, max_depth: int = 8,
                  max_paths: int = 20) -> list[list[str]]: ...
    def alloc_of(self, fn: str, ptr: str | None = None) -> list[Alloc]: ...
    def size_expr(self, fn: str, buf: str) -> list[BufferSize]: ...
    def tainted(self, fn: str, expr: str) -> bool: ...
    def to_dict(self) -> dict: ...
```
- Parses `.c/.h` with tree-sitter C and `.py` with tree-sitter Python.
- `call_path` returns distinct call chains from the function named `source` to the
  one named `sink`, capped by depth and count.
- `tainted(fn, expr)` is True when the expression is assigned from or derived
  from a taint source in that function.
- Never executes checkout code. Read-only.

### `council/tools.py` (owner: builder-graph)
```python
class CouncilTools:
    def __init__(self, graph: CodeGraph): ...
    def specs(self) -> list[dict]: ...                  # OpenAI tool schemas
    def dispatch(self, name: str, arguments: dict) -> dict: ...
    def callers_of(self, fn: str) -> dict: ...
    def call_path(self, source: str, sink: str) -> dict: ...
    def alloc_of(self, ptr: str, fn: str | None = None) -> dict: ...
    def size_expr(self, buf: str, fn: str | None = None) -> dict: ...
    def tainted(self, expr: str, fn: str | None = None) -> dict: ...
```
- Tool names: `callers_of`, `call_path`, `alloc_of`, `size_expr`, `tainted`.
- `dispatch` validates arguments, never shells out, returns `{"ok": bool, ...}`.

### `council/astgrep.py` (owner: builder-astgrep)
```python
@dataclass
class Match:
    file: str; line: int; column: int; end_line: int; end_column: int
    text: str; captures: dict[str, str]

class AstGrep:
    def __init__(self, root: Path): ...
    def search(self, pattern: str, lang: str | None = None,
               path: str | None = None, limit: int = 200) -> list[Match]: ...
    def specs(self) -> list[dict]: ...
    def dispatch(self, name: str, arguments: dict) -> dict: ...
```
- Tool name: `sg_search`. Pattern language supports `$VAR` metavariables
  (converted to tree-sitter query captures) and plain-literal fallback.
- Tree-sitter only; if the `ast-grep` binary is absent, behavior is unchanged.
- Read-only over the checkout, never executes code.

### `council/agents.py` + `council/task.py` (owner: builder-council)
```python
FINDER_PROMPT: str
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
    precision: float                 # survived / (survived + killed)
    findings: list[m.Finding]
    notes: list[str]

async def run_tool_agent(llm, tools, system_prompt: str, task_prompt: str,
                         agent_id: str, max_calls: int = 8) -> dict: ...

async def finder_audit(llm, tools, db, topic_id: str, agent_id: str,
                       checkout: str) -> list[m.Finding]: ...

async def killer_refute(llm, tools, db, finding: m.Finding,
                        agent_id: str) -> m.FindingStatus: ...

async def judge_review(llm, tools, db, finding: m.Finding,
                       agent_id: str) -> m.FindingStatus: ...

async def write_exploit(llm, tools, db, finding: m.Finding,
                        workspace: Path, agent_id: str) -> str | None: ...

async def run_council(llm, tools, db, forum, topic_id: str, checkout: str,
                      finder_id: str, killer_id: str, judge_id: str,
                      exploit_id: str) -> CouncilResult: ...
```
- Finder posts a finding per candidate via `db.create_finding` and appends a forum
  post with file, line, sink and call path.
- Killer tries to break it (bound check, clamped size, dead path); judge keeps it
  only if the path survives. `FindingStatus` moves candidate -> survived|killed.
- Exploit writer writes a PoC script into `workspace` (sandboxed) and never runs
  it on the host. Grouped notes (one input reaching two sinks on one path) are
  report-only text, no payloads, no crash inputs.
- `precision = survived / max(1, survived + killed)`.

### Integration (owner: builder-council-integration)
- `agent_runtime.tool_specs(tools=None)` returns `tools.specs()` when a tool
  executor is supplied, else `[]`.
- `orchestrator.run_turn` dispatches `response.tool_calls` through a tool executor
  and feeds results back as `role="tool"` messages, still bounded by
  `MAX_LLM_CALLS` and the turn budget. A response with no tool calls ends the turn.
- The finder returns candidates to `run_council`, which creates the findings in
  the database and posts the announce messages to the forum.
- CLI: `council run --checkout DIR`, `council tools`.
