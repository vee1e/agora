PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS agents (
    id            TEXT PRIMARY KEY,
    role          TEXT NOT NULL,
    state         TEXT NOT NULL,
    container_id  TEXT,
    topic_id      TEXT,
    context_path  TEXT,
    created_at    REAL NOT NULL DEFAULT 0,
    updated_at    REAL NOT NULL DEFAULT 0,
    last_wake_at  REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_agents_state ON agents(state);

CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT NOT NULL,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    state       TEXT NOT NULL,
    tokens_in   INTEGER NOT NULL DEFAULT 0,
    tokens_out  INTEGER NOT NULL DEFAULT 0,
    reason      TEXT
);
CREATE INDEX IF NOT EXISTS idx_turns_agent ON turns(agent_id);

CREATE TABLE IF NOT EXISTS containers (
    id             TEXT PRIMARY KEY,
    agent_id       TEXT,
    image          TEXT NOT NULL,
    workspace_path TEXT NOT NULL,
    status         TEXT NOT NULL,
    created_at     REAL NOT NULL DEFAULT 0,
    updated_at     REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_containers_agent ON containers(agent_id);

CREATE TABLE IF NOT EXISTS mailbox (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT NOT NULL,
    sender_id   TEXT,
    topic_id    TEXT,
    post_id     INTEGER,
    body        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    read_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_mailbox_unread ON mailbox(agent_id, read_at);

CREATE TABLE IF NOT EXISTS budget (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scope       TEXT NOT NULL,
    key         TEXT NOT NULL,
    day         TEXT NOT NULL,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    cap         INTEGER NOT NULL DEFAULT 0,
    updated_at  REAL NOT NULL DEFAULT 0,
    UNIQUE(scope, key, day)
);

CREATE TABLE IF NOT EXISTS topics (
    id                  TEXT PRIMARY KEY,
    title               TEXT NOT NULL UNIQUE,
    owner_id            TEXT,
    created_at          REAL NOT NULL DEFAULT 0,
    updated_at          REAL NOT NULL DEFAULT 0,
    post_count          INTEGER NOT NULL DEFAULT 0,
    last_digest_post_id INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS posts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id  TEXT NOT NULL,
    time      REAL NOT NULL,
    topic_id  TEXT NOT NULL,
    parent_id INTEGER,
    body      TEXT NOT NULL,
    refs      TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_posts_topic ON posts(topic_id, id);
CREATE INDEX IF NOT EXISTS idx_posts_agent_time ON posts(agent_id, time);

CREATE TRIGGER IF NOT EXISTS posts_append_only_insert
BEFORE INSERT ON posts
WHEN EXISTS (SELECT 1 FROM posts WHERE id = NEW.id)
BEGIN
    SELECT RAISE(ABORT, 'posts are append-only: id exists');
END;

CREATE TRIGGER IF NOT EXISTS posts_append_only_update
BEFORE UPDATE ON posts
BEGIN
    SELECT RAISE(ABORT, 'posts are append-only: update rejected');
END;

CREATE TRIGGER IF NOT EXISTS posts_append_only_delete
BEFORE DELETE ON posts
BEGIN
    SELECT RAISE(ABORT, 'posts are append-only: delete rejected');
END;

CREATE TABLE IF NOT EXISTS subscriptions (
    agent_id   TEXT NOT NULL,
    topic_id   TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (agent_id, topic_id)
);
CREATE INDEX IF NOT EXISTS idx_subscriptions_topic ON subscriptions(topic_id);

CREATE TABLE IF NOT EXISTS digests (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id     TEXT NOT NULL,
    from_post_id INTEGER NOT NULL,
    to_post_id   INTEGER NOT NULL,
    body         TEXT NOT NULL,
    created_at   REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS findings (
    id         TEXT PRIMARY KEY,
    topic_id   TEXT NOT NULL,
    agent_id   TEXT NOT NULL,
    file       TEXT NOT NULL,
    line       INTEGER NOT NULL,
    sink       TEXT NOT NULL,
    call_path  TEXT NOT NULL DEFAULT '[]',
    status     TEXT NOT NULL DEFAULT 'candidate',
    detail     TEXT,
    created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_findings_topic ON findings(topic_id);
CREATE INDEX IF NOT EXISTS idx_findings_status ON findings(status);

CREATE TABLE IF NOT EXISTS poc_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    finding_id   TEXT NOT NULL,
    agent_id     TEXT,
    attempt      INTEGER NOT NULL,
    exit_code    INTEGER NOT NULL,
    success      INTEGER NOT NULL,
    script       TEXT NOT NULL,
    stdout       TEXT NOT NULL DEFAULT '',
    stderr       TEXT NOT NULL DEFAULT '',
    container_id TEXT,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_poc_runs_finding ON poc_runs(finding_id);
