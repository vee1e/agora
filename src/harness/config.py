"""Configuration.

Every value can be overridden with an environment variable. Secrets are never
hard-coded: the LLM key comes from the environment, and if it is missing we read
it from the machine's opencode auth file.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB = Path("data/harness.db")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:18000/v1"
DEFAULT_LLM_MODEL = "deepseek-v4.1-flash"
DEFAULT_DOCKER_IMAGE = "agent-forum-box:latest"

_OPENCODE_AUTH = Path.home() / ".local/share/opencode/auth.json"


def _read_auth_key(provider: str = "opencode-go") -> str | None:
    try:
        data = json.loads(_OPENCODE_AUTH.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    entry = data.get(provider)
    if not isinstance(entry, dict):
        return None
    key = entry.get("key")
    return key if isinstance(key, str) and key else None


@dataclass(frozen=True)
class Settings:
    db_path: Path = DEFAULT_DB
    data_dir: Path = DEFAULT_DATA_DIR
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_api_key: str = ""
    llm_model: str = DEFAULT_LLM_MODEL
    docker_image: str = DEFAULT_DOCKER_IMAGE

    floor: int = 5
    hard_cap: int = 10
    worker_pool_size: int = 10

    turn_token_budget: int = 20_000
    daily_agent_token_cap: int = 200_000
    global_token_cap: int = 2_000_000

    posts_per_agent_per_hour: int = 20
    digest_every_n_posts: int = 25

    docker_enabled: bool = True
    sandbox_network: str = "none"
    sandbox_user: str = f"{os.getuid()}:{os.getgid()}"
    sandbox_mem_limit: str = "1g"
    sandbox_pids_limit: int = 256
    sandbox_cpu_period: int = 100_000
    sandbox_cpu_quota: int = 100_000
    sandbox_read_only: bool = True
    sandbox_tmpfs_size: str = "64m"
    max_response_bytes: int = 8_000_000
    log_max_bytes: int = 50_000_000

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def workspace_dir(self) -> Path:
        return self.data_dir / "workspaces"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENCODE_GO_API_KEY") or ""
    if not api_key:
        api_key = _read_auth_key() or ""

    return Settings(
        db_path=Path(os.environ.get("HARNESS_DB", str(DEFAULT_DB))),
        data_dir=Path(os.environ.get("HARNESS_DATA_DIR", str(DEFAULT_DATA_DIR))),
        llm_base_url=os.environ.get("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
        llm_api_key=api_key,
        llm_model=os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL),
        docker_image=os.environ.get("HARNESS_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE),
        floor=_env_int("HARNESS_FLOOR", 5),
        hard_cap=_env_int("HARNESS_HARD_CAP", 10),
        worker_pool_size=_env_int("HARNESS_WORKERS", 10),
        turn_token_budget=_env_int("HARNESS_TURN_BUDGET", 20_000),
        daily_agent_token_cap=_env_int("HARNESS_AGENT_DAILY_CAP", 200_000),
        global_token_cap=_env_int("HARNESS_GLOBAL_CAP", 2_000_000),
        posts_per_agent_per_hour=_env_int("HARNESS_POST_RATE", 20),
        digest_every_n_posts=_env_int("HARNESS_DIGEST_EVERY", 25),
        docker_enabled=_env_bool("HARNESS_DOCKER", True),
        sandbox_network=os.environ.get("HARNESS_SANDBOX_NETWORK", "none"),
        sandbox_user=os.environ.get("HARNESS_SANDBOX_USER", f"{os.getuid()}:{os.getgid()}"),
        sandbox_mem_limit=os.environ.get("HARNESS_SANDBOX_MEM", "1g"),
        sandbox_pids_limit=_env_int("HARNESS_SANDBOX_PIDS", 256),
        sandbox_cpu_period=_env_int("HARNESS_SANDBOX_CPU_PERIOD", 100_000),
        sandbox_cpu_quota=_env_int("HARNESS_SANDBOX_CPU_QUOTA", 100_000),
        sandbox_read_only=_env_bool("HARNESS_SANDBOX_READ_ONLY", True),
        sandbox_tmpfs_size=os.environ.get("HARNESS_SANDBOX_TMPFS", "64m"),
        max_response_bytes=_env_int("HARNESS_MAX_RESPONSE_BYTES", 8_000_000),
        log_max_bytes=_env_int("HARNESS_LOG_MAX_BYTES", 50_000_000),
    )
