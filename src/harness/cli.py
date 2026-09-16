"""Command-line entry point for the harness.

Runs as ``python -m harness.cli`` and as the ``harness`` console script.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from . import models as m
from .budget import BudgetExceeded, BudgetGuard
from .config import Settings, load_settings
from .containers import ContainerManager
from .council.agents import CouncilResult
from .council.executor import build_combined_tools
from .council.task import run_council
from .db import Database
from .forum import Forum, InvalidPost, RateLimited
from .llm import LLMBudgetError, LLMClient, LLMError
from .logging_ import EventLogger
from .mailbox import Mailbox
from .orchestrator import SpawnRefused, build_orchestrator
from .summarizer import summarize_topic

DEFAULT_COUNCIL_TOPIC = "council"

ROLE_CHOICES = [role.value for role in m.AgentRole]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness", description="Agent forum harness control CLI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="create the database schema")

    serve = subparsers.add_parser("serve", help="run the orchestrator loop")
    serve.add_argument("--interval", type=float, default=1.0)
    serve.add_argument("--checkout", default=None)

    subparsers.add_parser("status", help="print agent and budget status")

    spawn = subparsers.add_parser("spawn", help="spawn one agent")
    spawn.add_argument("--role", required=True, choices=ROLE_CHOICES)

    post = subparsers.add_parser("post", help="append a forum post")
    post.add_argument("--topic", required=True)
    post.add_argument("--body", required=True)
    post.add_argument("--agent", required=True)

    subscribe = subparsers.add_parser("subscribe", help="subscribe an agent to a topic")
    subscribe.add_argument("--agent", required=True)
    subscribe.add_argument("--topic", required=True)

    subparsers.add_parser("topics", help="list topics with owner and post count")

    digest = subparsers.add_parser("digest", help="force a topic digest")
    digest.add_argument("--topic", required=True)

    council = subparsers.add_parser("council", help="run the council audit task")
    council_sub = council.add_subparsers(dest="council_command", required=True)

    council_run = council_sub.add_parser("run", help="run the full council pass")
    council_run.add_argument("--checkout", required=True)
    council_run.add_argument("--topic", default=DEFAULT_COUNCIL_TOPIC)
    council_run.add_argument("--no-docker", action="store_true")
    council_run.add_argument(
        "--max-review",
        type=int,
        default=None,
        help="cap how many candidates go through killer/judge/exploit",
    )
    council_run.add_argument(
        "--max-pocs",
        type=int,
        default=None,
        help="cap how many survivors get a proof-of-concept",
    )

    council_tools = council_sub.add_parser(
        "tools", help="print the council tool schemas"
    )
    council_tools.add_argument("--checkout", default=".")

    return parser


async def _cmd_init(settings: Settings) -> int:
    settings.ensure_dirs()
    db = Database(settings.db_path)
    await db.connect()
    try:
        await db.init_schema()
    finally:
        await db.close()
    print(f"initialized {settings.db_path}")
    return 0


async def _cmd_status(settings: Settings) -> int:
    db = Database(settings.db_path)
    await db.connect()
    try:
        await db.init_schema()
        budget = BudgetGuard(
            db,
            settings,
            logger=EventLogger(settings.log_dir, max_bytes=settings.log_max_bytes),
        )
        print("agents by state:")
        for state in m.AgentState:
            print(f"  {state.value}: {await db.count_agents(state)}")

        running = await db.count_agents(m.AgentState.RUNNING)
        paused = await db.count_agents(m.AgentState.PAUSED)
        print(f"running: {running}")
        print(f"paused: {paused}")

        agent_tokens = 0
        for agent in await db.list_agents():
            agent_tokens += await budget.agent_used(agent.id)
        print(f"tokens today (agents): {agent_tokens}")
        print(
            f"tokens today (global): {await budget.global_used()}"
            f" / {settings.global_token_cap}"
        )
        return 0
    finally:
        await db.close()


async def _cmd_spawn(settings: Settings, role_name: str) -> int:
    orchestrator = await build_orchestrator(settings)
    try:
        role = m.AgentRole(role_name)
        agent = await orchestrator.spawn_agent(role)
        print(
            f"spawned {agent.id} role={agent.role.value} state={agent.state.value}"
        )
        return 0
    except BudgetExceeded as exc:
        print(f"budget exceeded: {exc}", file=sys.stderr)
        return 1
    except SpawnRefused as exc:
        print(f"spawn refused: {exc}", file=sys.stderr)
        return 1
    finally:
        await orchestrator.close()


async def _cmd_post(
    settings: Settings, topic_title: str, body: str, agent_id: str
) -> int:
    db = Database(settings.db_path)
    await db.connect()
    logger = EventLogger(settings.log_dir, max_bytes=settings.log_max_bytes)
    try:
        await db.init_schema()
        forum = Forum(db, Mailbox(db, logger=logger), settings, logger=logger)
        try:
            topic = await forum.ensure_topic(topic_title)
            result = await forum.post(agent_id, topic.id, body)
        except RateLimited as exc:
            print(f"rate limited: {exc}", file=sys.stderr)
            return 1
        except InvalidPost as exc:
            print(f"invalid post: {exc}", file=sys.stderr)
            return 1
        print(result.post.id)
        return 0
    finally:
        await db.close()


async def _cmd_subscribe(
    settings: Settings, agent_id: str, topic_title: str
) -> int:
    db = Database(settings.db_path)
    await db.connect()
    logger = EventLogger(settings.log_dir, max_bytes=settings.log_max_bytes)
    try:
        await db.init_schema()
        forum = Forum(db, Mailbox(db, logger=logger), settings, logger=logger)
        try:
            topic = await forum.ensure_topic(topic_title)
            await forum.subscribe(agent_id, topic.id)
        except InvalidPost as exc:
            print(f"invalid topic: {exc}", file=sys.stderr)
            return 1
        print(f"subscribed {agent_id} to {topic.id}")
        return 0
    finally:
        await db.close()


async def _cmd_topics(settings: Settings) -> int:
    db = Database(settings.db_path)
    await db.connect()
    try:
        await db.init_schema()
        for topic in await db.list_topics():
            owner = topic.owner_id or "-"
            print(
                f"{topic.id}\towner={owner}\tposts={topic.post_count}\t{topic.title}"
            )
        return 0
    finally:
        await db.close()


async def _cmd_digest(settings: Settings, topic_title: str) -> int:
    db = Database(settings.db_path)
    await db.connect()
    logger = EventLogger(settings.log_dir, max_bytes=settings.log_max_bytes)
    llm = LLMClient(
        settings, logger=logger, budget=BudgetGuard(db, settings, logger=logger)
    )
    try:
        await db.init_schema()
        topic = await db.topic_by_title(topic_title)
        if topic is None:
            print(f"unknown topic: {topic_title}", file=sys.stderr)
            return 1
        try:
            result = await summarize_topic(
                db,
                llm,
                topic.id,
                force=True,
                logger=logger,
                agent_id="summarizer",
            )
        except LLMBudgetError as exc:
            print(f"budget exceeded: {exc}", file=sys.stderr)
            return 1
        if result is None:
            print("no posts to digest")
            return 0
        print(result["body"])
        return 0
    finally:
        await llm.aclose()
        await db.close()


def _print_council_result(result: CouncilResult) -> None:
    pocs_run = getattr(result, "pocs_run", 0)
    pocs_succeeded = getattr(result, "pocs_succeeded", 0)
    pocs_written = getattr(result, "pocs_written", 0)
    reviewed = getattr(result, "reviewed", result.candidates)
    pending = getattr(result, "pending", 0)
    extra = f" reviewed={reviewed} pending={pending}"
    if pocs_written:
        extra += f" written={pocs_written}"
    print(
        f"council {result.topic_id}: candidates={result.candidates} "
        f"survived={result.survived} killed={result.killed} "
        f"exploited={result.exploited} pocs={pocs_run} ok={pocs_succeeded}"
        f"{extra} precision={result.precision:.2f}"
    )
    for finding in result.findings:
        print(
            f"  {finding.id} {finding.status.value} "
            f"{finding.file}:{finding.line} sink={finding.sink}"
        )
    for note in result.notes:
        print(f"  note: {note}")


async def _mark_agents_dead(db: Database, agent_ids: list[str]) -> None:
    for agent_id in agent_ids:
        await db.set_agent_state(agent_id, m.AgentState.DEAD)


async def _cmd_council_run(
    settings: Settings,
    checkout: str,
    topic_title: str,
    no_docker: bool = False,
    max_reviewed: int | None = None,
    max_pocs: int | None = None,
) -> int:
    settings.ensure_dirs()
    db = Database(settings.db_path)
    await db.connect()
    logger = EventLogger(settings.log_dir, max_bytes=settings.log_max_bytes)
    llm = LLMClient(
        settings, logger=logger, budget=BudgetGuard(db, settings, logger=logger)
    )
    containers = None if no_docker else ContainerManager(settings, db, logger)
    created: list[str] = []
    try:
        await db.init_schema()
        forum = Forum(db, Mailbox(db, logger=logger), settings, logger=logger)
        finder = await db.create_agent(m.AgentRole.FINDER)
        created.append(finder.id)
        killer = await db.create_agent(m.AgentRole.KILLER)
        created.append(killer.id)
        judge = await db.create_agent(m.AgentRole.JUDGE)
        created.append(judge.id)
        exploit = await db.create_agent(m.AgentRole.EXPLOIT)
        created.append(exploit.id)
        topic = await forum.ensure_topic(topic_title, owner_id=finder.id)
        tools = build_combined_tools(checkout)
        try:
            result = await run_council(
                llm,
                tools,
                db,
                forum,
                topic.id,
                checkout,
                finder.id,
                killer.id,
                judge.id,
                exploit.id,
                containers=containers,
                logger=logger,
                max_reviewed=max_reviewed,
                max_pocs=max_pocs,
            )
        except (BudgetExceeded, LLMBudgetError) as exc:
            print(f"budget exceeded: {exc}", file=sys.stderr)
            return 1
        except LLMError as exc:
            print(f"council run failed: {exc}", file=sys.stderr)
            return 1
        _print_council_result(result)
        return 0
    finally:
        try:
            await _mark_agents_dead(db, created)
        except Exception:
            pass
        if containers is not None:
            try:
                await containers.aclose()
            except Exception:
                pass
        await llm.aclose()
        await db.close()


async def _cmd_council_tools(checkout: str) -> int:
    tools = build_combined_tools(checkout)
    for spec in tools.specs():
        print(json.dumps(spec))
    return 0


async def _cmd_serve(
    settings: Settings, interval: float, checkout: str | None = None
) -> int:
    orchestrator = await build_orchestrator(settings, checkout=checkout)
    orchestrator.interval = interval
    try:
        await orchestrator.start()
    except (BudgetExceeded, SpawnRefused) as exc:
        print(f"cannot serve: {exc}", file=sys.stderr)
        await orchestrator.aclose()
        return 1
    print("orchestrator running; press Ctrl+C to stop", flush=True)
    try:
        while not orchestrator.stopping:
            await asyncio.sleep(1.0)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await orchestrator.stop()
        await orchestrator.aclose()
    return 0


async def _dispatch(args: argparse.Namespace, settings: Settings) -> int:
    command: str = args.command
    if command == "init":
        return await _cmd_init(settings)
    if command == "status":
        return await _cmd_status(settings)
    if command == "spawn":
        return await _cmd_spawn(settings, args.role)
    if command == "post":
        return await _cmd_post(settings, args.topic, args.body, args.agent)
    if command == "subscribe":
        return await _cmd_subscribe(settings, args.agent, args.topic)
    if command == "topics":
        return await _cmd_topics(settings)
    if command == "digest":
        return await _cmd_digest(settings, args.topic)
    if command == "serve":
        return await _cmd_serve(settings, args.interval, args.checkout)
    if command == "council":
        if args.council_command == "tools":
            return await _cmd_council_tools(args.checkout)
        if args.council_command == "run":
            return await _cmd_council_run(
                settings, args.checkout, args.topic, args.no_docker,
                args.max_review, args.max_pocs
            )
        print(f"unknown council command: {args.council_command}", file=sys.stderr)
        return 2
    print(f"unknown command: {command}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()
    try:
        return asyncio.run(_dispatch(args, settings))
    except (BudgetExceeded, LLMBudgetError) as exc:
        print(f"budget exceeded: {exc}", file=sys.stderr)
        return 1
    except LLMError as exc:
        print(f"llm error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
