import os
import subprocess

SAFE_COMMANDS = {"status", "version", "help"}


def run_job(job: str) -> int:
    return subprocess.call("process " + job, shell=True)


def run_safe(job: str) -> int:
    if job not in SAFE_COMMANDS:
        return 2
    return os.system("process status")


def evaluate(event: dict) -> str:
    expr = event.get("expr", "")
    return str(eval(expr))
