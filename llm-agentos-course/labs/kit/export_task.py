#!/usr/bin/env python3
"""Write a task's fixture files into a directory and print its prompt, to run it in another harness.

    python export_task.py b07_two_hop /tmp/b07        # then open Hermes, Codex CLI, ... in /tmp/b07
    python export_task.py b07_two_hop /tmp/b07 --check "Priya Natarajan"   # score an answer afterwards

With --check, the fixture is left alone and the task's checker is run on the directory and the answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasks import load_tier  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_id")
    ap.add_argument("directory")
    ap.add_argument("--tier", default="basic")
    ap.add_argument("--check", metavar="ANSWER", help="score this final answer instead of writing the fixture")
    args = ap.parse_args(argv)
    task = next((t for t in load_tier(args.tier) if t.id.startswith(args.task_id)), None)
    if task is None:
        print(f"no task {args.task_id!r} in tier {args.tier}", file=sys.stderr)
        return 2
    ws = Path(args.directory)
    if args.check is not None:
        ok = task.check(ws, args.check)
        print("PASS" if ok else "fail")
        return 0 if ok else 1
    if ws.exists() and any(ws.iterdir()):
        print(f"{ws} is not empty; use a fresh directory", file=sys.stderr)
        return 2
    ws.mkdir(parents=True, exist_ok=True)
    task.setup(ws)
    print(task.prompt)
    if task.tools:
        print(f"\n(allowed tools: {', '.join(task.tools)})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
