"""The task suite.

A task is a prompt, a fixture (files written into a fresh workspace before the run), a checker that
decides success from the workspace and the final answer, and a demo script: scripted model turns that
solve the task, so the runner and the graders can be tested offline.

Tiers:

    basic     short tasks over small fixture files (this kit, tasks/basic.py)
    long      40+ step tasks for Labs 3 and 9 (added with Lab 3)
    heldout   kept private for the capstone
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class Task:
    id: str
    tier: str
    prompt: str
    files: dict[str, str]                         # relative path -> content, written before the run
    check: Callable[[Path, str], bool]            # (workspace, final answer) -> success
    demo: list[dict[str, Any]] = field(default_factory=list)  # a scripted solution (see models/scripted.py)

    def setup(self, workspace: Path) -> None:
        for rel, content in self.files.items():
            p = workspace / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")


def load_tier(tier: str) -> list[Task]:
    if tier == "basic":
        from tasks.basic import TASKS
        return list(TASKS)
    raise ValueError(f"unknown tier {tier!r} (available: basic)")
