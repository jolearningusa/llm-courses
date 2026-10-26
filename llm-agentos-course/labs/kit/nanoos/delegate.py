"""Lab 9: sub-agents (the multi-agent deck): fork a child loop with a fresh context, wait for its summary.

A parent agent gets two extra tools. Each child is a new run of the loop: it sees only its brief (never the
parent's history), gets a restricted tool set, and returns only its final answer, which is all that enters the
parent's context. Limits hold the tree in check, the way ulimit does for processes:

    max_depth         how deep children may nest (1: children cannot delegate; OpenClaw's maxSpawnDepth)
    max_children      children per session, across the whole tree (OpenClaw's maxChildrenPerAgent)
    max_concurrency   children running at once in one delegate_many call
    child_max_steps   the child's step budget; child_timeout_s its wall-clock limit

    d = Delegator(WorkspaceTools(ws), model_factory=lambda brief, depth: worker_model)
    result = loop.run(parent_model, task, d)        # d has the parent's tools plus delegate, delegate_many
    d.records                                       # one ChildRecord per child: status, steps, usage, time

Implement the two parts marked TODO (`make grade LAB=9`): `Delegator.spawn` and `Delegator.delegate_many`.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from nanoos import baseline
from nanoos.types import Model, Usage

DELEGATE_TOOLS = ("delegate", "delegate_many")


@dataclass
class ChildRecord:
    id: int
    depth: int
    brief: str
    status: str = "running"          # ok, error (the child failed or ran out of steps), timeout
    answer: str = ""
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    duration_s: float = 0.0
    tools: list[str] = field(default_factory=list)


class RestrictedTools:
    """Only the named tools of another tool set (provided)."""

    def __init__(self, tools: Any, allowed: list[str]):
        self.tools, self.allowed = tools, list(allowed)

    def schemas(self) -> list[dict[str, Any]]:
        return [s for s in self.tools.schemas() if s["function"]["name"] in self.allowed]

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        if name not in self.allowed:
            return f"ERROR: tool {name!r} is not available here. Available: {', '.join(self.allowed)}.", True
        return self.tools.call(name, arguments)


class _Shared:
    """State shared by every Delegator in one tree: records, counters and a lock (provided)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.records: list[ChildRecord] = []
        self.active = 0
        self.peak = 0


_TOOLS_PARAM = {"type": "array", "items": {"type": "string"},
                "description": "optional: the names of the tools the worker may use (default: all of yours "
                               "except delegation)"}
SCHEMAS = [
    {"type": "function", "function": {
        "name": "delegate",
        "description": "Give one self-contained sub-task to a worker agent. The worker starts fresh: it sees only "
                       "the brief, not this conversation, so include every fact and file path it needs and say "
                       "what to reply with. Returns the worker's final answer.",
        "parameters": {"type": "object", "properties": {
            "brief": {"type": "string", "description": "the complete task for the worker"},
            "tools": _TOOLS_PARAM}, "required": ["brief"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "delegate_many",
        "description": "Run several independent sub-tasks in parallel, one worker each. Use it when the parts do "
                       "not depend on each other. Returns every worker's answer, numbered in the order given.",
        "parameters": {"type": "object", "properties": {
            "briefs": {"type": "array", "items": {"type": "string"},
                       "description": "one complete, self-contained task per worker"},
            "tools": _TOOLS_PARAM}, "required": ["briefs"], "additionalProperties": False}}},
]


class Delegator:
    def __init__(self, tools: Any, model_factory: Callable[[str, int], Model],
                 run: Callable[..., Any] = baseline.run, *, depth: int = 0, max_depth: int = 1,
                 max_children: int = 8, max_concurrency: int = 4, child_max_steps: int = 20,
                 child_timeout_s: float = 120.0, max_summary_chars: int = 2000, _shared: _Shared | None = None):
        self.tools = tools
        self.model_factory = model_factory        # (brief, child depth) -> the model the child runs on
        self.run = run                            # the loop: run(model, prompt, tools, tracer, max_steps=...)
        self.depth = depth
        self.max_depth = max_depth
        self.max_children = max_children
        self.max_concurrency = max_concurrency
        self.child_max_steps = child_max_steps
        self.child_timeout_s = child_timeout_s
        self.max_summary_chars = max_summary_chars
        self.shared = _shared or _Shared()

    # --- provided -------------------------------------------------------------------------------------------
    @property
    def records(self) -> list[ChildRecord]:
        return self.shared.records

    @property
    def peak_concurrency(self) -> int:
        return self.shared.peak

    def children_usage(self) -> Usage:
        total = Usage()
        for r in self.records:
            total += r.usage
        return total

    def base_tool_names(self) -> list[str]:
        return [s["function"]["name"] for s in self.tools.schemas()]

    def schemas(self) -> list[dict[str, Any]]:
        return self.tools.schemas() + (SCHEMAS if self.depth < self.max_depth else [])

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        if name not in DELEGATE_TOOLS:
            return self.tools.call(name, arguments)
        if not isinstance(arguments, dict):
            return "ERROR: the arguments must be a JSON object.", True
        names = arguments.get("tools")
        if names is not None and (not isinstance(names, list) or not all(isinstance(n, str) for n in names)):
            return "ERROR: tools must be a list of tool names.", True
        if name == "delegate":
            brief = arguments.get("brief")
            if not isinstance(brief, str) or not brief.strip():
                return "ERROR: delegate needs a non-empty brief.", True
            return self.spawn(brief, names)
        briefs = arguments.get("briefs")
        if not isinstance(briefs, list) or not briefs or not all(isinstance(b, str) and b.strip() for b in briefs):
            return "ERROR: delegate_many needs a non-empty list of briefs.", True
        return self.delegate_many(briefs, names)

    def _child_tools(self, names: list[str]) -> Any:
        """The child's tool set: the named tools, plus delegation when it may nest further (provided)."""
        restricted = RestrictedTools(self.tools, names)
        if self.depth + 1 < self.max_depth:
            return Delegator(restricted, self.model_factory, self.run, depth=self.depth + 1,
                             max_depth=self.max_depth, max_children=self.max_children,
                             max_concurrency=self.max_concurrency, child_max_steps=self.child_max_steps,
                             child_timeout_s=self.child_timeout_s, max_summary_chars=self.max_summary_chars,
                             _shared=self.shared)
        return restricted

    # --- TODO -----------------------------------------------------------------------------------------------
    def spawn(self, brief: str, tool_names: list[str] | None = None) -> tuple[str, bool]:
        """Run one child to completion and return (summary, is_error). Never raises, never hangs.

        Check first, returning ("ERROR: ...", True) without starting a child:
        - self.depth >= self.max_depth: delegation is not allowed at this depth.
        - tool_names (default: every name in self.base_tool_names()) must all be among self.base_tool_names();
          otherwise name the unknown ones and list the available ones.
        - Holding self.shared.lock (children run in parallel): if len(self.records) >= self.max_children, the
          budget is spent (say how many ran). Otherwise create a ChildRecord (id = len(records) + 1,
          depth = self.depth + 1, the brief and tool names), append it, add one to shared.active and update
          shared.peak.

        Run self.run(self.model_factory(brief, depth + 1), brief, self._child_tools(names), None,
        max_steps=self.child_max_steps) in a daemon thread and wait at most self.child_timeout_s. The child
        sees nothing but its brief. Then fill in the record (steps, usage, duration_s) and decrement
        shared.active (also when the child failed or timed out). Outcomes:

        - Still running after the timeout: status "timeout", ("ERROR: worker <id> timed out after <t>s", True).
        - It raised: status "error", ("ERROR: worker <id> failed: <ExceptionType>: <message>", True).
        - It stopped without an answer (result.stopped_by != "answer"): status "error",
          ("ERROR: worker <id> stopped (<stopped_by>) after <steps> steps without an answer", True).
        - Otherwise status "ok" and (answer, False), the answer cut to self.max_summary_chars characters with
          " [summary truncated]" appended when it was longer. Store the returned text in record.answer.
        """
        # TODO(lab 9)
        raise NotImplementedError("Lab 9: implement Delegator.spawn")

    def delegate_many(self, briefs: list[str], tool_names: list[str] | None = None) -> tuple[str, bool]:
        """Run spawn for every brief, at most self.max_concurrency at a time, and combine the results.

        Use a thread pool with min(self.max_concurrency, len(briefs)) workers. The output is one block per
        brief, in the order given (not the order they finished): "[<i>] <summary>" with i counting from 1,
        blocks joined with "\\n\\n". is_error is True only when every child failed.
        """
        # TODO(lab 9)
        raise NotImplementedError("Lab 9: implement Delegator.delegate_many")
