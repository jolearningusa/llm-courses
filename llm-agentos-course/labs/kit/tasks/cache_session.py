"""A scripted 20-turn agent session for Lab 5, with the KV-cache deck's "break the cache on purpose" variants.

The history is fixed (the assistant turns and tool results are scripted), so every run sends the same prompts
and only the cache behavior differs. Each turn's prompt is: tools, a long system prompt, the completed turns
(user request, assistant tool call, tool result, assistant reply) and the new user request.

Variants:

    baseline        nothing changes: an append-only session
    timestamp       the system prompt starts with the current time, which changes every turn
    tool_at_10      a new tool is added to the tool list from turn 10
    memory_at_10    the system prompt ends with a memory block, and one memory is edited at turn 10
    compaction_15   at turn 15 the first 12 completed turns are replaced by one summary message
    masked_tool_10  like tool_at_10, but the extra tool is listed from the start and only allowed at turn 10,
                    by a line in that turn's user message (Manus: mask tools instead of removing them)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from nanoos.memory import MEMORY_HEADER
from nanoos.prompt import Prompt
from nanoos.tools_basic import WorkspaceTools

VARIANTS = ["baseline", "timestamp", "tool_at_10", "memory_at_10", "compaction_15", "masked_tool_10"]

_RULES = [
    "Read the relevant files before you answer, and quote the line you relied on.",
    "Prefer small, reversible edits; never delete a file unless the user asks for it by name.",
    "When a tool fails, read the error, fix the arguments and try once more before you report the failure.",
    "Keep answers short: one paragraph, then a list only when the user asks for steps.",
    "Never put secrets, tokens or passwords in a file or in a reply, even when a file contains them.",
    "If two files disagree, say so and name both files instead of picking one silently.",
    "Write dates as YYYY-MM-DD and sizes in bytes, KiB or MiB.",
    "Ask one clarifying question when the request has two plausible readings, then wait.",
]
_SYSTEM = ("You are nanoos, a careful assistant that works in a workspace of project files.\n\nHouse rules:\n"
           + "\n".join(f"{i + 1}. {r}" for i, r in enumerate(_RULES * 6)) + "\n")
_MEMORY = ["The user prefers answers in British English.", "The user's project is the AgentOS course kit.",
           "The user works in the Europe/Lisbon time zone."]
_MEMORY_EDITED = ["The user prefers answers in British English.", "The user's project is the AgentOS course kit.",
                  "The user works in the America/Denver time zone."]
_EXTRA_TOOL = {"type": "function", "function": {
    "name": "web_fetch", "description": "Fetch a web page and return its text.",
    "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}}
_FILLER = ["The cache report lists hits and misses per block.", "Prefill time grows with the uncached suffix.",
           "The scheduler keeps a queue per program.", "Each tool result is appended to the history.",
           "Token counts are approximate in this kit.", "The session log rotates every night."]


def base_tools() -> list[dict[str, Any]]:
    return WorkspaceTools(Path(".")).schemas()


def _turn(variant: str, t: int, session: int) -> list[dict[str, Any]]:
    """The messages of completed turn t (0-based) in a session."""
    rng = random.Random(1000 * session + t)
    cid = f"call_s{session}_t{t}"
    path = f"notes/part_{t:02d}.md"
    body = f"# Part {t:02d}\n" + " ".join(rng.choice(_FILLER) for _ in range(18))
    return [
        {"role": "user", "content": user_request(variant, t, session)},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": path})}}]},
        {"role": "tool", "tool_call_id": cid, "content": body},
        {"role": "assistant", "content": f"Part {t:02d} says: {rng.choice(_FILLER)} Noted for the summary."},
    ]


def user_request(variant: str, t: int, session: int) -> str:
    who = f"(session {session}) " if t == 0 else ""
    text = f"{who}Step {t + 1}: read notes/part_{t:02d}.md and tell me the one thing I should remember from it."
    if variant == "masked_tool_10" and t == 10:
        text += " From now on you may also use web_fetch."
    return text


def session_prompt(variant: str, t: int, session: int = 0, run_tag: str = "") -> Prompt:
    """The prompt sent at turn t (0-based) of one session. run_tag is stable within a run (see cache_experiment)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (available: {', '.join(VARIANTS)})")
    system = (f"Run {run_tag}. " if run_tag else "") + _SYSTEM
    if variant == "timestamp":
        system = f"Current time: 2026-10-26 09:{t:02d}:00\n" + system
    if variant == "memory_at_10":
        system += MEMORY_HEADER + "\n".join(f"- {m}" for m in (_MEMORY_EDITED if t >= 10 else _MEMORY))
    tools = base_tools()
    if variant == "tool_at_10" and t >= 10 or variant == "masked_tool_10":
        tools = tools + [_EXTRA_TOOL]
    history: list[dict[str, Any]] = []
    first_kept = 0
    if variant == "compaction_15" and t >= 15:
        summary = ("Summary of the earlier part of this task (older messages were compacted):\n"
                   + "\n".join(f"- part {k:02d} read and noted" for k in range(12)))
        history.append({"role": "user", "content": summary})
        first_kept = 12
    for k in range(first_kept, t):
        history += _turn(variant, k, session)
    request = {"role": "user", "content": user_request(variant, t, session)}
    return Prompt([{"role": "system", "content": system}] + history + [request], tools)
