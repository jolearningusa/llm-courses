"""Lab 3: context engines for nanoos.

A context engine decides what the model sees on each call. It sits between the loop and the model:

    model = ContextManagedModel(model, ThresholdCompaction(budget_tokens=6000, summarizer=...))

so it works with any loop (the baseline or your Lab 1 loop), and run_suite.py --context threshold|clearing
sets it up for you. The loop keeps the full history; the engine returns the view to send.

Implement the two engines marked TODO (`make grade LAB=3`). They are the context-manager deck's "compress"
strategies: summary compaction, and the lighter tool-result clearing. Everything else here is provided.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from nanoos.types import Model, ModelResponse, Usage, estimate_tokens

SUMMARY_PREFIX = "Summary of the earlier part of this task (older messages were compacted):\n"
PINNED_HEADER = "\nPinned notes (copied verbatim):\n"
PIN_TAG = "[pin]"
CLEARED = "[tool result cleared to save context: {chars} characters. Call the tool again if you need it.]"

Summarizer = Callable[["str | None", list[dict[str, Any]]], str]


def message_text(m: dict[str, Any]) -> str:
    text = m.get("content") or ""
    if m.get("tool_calls"):
        text += json.dumps(m["tool_calls"])
    return text


def count_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_tokens(message_text(m)) for m in messages)


class ContextEngine:
    name = "none"
    compactions = 0

    def prepare(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return messages


class ThresholdCompaction(ContextEngine):
    """Summarize the middle of the history when the prompt grows past a token budget.

    prepare(messages) returns the view to send:

    - Before any compaction the view is `messages` itself. After compaction k, it is
          messages[:2] + [summary message] + messages[upto:]
      where messages[:2] are the system prompt and the task, and `upto` is where compaction k stopped.
      The summary message is {"role": "user", "content": SUMMARY_PREFIX + summary [+ pinned notes]}.
    - If count_tokens(view) <= budget_tokens, return the view and do nothing else. In particular, do not call
      the summarizer again while the compacted view still fits: the summary stays byte-identical between
      compactions, which keeps the prompt prefix cacheable (Lab 5).
    - Otherwise compact once more: the messages from the current `upto` (2 before the first compaction) up to
      a cut point are summarized, where the cut starts at len(messages) - keep_last and moves earlier while
      messages[cut] is a "tool" message, so a tool result is never separated from the assistant message that
      called it. If the cut is not after `upto`, there is nothing to compact: return the view.
      Call summarizer(previous_summary_content_or_None, chunk) exactly once, count the compaction in
      self.compactions, and return the new view. (One compaction per call is enough, even if the new view is
      still over budget.)
    - Pinned notes: every line of a compacted message's content that contains PIN_TAG is kept verbatim, in the
      order first seen and without duplicates, and appended to every later summary after PINNED_HEADER.
    - Never modify the input list or its dicts.
    """

    name = "threshold"

    def __init__(self, budget_tokens: int, summarizer: Summarizer, keep_last: int = 6):
        self.budget_tokens = budget_tokens
        self.summarizer = summarizer
        self.keep_last = keep_last
        self.compactions = 0
        # TODO(lab 3): the state you need between calls (the summary message, upto, pinned lines)

    def prepare(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # TODO(lab 3)
        raise NotImplementedError("Lab 3: implement ThresholdCompaction.prepare")


class ToolResultClearing(ContextEngine):
    """Replace the content of old tool results with a short placeholder; keep everything else.

    prepare(messages) returns a list of the same length and order. The last `keep_last_results` tool messages
    are returned as they are; every older tool message whose content is longer than `min_chars` becomes a copy
    with content CLEARED.format(chars=<length of the original content>), keeping its "tool_call_id". Other
    messages are returned unchanged. Count one compaction for each call that cleared at least one result.
    Never modify the input list or its dicts.
    """

    name = "clearing"

    def __init__(self, keep_last_results: int = 3, min_chars: int = 200):
        self.keep_last_results = keep_last_results
        self.min_chars = min_chars
        self.compactions = 0

    def prepare(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # TODO(lab 3)
        raise NotImplementedError("Lab 3: implement ToolResultClearing.prepare")


class ContextManagedModel:
    def __init__(self, model: Model, engine: ContextEngine):
        self.model = model
        self.engine = engine
        self.name = f"{model.name}+{engine.name}"
        self.sent_tokens: list[int] = []

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        prepared = self.engine.prepare(messages)
        self.sent_tokens.append(count_tokens(prepared))
        return self.model.complete(prepared, tools)


def heuristic_summarizer(previous: str | None, chunk: list[dict[str, Any]]) -> str:
    lines = [previous.split("\n", 1)[1].split(PINNED_HEADER)[0]] if previous else []
    for m in chunk:
        if m["role"] == "assistant":
            for c in m.get("tool_calls") or []:
                lines.append(f"- called {c['function']['name']} {c['function']['arguments']}")
            if m.get("content"):
                lines.append(f"- said: {m['content'][:200]}")
        elif m["role"] == "tool":
            lines.append(f"  result: {(m.get('content') or '')[:120]!r}")
        elif m["role"] == "user":
            lines.append(f"- user: {(m.get('content') or '')[:200]}")
    return "\n".join(lines)


class ModelSummarizer:
    PROMPT = ("Summarize the agent's work below so it can continue the task without the original messages. "
              "Keep every instruction it was given, every file it read with the facts it found (numbers, names, "
              "codes), and what is left to do. Be concise.")

    def __init__(self, model: Model):
        self.model = model
        self.usage = Usage()

    def __call__(self, previous: str | None, chunk: list[dict[str, Any]]) -> str:
        body = (f"Earlier summary:\n{previous}\n\n" if previous else "") + "\n".join(
            f"[{m['role']}] {message_text(m)[:4000]}" for m in chunk)
        resp = self.model.complete([{"role": "system", "content": self.PROMPT}, {"role": "user", "content": body}], [])
        self.usage += resp.usage
        return resp.content or ""
