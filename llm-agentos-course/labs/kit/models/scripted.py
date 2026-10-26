"""A scripted model: replays canned assistant turns instead of calling an LLM.

The autograder uses it so that tests check the harness, not the model: they are free, offline and give
the same result every time. A script is a list of turns; each turn is a dict with

    content     the assistant text (optional)
    tool_calls  [{"name": ..., "arguments": {...}}, ...] (optional)
    usage       {"input_tokens": .., "cached_input_tokens": .., "output_tokens": ..} (optional; estimated
                from the prompt and reply when missing)
    raise       an exception message: the call raises ModelError instead of answering (to test retries)
    delay_s     sleep this long before answering (to test timeouts and interrupts)

Scripts can live in JSON files (see tasks/basic.py for the demo scripts that solve each task):

    model = ScriptedModel.from_file("script.json")

Every call is recorded in `model.calls` (the messages and tool schemas it was given), so a test can
assert on what the harness sent.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from nanoos.types import ModelResponse, ToolCall, Usage, estimate_tokens


class ModelError(Exception):
    """Raised by a model client when a call fails (rate limit, timeout, server error, ...)."""


class ScriptedModel:
    def __init__(self, script: list[dict[str, Any]], name: str = "scripted"):
        self.script = list(script)
        self.name = name
        self.calls: list[dict[str, Any]] = []
        self._next = 0

    @classmethod
    def from_file(cls, path: str | Path) -> "ScriptedModel":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), name=f"scripted:{Path(path).name}")

    @property
    def exhausted(self) -> bool:
        return self._next >= len(self.script)

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        self.calls.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        if self.exhausted:
            return ModelResponse(content="(script ended)", usage=self._estimate(messages, tools, "(script ended)"))
        turn = self.script[self._next]
        self._next += 1
        if turn.get("delay_s"):
            time.sleep(float(turn["delay_s"]))
        if "raise" in turn:
            raise ModelError(turn["raise"])
        calls = [ToolCall(id=f"call_{self._next}_{i}", name=c["name"], arguments=dict(c.get("arguments", {})))
                 for i, c in enumerate(turn.get("tool_calls", []))]
        content = turn.get("content")
        if "usage" in turn:
            usage = Usage(**turn["usage"])
        else:
            usage = self._estimate(messages, tools, (content or "") + json.dumps(turn.get("tool_calls", [])))
        return ModelResponse(content=content, tool_calls=calls, usage=usage)

    @staticmethod
    def _estimate(messages: list[dict[str, Any]], tools: list[dict[str, Any]], reply: str) -> Usage:
        prompt = json.dumps(messages) + json.dumps(tools)
        return Usage(input_tokens=estimate_tokens(prompt), output_tokens=estimate_tokens(reply))
