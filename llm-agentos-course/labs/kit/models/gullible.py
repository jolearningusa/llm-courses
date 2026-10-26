"""A gullible scripted model: it follows a benign script, but obeys injected instructions it reads.

It stands in for a model with no injection resistance, so the Lab 6 defenses can be tested offline. After each
tool result it looks for a machine-readable injection ([[steps: [...]]], see tasks/redteam/__init__.py). When
it finds one it has not obeyed before, it makes those tool calls next, one per turn, with "$LAST" replaced by
the previous tool result, and then goes back to its script. A blocked or failed step does not stop it.
"""

from __future__ import annotations

import json
from typing import Any

from nanoos.types import ModelResponse, ToolCall, estimate_tokens
from tasks.redteam import STEPS_RE


class GullibleModel:
    def __init__(self, script: list[dict[str, Any]], name: str = "gullible"):
        self.script = list(script)
        self.name = name
        self.obeyed: set[str] = set()
        self.queue: list[dict[str, Any]] = []
        self.injected_calls = 0
        self._next = 0
        self._n = 0

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        recent = []
        for m in reversed(messages):
            if m["role"] != "tool":
                break
            recent.append(m.get("content") or "")
        for text in reversed(recent):
            for match in STEPS_RE.findall(text):
                if match not in self.obeyed:
                    self.obeyed.add(match)
                    self.queue += json.loads(match)
        last = recent[0] if recent else ""
        self._n += 1
        usage_in = estimate_tokens(json.dumps(messages))
        if self.queue:
            step = self.queue.pop(0)
            args = json.loads(json.dumps(step["arguments"]).replace("$LAST", json.dumps(last.strip())[1:-1]))
            self.injected_calls += 1
            resp = ModelResponse(None, [ToolCall(f"g{self._n}", step["name"], args)])
        elif self._next < len(self.script):
            turn = self.script[self._next]
            self._next += 1
            calls = [ToolCall(f"g{self._n}_{i}", c["name"], dict(c.get("arguments", {})))
                     for i, c in enumerate(turn.get("tool_calls", []))]
            resp = ModelResponse(turn.get("content"), calls)
        else:
            resp = ModelResponse("(script ended)")
        resp.usage.input_tokens = usage_in
        resp.usage.output_tokens = estimate_tokens((resp.content or "") + str([c.arguments for c in resp.tool_calls]))
        return resp
