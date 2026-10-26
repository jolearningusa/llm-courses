"""Shared data types for nanoos.

Messages use the OpenAI chat format as plain dicts, so any OpenAI-compatible endpoint (a hosted API or
vLLM) can take them unchanged:

    {"role": "system" | "user" | "assistant" | "tool", "content": str, ...}

An assistant message that calls tools carries "tool_calls"; each tool result is a "tool" message with the
matching "tool_call_id".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_message_part(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": json.dumps(self.arguments)}}


@dataclass
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0  # part of input_tokens served from a prefix cache
    output_tokens: int = 0

    def __iadd__(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.output_tokens += other.output_tokens
        return self


@dataclass
class ModelResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def to_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [c.to_message_part() for c in self.tool_calls]
        return msg


class Model(Protocol):
    """Anything with this method can drive a loop: the scripted model, an API client, a test double."""

    name: str

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse: ...


@dataclass
class RunResult:
    answer: str
    steps: int
    usage: Usage
    tool_errors: int = 0
    approvals: int = 0
    stopped_by: str = "answer"  # "answer" | "budget" | "error"
    messages: list[dict[str, Any]] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token). Good enough for budgets and breakdowns."""
    return max(1, (len(text) + 3) // 4) if text else 0
