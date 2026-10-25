"""
Lab 1A: a Hermes Agent MemoryProvider and ContextEngine plugin template.

API reference (checked Oct 2026):
  - Memory providers: https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin
  - Context engines:  https://hermes-agent.nousresearch.com/docs/developer-guide/context-engine-plugin

Install as a user plugin (one plugin per directory):

    ~/.hermes/plugins/course-graph-mem/
    ├── __init__.py   <- copy this file here
    └── plugin.yaml   <- see PLUGIN_YAML below

Then activate in ~/.hermes/config.yaml (plugins are never auto-activated):

    memory:
      provider: course-graph-mem
    context:
      engine: course-threshold   # optional: also swap the context engine

Or ship it as a package and register the entry point in pyproject.toml:

    [project.entry-points."hermes_agent.memory_providers"]
    course-graph-mem = "hermes_custom_memory:register"

Check it loaded with `hermes memory status`.

The module also runs standalone (`python hermes_custom_memory.py`): if Hermes is not
installed it falls back to minimal stand-in base classes so you can exercise the hooks.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional

try:  # Real base classes when running inside Hermes Agent.
    from agent.context_engine import ContextEngine
    from agent.memory_provider import MemoryProvider
except ImportError:  # Stand-ins so the lab runs without Hermes installed.
    from abc import ABC, abstractmethod

    class MemoryProvider(ABC):  # type: ignore[no-redef]
        @property
        @abstractmethod
        def name(self) -> str: ...

    class ContextEngine(ABC):  # type: ignore[no-redef]
        last_prompt_tokens = 0
        last_completion_tokens = 0
        last_total_tokens = 0
        threshold_tokens = 0
        context_length = 0
        compression_count = 0

        @property
        @abstractmethod
        def name(self) -> str: ...


PLUGIN_YAML = """\
name: course-graph-mem
version: 0.1.0
description: LLM AgentOS course - keyword-recall memory provider (Lab 1A)
hooks:
  - prefetch
  - queue_prefetch
  - sync_turn
  - on_pre_compress
  - on_session_end
"""


# ---------------------------------------------------------------------------
# Plane 1: a user / cognitive memory provider
# ---------------------------------------------------------------------------
class CourseGraphMemoryProvider(MemoryProvider):
    """Stores one note per turn in a JSONL file and recalls notes by keyword overlap.

    The interesting part for the course is *when* each hook fires:
      prefetch        -> before every model call (page memory into the context window)
      queue_prefetch  -> after a turn, to pre-warm the next prefetch
      sync_turn       -> after a turn, persist it (must not block the agent loop)
      on_pre_compress -> right before the context engine drops old messages (last chance to extract)
    """

    def __init__(self) -> None:
        self._notes: List[str] = []
        self._path: Optional[str] = None
        self._lock = threading.Lock()
        self._warm: Dict[str, str] = {}

    # --- required -----------------------------------------------------------
    @property
    def name(self) -> str:
        return "course-graph-mem"

    def is_available(self) -> bool:
        # Called during discovery: must be cheap and make no network calls.
        return True

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        # kwargs always contains hermes_home; keep all storage under it.
        hermes_home = kwargs.get("hermes_home", os.path.expanduser("~/.hermes"))
        store_dir = os.path.join(hermes_home, "course-graph-mem")
        os.makedirs(store_dir, exist_ok=True)
        self._path = os.path.join(store_dir, "notes.jsonl")
        self.session_id = session_id
        if os.path.exists(self._path):
            with open(self._path, encoding="utf-8") as f:
                self._notes = [json.loads(line)["text"] for line in f if line.strip()]

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # Expose one tool so the model can search memory on demand (MemGPT-style self-paging).
        return [
            {
                "name": "course_memory_search",
                "description": "Search long-term course notes by keyword.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if tool_name != "course_memory_search":
            return json.dumps({"error": f"unknown tool {tool_name}"})
        return json.dumps({"results": self._search(args.get("query", ""), k=5)})

    def get_config_schema(self) -> List[Dict[str, Any]]:
        # Rendered by `hermes memory setup`.
        return [
            {"key": "max_recall", "description": "Notes injected per turn", "default": 3},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        path = os.path.join(hermes_home, "course-graph-mem", "config.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(values, f)

    # --- optional hooks -----------------------------------------------------
    def system_prompt_block(self) -> str:
        # Static text only: anything that changes per turn here would break the provider's prompt cache.
        return "You have long-term notes; call course_memory_search when the user refers to past work."

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if query in self._warm:
            return self._warm.pop(query)
        return "\n".join(self._search(query, k=3))

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        threading.Thread(target=lambda: self._warm.__setitem__(query, "\n".join(self._search(query, 3))),
                         daemon=True).start()

    def sync_turn(self, user: str, assistant: str, *, session_id: str = "", messages=None) -> None:
        # Non-blocking: write on a background thread.
        note = f"User: {user[:200]} | Agent: {assistant[:200]}"
        threading.Thread(target=self._append, args=(note,), daemon=True).start()

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> None:
        # The context engine is about to summarize these messages away: extract what we want to keep.
        for msg in messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                self._append(f"[pre-compress] {msg['content'][:200]}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        pass

    def shutdown(self) -> None:
        pass

    # --- helpers ------------------------------------------------------------
    def _search(self, query: str, k: int) -> List[str]:
        words = {w for w in query.lower().split() if len(w) > 3}
        scored = [(sum(w in n.lower() for w in words), n) for n in self._notes]
        return [n for s, n in sorted(scored, key=lambda t: -t[0]) if s > 0][:k]

    def _append(self, note: str) -> None:
        with self._lock:
            self._notes.append(note)
            if self._path:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"text": note}) + "\n")


# ---------------------------------------------------------------------------
# Context engine: decides when and how history is compressed
# ---------------------------------------------------------------------------
class CourseThresholdContextEngine(ContextEngine):
    """Keeps the system prompt and the last N messages, drops the middle once over threshold.

    Leaving the head of the prompt untouched keeps the provider-side prefix cache warm,
    which is the same trade-off Module 3 studies at the KV-cache level.
    """

    def __init__(self, keep_last: int = 12, threshold_ratio: float = 0.75) -> None:
        self.keep_last = keep_last
        self.threshold_ratio = threshold_ratio
        self.context_length = 200_000
        self.threshold_tokens = int(self.context_length * threshold_ratio)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0

    @property
    def name(self) -> str:
        return "course-threshold"

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens", 0))
        self.last_completion_tokens = int(usage.get("completion_tokens", 0))
        self.last_total_tokens = self.last_prompt_tokens + self.last_completion_tokens

    def should_compress(self, prompt_tokens: Optional[int] = None) -> bool:
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return tokens >= self.threshold_tokens

    def compress(self, messages: List[Dict[str, Any]], current_tokens: Optional[int] = None,
                 focus_topic: Optional[str] = None) -> List[Dict[str, Any]]:
        # MemoryProvider.on_pre_compress has already run by the time this is called.
        head = [m for m in messages[:1] if m.get("role") == "system"]
        tail = messages[-self.keep_last:]
        dropped = len(messages) - len(head) - len(tail)
        if dropped <= 0:
            return messages
        self.compression_count += 1
        marker = {"role": "user", "content": f"[{dropped} earlier messages were compressed; "
                                            f"use course_memory_search to recall details]"}
        return head + [marker] + tail

    def update_model(self, model: str, context_length: int, *args: Any, **kwargs: Any) -> None:
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_ratio)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------
def register(ctx) -> None:
    # Fresh instances per load; Hermes rejects a second context engine registration.
    ctx.register_memory_provider(CourseGraphMemoryProvider())
    if hasattr(ctx, "register_context_engine"):
        ctx.register_context_engine(CourseThresholdContextEngine())


if __name__ == "__main__":
    import tempfile

    mem = CourseGraphMemoryProvider()
    mem.initialize("demo-session", hermes_home=tempfile.mkdtemp())
    mem._append("User: benchmark the AIOS scheduler with 8 agents | Agent: FIFO vs round-robin results saved")
    print("prefetch ->", mem.prefetch("what did the scheduler benchmark show?"))

    engine = CourseThresholdContextEngine(keep_last=2)
    history = [{"role": "system", "content": "sys"}] + [{"role": "user", "content": str(i)} for i in range(6)]
    print("should_compress(160k) ->", engine.should_compress(160_000))
    print("compressed ->", [m["content"] for m in engine.compress(history)])
