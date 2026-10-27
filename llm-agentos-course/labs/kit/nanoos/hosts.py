"""Lab 12: two small hosts that call plugins the way Hermes Agent and OpenClaw do (provided).

They are not the harnesses. They are just enough of each one's plugin contract to test a port offline and to
measure it on the long tier: the hooks run in the documented order, with the documented arguments. Where the
docs leave something open, the choice made here is written down, and Part C of the handout checks it against
the real harness.

    host = HermesHost(model, tools, engine=HermesContextEngine(...), memory=HermesMemoryProvider(...))
    result = host.run(["first user message", "second user message"])

    host = OpenClawHost(model, tools, engine=OpenClawEngine(...), token_budget=6000)   # or BridgeClient(...)
    result = host.run(["..."])

HostResult.requests holds every message list sent to the model, so you can see what the engine did.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanoos.context import count_tokens
from nanoos.types import Model

SYSTEM_PROMPT = ("You are a careful assistant working in a workspace of files. Use the tools to look at the "
                 "files before you answer. When the task asks a question, end with a short final answer and no "
                 "tool calls.")


@dataclass
class HostResult:
    answers: list[str] = field(default_factory=list)
    stopped_by: str = "answer"
    history: list[dict[str, Any]] = field(default_factory=list)       # the stored transcript at the end
    requests: list[list[dict[str, Any]]] = field(default_factory=list)  # every message list sent to the model
    prompt_tokens: list[int] = field(default_factory=list)            # count_tokens of each request
    compressions: int = 0                                             # transcript rewrites (compress, compact)
    view_compactions: int = 0                                         # compactions inside per-request views
    tool_errors: int = 0
    hooks: list[str] = field(default_factory=list)                    # the order hooks ran in

    @property
    def total_prompt_tokens(self) -> int:
        return sum(self.prompt_tokens)

    def cacheable_share(self) -> float:
        """Share of prompt tokens that repeat the previous request's leading messages exactly: a stand-in for
        what a provider's prefix cache could reuse (Lab 5). Rewriting the head of the history lowers it."""
        reused = 0
        for prev, cur in zip(self.requests, self.requests[1:]):
            n = 0
            while n < min(len(prev), len(cur)) and prev[n] == cur[n]:
                n += 1
            reused += count_tokens(cur[:n])
        total = sum(self.prompt_tokens[1:])
        return reused / total if total else 0.0


def _dispatch(name: str, args: dict[str, Any], tools: Any, plugin_tools: dict[str, Any]) -> tuple[str, bool]:
    if name in plugin_tools:
        out = plugin_tools[name].handle_tool_call(name, args)
        return out, '"error"' in out[:40]
    return tools.call(name, args)


class HermesHost:
    """Hermes Agent's plugin calls for one session, as in its developer guide.

    Session start: memory.initialize(session_id, hermes_home=..., user_id=...); engine.on_session_start();
    the system prompt gets memory.system_prompt_block(); tools get both plugins' get_tool_schemas().
    Each user turn: memory.prefetch(user text), added to the turn's user message (where your Hermes version
    puts it is a Part C question). Then, before each model call: if engine.should_compress(last prompt
    tokens), memory.on_pre_compress(snapshot) and history = engine.compress(snapshot); the request is
    engine.select_context(history) or history; after the call, engine.update_from_response(usage). After the
    turn: memory.sync_turn(user, answer, messages=...), memory.queue_prefetch(user). Session end:
    engine.on_session_end(), memory.on_session_end(history), memory.shutdown().
    """

    def __init__(self, model: Model, tools: Any, engine: Any = None, memory: Any = None, *,
                 hermes_home: str | Path | None = None, user_id: str = "default", session_id: str = "s1",
                 max_steps: int = 60, system_prompt: str = SYSTEM_PROMPT):
        self.model, self.tools, self.engine, self.memory = model, tools, engine, memory
        self.hermes_home = str(hermes_home or tempfile.mkdtemp(prefix="hermes-home-"))
        self.user_id, self.session_id = user_id, session_id
        self.max_steps, self.system_prompt = max_steps, system_prompt

    def run(self, user_turns: list[str]) -> HostResult:
        r = HostResult()
        eng, mem = self.engine, self.memory
        if mem:
            mem.initialize(self.session_id, hermes_home=self.hermes_home, user_id=self.user_id, platform="cli")
            r.hooks.append("memory.initialize")
        if eng and hasattr(eng, "on_session_start"):
            eng.on_session_start(self.session_id)
            r.hooks.append("engine.on_session_start")
        system = self.system_prompt + ("\n\n" + mem.system_prompt_block() if mem else "")
        plugin_tools: dict[str, Any] = {}
        schemas = list(self.tools.schemas())
        for plugin in (mem, eng):
            for s in (plugin.get_tool_schemas() if plugin else []):
                plugin_tools[s["name"]] = plugin
                schemas.append({"type": "function", "function": s})
        history: list[dict[str, Any]] = [{"role": "system", "content": system}]
        last_prompt = 0
        for text in user_turns:
            content = text
            if mem:
                recalled = mem.prefetch(text, session_id=self.session_id)
                r.hooks.append("memory.prefetch")
                if recalled:
                    content = f"{text}\n\n<memory-context>\n{recalled}\n</memory-context>"
            history.append({"role": "user", "content": content})
            answer, done = "", False
            for _ in range(self.max_steps):
                if eng and eng.should_compress(last_prompt):
                    snapshot = copy.deepcopy(history)
                    if mem:
                        mem.on_pre_compress(copy.deepcopy(snapshot))
                        r.hooks.append("memory.on_pre_compress")
                    history = eng.compress(snapshot, last_prompt)
                    r.hooks.append("engine.compress")
                request = history
                if eng and hasattr(eng, "select_context"):
                    selected = eng.select_context(copy.deepcopy(history), conversation_messages=history)
                    request = selected if selected is not None else history
                r.requests.append(copy.deepcopy(request))
                r.prompt_tokens.append(count_tokens(request))
                resp = self.model.complete(request, schemas)
                last_prompt = resp.usage.input_tokens
                if eng:
                    eng.update_from_response({"prompt_tokens": resp.usage.input_tokens,
                                              "completion_tokens": resp.usage.output_tokens})
                history.append(resp.to_message())
                if not resp.tool_calls:
                    answer, done = resp.content or "", True
                    break
                for call in resp.tool_calls:
                    out, err = _dispatch(call.name, call.arguments, self.tools, plugin_tools)
                    r.tool_errors += err
                    history.append({"role": "tool", "tool_call_id": call.id, "content": out})
            r.answers.append(answer)
            if not done:
                r.stopped_by = "budget"
            if mem:
                mem.sync_turn(text, answer, session_id=self.session_id, messages=copy.deepcopy(history))
                mem.queue_prefetch(text, session_id=self.session_id)
                r.hooks += ["memory.sync_turn", "memory.queue_prefetch"]
        if eng:
            eng.on_session_end(self.session_id, history)
            r.compressions = eng.compression_count
            r.view_compactions = getattr(getattr(eng, "engine", None), "compactions", 0) if \
                getattr(eng, "mode", "") == "select" else 0
            r.hooks.append("engine.on_session_end")
        if mem:
            mem.on_session_end(copy.deepcopy(history))
            mem.shutdown()
            r.hooks += ["memory.on_session_end", "memory.shutdown"]
        r.history = history
        return r


class BridgeClient:
    """Talks to `python -m nanoos.ports` over stdio, as the TypeScript shell does (provided).

    call(method, params) returns the result or raises RuntimeError with the engine's error message.
    """

    def __init__(self, args: list[str] | None = None, cwd: str | Path | None = None):
        kit = Path(__file__).resolve().parent.parent
        self.proc = subprocess.Popen([sys.executable, "-m", "nanoos.ports", *(args or [])], cwd=str(cwd or kit),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._id = 0

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self._id += 1
        self.proc.stdin.write(json.dumps({"id": self._id, "method": method, "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        reply = json.loads(self.proc.stdout.readline())
        if "error" in reply:
            raise RuntimeError(reply["error"]["message"])
        return reply["result"]

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


class _Direct:
    """Calls an OpenClawEngine object with the bridge's method names."""

    NAMES = {"afterTurn": "after_turn"}

    def __init__(self, engine: Any):
        self.engine = engine

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        return getattr(self.engine, self.NAMES.get(method, method))(params or {})


class OpenClawHost:
    """OpenClaw's context-engine calls for one session.

    Every message added to the session is ingest()ed. Before each model call, assemble() with the token
    budget and no messages (the engine answers from its own store) gives the request. If the engine does not
    own compaction, this host does what the runtime's built-in compactor would at its simplest: it sends
    only the newest messages that fit the budget. If assemble() reports more tokens than the budget, the host
    calls compact() and assembles again. afterTurn() follows each turn. If any engine call raises, the engine
    is quarantined for the rest of the session and the host falls back to the "legacy" behavior: the whole
    transcript, cut to the budget.
    """

    def __init__(self, model: Model, tools: Any, engine: Any, token_budget: int = 6000, *, session_id: str = "s1",
                 max_steps: int = 60, system_prompt: str = SYSTEM_PROMPT):
        self.model, self.tools = model, tools
        self.engine = engine if hasattr(engine, "call") else _Direct(engine)
        self.token_budget, self.session_id = token_budget, session_id
        self.max_steps, self.system_prompt = max_steps, system_prompt
        self.quarantined: str | None = None

    def _engine(self, method: str, params: dict[str, Any]) -> Any:
        if self.quarantined:
            return None
        try:
            return self.engine.call(method, params)
        except Exception as e:  # noqa: BLE001 - OpenClaw isolates engine failures from the reply path
            self.quarantined = f"{method}: {e}"
            return None

    def run(self, user_turns: list[str]) -> HostResult:
        from nanoos.ports import fit_budget
        r = HostResult()
        info = self._engine("info", {}) or {}
        owns = bool(info.get("ownsCompaction"))
        transcript: list[dict[str, Any]] = []
        schemas = self.tools.schemas()
        sid = self.session_id

        def add(message: dict[str, Any]) -> None:
            transcript.append(message)
            self._engine("ingest", {"sessionId": sid, "message": message})

        add({"role": "system", "content": self.system_prompt})
        for text in user_turns:
            add({"role": "user", "content": text})
            answer, done = "", False
            for _ in range(self.max_steps):
                params = {"sessionId": sid, "sessionKey": sid, "messages": [], "tokenBudget": self.token_budget}
                out = self._engine("assemble", params)
                if out and owns and out.get("estimatedTokens", 0) > self.token_budget:
                    compacted = self._engine("compact", {"sessionId": sid, "tokenBudget": self.token_budget})
                    if compacted and compacted.get("compacted"):
                        r.compressions += 1
                        r.hooks.append("compact")
                    out = self._engine("assemble", params)
                if out is None:
                    request = fit_budget(transcript, self.token_budget)      # legacy fallback
                else:
                    request = out["messages"] if owns else fit_budget(out["messages"], self.token_budget)
                r.hooks.append("assemble")
                r.requests.append(copy.deepcopy(request))
                r.prompt_tokens.append(count_tokens(request))
                resp = self.model.complete(request, schemas)
                add(resp.to_message())
                if not resp.tool_calls:
                    answer, done = resp.content or "", True
                    break
                for call in resp.tool_calls:
                    res, err = self.tools.call(call.name, call.arguments)
                    r.tool_errors += err
                    add({"role": "tool", "tool_call_id": call.id, "content": res})
            r.answers.append(answer)
            if not done:
                r.stopped_by = "budget"
            self._engine("afterTurn", {"sessionId": sid})
        r.history = transcript
        return r
