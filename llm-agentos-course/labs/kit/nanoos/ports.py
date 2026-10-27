"""Lab 12: port your nanoos engines to real harness plugin APIs (the harness deep-dive slides).

Labs 3 and 4 built a context engine and a memory system behind nanoos's own small interfaces. Production
harnesses have their own: Hermes Agent has a MemoryProvider and a ContextEngine (Python), and OpenClaw has a
context-engine slot (TypeScript). This module adapts your engines to both, without changing them:

    HermesContextEngine(make_engine("threshold"), mode="compress")   # Hermes rewrites history with it
    HermesContextEngine(make_engine("clearing"), mode="select")      # a per-request view; history stays whole
    HermesMemoryProvider(extractor_factory)                          # your MemoryStore behind Hermes's hooks
    OpenClawEngine(make_engine("threshold"))                         # ingest / assemble / compact, over stdio

The one idea to take away: a nanoos engine returns a *view* of an append-only history on every call. Hermes
offers two places for it. compress() rewrites the stored transcript once it crosses a threshold (cheap later
turns, but the prefix changes at each compression), and select_context() changes one request only (the
transcript stays whole, the view is rebuilt each time). OpenClaw's assemble() is a view too, under a token
budget, and an engine that ownsCompaction rewrites its own store in compact().

Implement the parts marked TODO (`make grade LAB=12`): HermesContextEngine.should_compress, .compress and
.select_context; HermesMemoryProvider.initialize, .prefetch, .sync_turn, .on_pre_compress and .on_session_end;
OpenClawEngine.assemble and .compact. hosts.py drives them the way each harness does; the plugin files in
../lab1_memory_plugins/ load them into the real harnesses.

APIs (checked 2026-10-27):
  https://hermes-agent.nousresearch.com/docs/developer-guide/context-engine-plugin
  https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin
  https://docs.openclaw.ai/concepts/context-engine
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from nanoos.context import ContextEngine as NanoosEngine
from nanoos.context import ThresholdCompaction, ToolResultClearing, count_tokens, heuristic_summarizer
from nanoos.memory import Extractor, MemoryStore

try:  # the real base classes when this runs inside Hermes Agent
    from agent.context_engine import ContextEngine as _HermesContextEngine
    from agent.memory_provider import MemoryProvider as _HermesMemoryProvider
except ImportError:  # stand-ins with the same attributes, so the kit runs without Hermes
    class _HermesContextEngine:  # type: ignore[no-redef]
        last_prompt_tokens = 0
        last_completion_tokens = 0
        last_total_tokens = 0
        threshold_tokens = 0
        context_length = 0
        compression_count = 0

    class _HermesMemoryProvider:  # type: ignore[no-redef]
        pass

EngineFactory = Callable[[int], NanoosEngine]       # budget_tokens -> a fresh nanoos context engine


def make_engine(kind: str, keep_last: int = 6, summarizer: Any = None) -> EngineFactory:
    """A factory for one of your Lab 3 engines (provided). The budget is only used by "threshold"."""
    if kind == "threshold":
        return lambda budget: ThresholdCompaction(budget, summarizer or heuristic_summarizer, keep_last=keep_last)
    if kind == "clearing":
        return lambda budget: ToolResultClearing(keep_last_results=keep_last)
    if kind == "none":
        return lambda budget: NanoosEngine()
    raise ValueError(f"unknown engine {kind!r}: threshold, clearing or none")


# --- Hermes: ContextEngine --------------------------------------------------------------------------------
class HermesContextEngine(_HermesContextEngine):
    """A nanoos engine as a Hermes ContextEngine (select it with context.engine: <name> in config.yaml).

        mode "compress"   should_compress() says when; compress() returns the new, shorter transcript
        mode "select"     select_context() returns a view for one request; the transcript is never rewritten

    Hermes keeps the token counters below up to date through update_from_response() and reads them for its
    status line, so they are maintained here (provided).
    """

    def __init__(self, factory: EngineFactory, mode: str = "compress", name: str = "nanoos-context",
                 threshold_ratio: float = 0.75, target_ratio: float = 0.5, context_length: int = 200_000):
        if mode not in ("compress", "select"):
            raise ValueError("mode must be 'compress' or 'select'")
        self.factory, self.mode, self._name = factory, mode, name
        self.threshold_ratio, self.target_ratio = threshold_ratio, target_ratio
        self.last_prompt_tokens = self.last_completion_tokens = self.last_total_tokens = 0
        self.compression_count = 0
        self.update_model("", context_length)

    # --- provided ---------------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def target_tokens(self) -> int:
        """What compress() aims for: target_ratio of the threshold, so it does not fire again next turn."""
        return int(self.threshold_tokens * self.target_ratio)

    def update_from_response(self, usage: dict[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens", 0))
        self.last_completion_tokens = int(usage.get("completion_tokens", 0))
        self.last_total_tokens = self.last_prompt_tokens + self.last_completion_tokens

    def update_model(self, model: str, context_length: int, *args: Any, **kwargs: Any) -> None:
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_ratio)
        self.engine = self.factory(self.threshold_tokens)      # select mode's long-lived engine

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        self.engine = self.factory(self.threshold_tokens)

    def on_session_end(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        pass

    def on_session_reset(self) -> None:
        self.last_prompt_tokens = self.last_completion_tokens = self.last_total_tokens = 0
        self.compression_count = 0
        self.engine = self.factory(self.threshold_tokens)

    def clone_for_agent(self) -> "HermesContextEngine":
        """Hermes shares the registered instance and clones it per agent: a fresh engine, the same settings."""
        return HermesContextEngine(self.factory, self.mode, self._name, self.threshold_ratio, self.target_ratio,
                                   self.context_length)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return []

    def get_status(self) -> dict[str, Any]:
        return {"engine": self._name, "mode": self.mode, "last_prompt_tokens": self.last_prompt_tokens,
                "threshold_tokens": self.threshold_tokens, "compression_count": self.compression_count}

    # --- TODO -------------------------------------------------------------------------------------------
    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        """True when Hermes should call compress(): only in "compress" mode, and only when the prompt has
        reached self.threshold_tokens. prompt_tokens defaults to self.last_prompt_tokens."""
        # TODO(lab 12)
        raise NotImplementedError("Lab 12: implement HermesContextEngine.should_compress")

    def compress(self, messages: list[dict[str, Any]], current_tokens: int | None = None,
                 focus_topic: str | None = None) -> list[dict[str, Any]]:
        """Return the new transcript: a fresh engine's view of `messages` (the input is Hermes's snapshot).

        Build the engine with self.factory(self.target_tokens) and call prepare(messages) once. A fresh engine
        each time, because the transcript it sees next time is the rewritten one, so a long-lived view
        engine's positions would be wrong. If the engine did not compact (its `compactions` is still 0), return
        `messages` unchanged and count nothing; otherwise add one to self.compression_count and return the view
        as a new list. focus_topic (from /compress <topic>) is ignored here; see the stretch.
        """
        # TODO(lab 12)
        raise NotImplementedError("Lab 12: implement HermesContextEngine.compress")

    def select_context(self, request_messages: list[dict[str, Any]], *,
                       conversation_messages: list[dict[str, Any]] | None = None,
                       incoming_message: Any = None, budget_tokens: int = 0) -> list[dict[str, Any]] | None:
        """The messages for this one request, or None to send request_messages as they are.

        Only in "select" mode: run self.engine.prepare(request_messages) (one engine for the session, since
        the transcript it sees only grows). Return None when the view is the input itself or equal to it, so
        Hermes keeps its own list; otherwise return the view as a new list. In "compress" mode return None.
        """
        # TODO(lab 12)
        raise NotImplementedError("Lab 12: implement HermesContextEngine.select_context")


# --- Hermes: MemoryProvider -------------------------------------------------------------------------------
SEARCH_SCHEMA = {
    "name": "nanoos_memory_search",
    "description": "Search what you remember about this user from earlier sessions. Use it when the user "
                   "refers to something they told you before.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "what to look for"},
        "k": {"type": "integer", "description": "how many memories, 1 to 10 (default 5)"}},
        "required": ["query"]}}
RECALL_HEADER = "What you remember about this user (from earlier sessions):"


class HermesMemoryProvider(_HermesMemoryProvider):
    """Your Lab 4 MemoryStore as a Hermes MemoryProvider (select it with memory.provider: <name>).

    Hermes calls, in this order: initialize() at startup; system_prompt_block() and get_tool_schemas() while
    it builds the prompt; prefetch() before each model call; handle_tool_call() for the search tool;
    sync_turn() and queue_prefetch() after each turn; on_pre_compress() before the context engine drops
    messages; on_session_end() and shutdown() at the end. Only one external provider can be active.

    extractor_factory() returns your Lab 4 extractor (a ModelExtractor in Hermes, a ScriptedExtractor in the
    tests). Memories are scoped to Hermes's user_id, so two users of one gateway never see each other's.
    store_cls is your MemoryStore (tests pass a stand-in, so this lab does not depend on your Lab 4 grade).
    """

    def __init__(self, extractor_factory: Callable[[], Extractor], name: str = "nanoos-memory", k: int = 3,
                 embedder: Any = None, trusted_tools: tuple[str, ...] = (), store_cls: Any = MemoryStore):
        self.extractor_factory, self._name, self.k = extractor_factory, name, k
        self.embedder, self.trusted_tools, self.store_cls = embedder, trusted_tools, store_cls
        self.store: Any = None
        self.user_id = "default"
        self.session_id = ""
        self.path: Path | None = None
        self.lock = threading.Lock()
        self.threads: list[threading.Thread] = []
        self.warm: dict[str, str] = {}
        self.extractor: Extractor | None = None

    # --- provided ---------------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True      # called during discovery: no network, no heavy imports

    def system_prompt_block(self) -> str:
        # Static text only: a block that changed per turn would break the provider's prompt cache (Lab 5).
        return ("You remember facts users told you in earlier sessions. Relevant ones are added to the "
                "conversation; call nanoos_memory_search to look for more.")

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [SEARCH_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if tool_name != SEARCH_SCHEMA["name"] or self.store is None:
            return json.dumps({"error": f"unknown tool {tool_name}"})
        k = args.get("k", 5) if isinstance(args.get("k"), int) else 5
        with self.lock:
            hits = self.store.search(self.user_id, str(args.get("query", "")), k=max(1, min(10, k)))
        return json.dumps({"results": [m.text for m in hits]})

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        self.spawn(lambda: self.warm.__setitem__(query, self.recall(query)))

    def shutdown(self) -> None:
        self.flush()

    def spawn(self, fn: Callable[[], None]) -> None:
        """Run fn on a daemon thread and remember it, so flush() can wait for it.

        Newer Hermes docs ask providers to use agent.memory_provider.spawn_context_thread instead of a bare
        thread, so the work keeps the active profile. Check your version (handout Part C).
        """
        t = threading.Thread(target=fn, daemon=True)
        self.threads.append(t)
        t.start()

    def flush(self) -> None:
        """Wait for pending background work, then save the store."""
        for t in list(self.threads):
            t.join(timeout=30)
        self.threads = [t for t in self.threads if t.is_alive()]
        if self.store is not None and self.path is not None:
            with self.lock:
                self.store.save(self.path)

    def recall(self, query: str) -> str:
        """The recalled block for one query: RECALL_HEADER and one "- text" line per hit, or "" (provided)."""
        if self.store is None:
            return ""
        with self.lock:
            hits = self.store.search(self.user_id, query, k=self.k)
        return "\n".join([RECALL_HEADER] + [f"- {m.text}" for m in hits]) if hits else ""

    def write(self, messages: list[dict[str, Any]]) -> None:
        """The write path: ingest messages under the lock (provided)."""
        if self.store is None or self.extractor is None:
            return
        with self.lock:
            self.store.ingest(self.user_id, messages, self.extractor, self.trusted_tools)

    # --- TODO -------------------------------------------------------------------------------------------
    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Open the store for this session. kwargs always has hermes_home; other fields may be missing.

        - self.session_id = session_id; self.user_id = kwargs.get("user_id") or "default".
        - self.path = <hermes_home>/<self.name>/memory.json (create the folder). Load it with
          self.store_cls.load(path, self.embedder) if it exists, else start self.store_cls(self.embedder).
        - self.extractor = self.extractor_factory(); clear self.warm.
        """
        # TODO(lab 12)
        raise NotImplementedError("Lab 12: implement HermesMemoryProvider.initialize")

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recalled context for the coming model call: a result queue_prefetch() warmed for this exact query
        (take it out of self.warm), else self.recall(query)."""
        # TODO(lab 12)
        raise NotImplementedError("Lab 12: implement HermesMemoryProvider.prefetch")

    def sync_turn(self, user: str, assistant: str, *, session_id: str = "",
                  messages: list[dict[str, Any]] | None = None) -> None:
        """Persist a finished turn. It MUST NOT block the agent loop: hand the work to self.spawn().

        The background work calls self.write([{"role": "user", "content": user}, {"role": "assistant",
        "content": assistant}]) and then saves the store to self.path (under self.lock). Your Lab 4 write path
        decides what is evidence: the assistant's words are not.
        """
        # TODO(lab 12)
        raise NotImplementedError("Lab 12: implement HermesMemoryProvider.sync_turn")

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> None:
        """The last chance to read messages before the context engine drops them: write them synchronously
        (Hermes runs this inside its timed compression pass). Tool results reach the extractor only if the tool
        is in self.trusted_tools; your Lab 4 trust gate decides. Duplicates of facts sync_turn already wrote are
        harmless, because consolidation ignores them."""
        # TODO(lab 12)
        raise NotImplementedError("Lab 12: implement HermesMemoryProvider.on_pre_compress")

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """Make the session durable: wait for pending writes and save (self.flush())."""
        # TODO(lab 12)
        raise NotImplementedError("Lab 12: implement HermesMemoryProvider.on_session_end")


# --- OpenClaw: the context-engine contract, served to the TypeScript shell ---------------------------------
def fit_budget(messages: list[dict[str, Any]], budget: int | None) -> list[dict[str, Any]]:
    """The newest messages that fit `budget` tokens, in order (provided).

    A leading system message is always kept. The kept tail never starts with a tool result, because a tool
    result whose assistant call was dropped is invalid in the OpenAI format.
    """
    if budget is None or count_tokens(messages) <= budget:
        return list(messages)
    head = [messages[0]] if messages and messages[0].get("role") == "system" else []
    used = count_tokens(head)
    tail: list[dict[str, Any]] = []
    for m in reversed(messages[len(head):]):
        cost = count_tokens([m])
        if used + cost > budget:
            break
        tail.insert(0, m)
        used += cost
    while tail and tail[0].get("role") == "tool":
        tail.pop(0)
    return head + tail


class _Session:
    def __init__(self):
        self.log: list[dict[str, Any]] = []
        self.engine: NanoosEngine | None = None
        self.epoch = 0


class OpenClawEngine:
    """A nanoos engine behind OpenClaw's context-engine contract (plugins.slots.contextEngine).

    OpenClaw calls ingest() for each message added to a session, assemble() before each model run (with a
    token budget), compact() on overflow or /compact, afterTurn() after a run, and dispose() at shutdown.
    The TypeScript shell in ../lab1_memory_plugins/openclaw_context_engine.ts forwards each call here as one
    JSON line (serve_stdio). Params and results use OpenClaw's field names.

    owns_compaction: True tells OpenClaw to turn off its built-in auto-compaction; compact() then has to
    shrink the session itself. If any method raises, OpenClaw quarantines the engine and falls back to its
    built-in "legacy" engine for the rest of the gateway process.
    """

    def __init__(self, factory: EngineFactory, engine_id: str = "nanoos-engine", owns_compaction: bool = True,
                 target_ratio: float = 0.5):
        self.factory, self.engine_id = factory, engine_id
        self.owns_compaction, self.target_ratio = owns_compaction, target_ratio
        self.sessions: dict[str, _Session] = {}

    # --- provided ---------------------------------------------------------------------------------------
    def info(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"id": self.engine_id, "name": f"nanoos ({self.engine_id})", "ownsCompaction": self.owns_compaction,
                "acceptedHostParams": ["sessionKey"]}

    def session(self, session_id: str) -> _Session:
        return self.sessions.setdefault(session_id, _Session())

    def ingest(self, params: dict[str, Any]) -> dict[str, Any]:
        if params.get("isHeartbeat"):
            return {"ingested": False}
        self.session(params["sessionId"]).log.append(params["message"])
        return {"ingested": True}

    def after_turn(self, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    def dispose(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.sessions.clear()
        return {}

    # --- TODO -------------------------------------------------------------------------------------------
    def assemble(self, params: dict[str, Any]) -> dict[str, Any]:
        """The context for one model run: {"messages": [...], "estimatedTokens": n}.

        - Source: params["messages"] when it is non-empty (the host's transcript), else the session's log.
        - The session keeps one engine, created on first use with self.factory(budget), where budget is
          params.get("tokenBudget") or a large default (10**9). Run engine.prepare(source).
        - Then fit_budget(view, params.get("tokenBudget")): the engine's view may still be over the budget, and
          OpenClaw expects assemble() to respect it.
        - estimatedTokens is count_tokens(messages). When the session's epoch is above 0 (it compacted), add
          "contextProjection": {"mode": "thread_bootstrap", "epoch": epoch}.
        """
        # TODO(lab 12)
        raise NotImplementedError("Lab 12: implement OpenClawEngine.assemble")

    def compact(self, params: dict[str, Any]) -> dict[str, Any]:
        """Shrink the session's stored log: {"ok": True, "compacted": bool}.

        - If not self.owns_compaction, compact nothing and return {"ok": True, "compacted": False}: the
          runtime's compactor is in charge.
        - Otherwise run a fresh engine, self.factory(int(budget * self.target_ratio)) with budget =
          params.get("tokenBudget") or count_tokens(log), over the log. If it compacted (compactions > 0),
          replace the log with the view, drop the session's assemble engine (positions changed), add one to the
          epoch and return compacted True. Else return compacted False. With params["force"] (a /compact), use
          a budget of 0 so it always compacts when there is anything to compact.
        """
        # TODO(lab 12)
        raise NotImplementedError("Lab 12: implement OpenClawEngine.compact")


METHODS = {"info": "info", "ingest": "ingest", "assemble": "assemble", "compact": "compact",
           "afterTurn": "after_turn", "dispose": "dispose"}


def serve_stdio(engine: OpenClawEngine, stdin: Any = None, stdout: Any = None) -> None:
    """One JSON request per line ({"id", "method", "params"}) -> one JSON reply per line (provided).

    A method that raises answers {"id", "error": {"message"}}; the TypeScript shell turns it into an exception,
    and OpenClaw then quarantines the engine.
    """
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    for line in stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        try:
            method = METHODS.get(req.get("method"))
            if method is None:
                raise ValueError(f"unknown method {req.get('method')!r}")
            reply: dict[str, Any] = {"id": req.get("id"), "result": getattr(engine, method)(req.get("params") or {})}
        except Exception as e:  # noqa: BLE001 - every failure goes back to the host
            reply = {"id": req.get("id"), "error": {"message": f"{type(e).__name__}: {e}"}}
        stdout.write(json.dumps(reply) + "\n")
        stdout.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Serve a nanoos context engine to OpenClaw over stdio.")
    ap.add_argument("--engine", choices=["threshold", "clearing", "none"], default="threshold")
    ap.add_argument("--keep-last", type=int, default=6)
    ap.add_argument("--id", default="nanoos-engine")
    ap.add_argument("--no-own-compaction", action="store_true")
    args = ap.parse_args(argv)
    serve_stdio(OpenClawEngine(make_engine(args.engine, args.keep_last), args.id, not args.no_own_compaction))
    return 0


if __name__ == "__main__":
    sys.exit(main())
