"""Lab 4: long-term memory for nanoos.

A memory system is a small data system: a write path, a store, an index and a read path (see the long-term
memory deck). This module has all four, scoped per user:

    store = MemoryStore()                                   # hybrid index: BM25 + vectors (HashEmbedder)
    store.ingest("alice", session_messages, extractor)      # write path: extract, then consolidate
    store.search("alice", "where do I live?", k=3)          # hybrid search, scoped to alice
    model = ContextManagedModel(model, AutoRecall(store, "alice"))   # read path A: inject before each call
    tools = MemoryTools(tools, store, "alice")                        # read path B: a search_memory tool

Implement the four parts marked TODO (`make grade LAB=4`): `MemoryStore.apply` (consolidation),
`MemoryStore.ingest` (the write path and its trust gate), `MemoryStore.search` (hybrid retrieval) and
`AutoRecall.prepare` (read path A). Everything else here is provided. memory_eval.py runs the deck's exercise.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
import zlib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from nanoos.types import Model, Usage, estimate_tokens

MEMORY_HEADER = "\n\nWhat you remember about this user (from earlier sessions):\n"
RRF_K = 60

# extractor(user_id, text, existing active memories of that user) -> list of operations (see MemoryStore.apply)
Extractor = Callable[[str, str, "list[Memory]"], "list[dict[str, Any]]"]


@dataclass
class Memory:
    id: str
    user_id: str
    text: str
    key: str | None = None            # a profile slot such as "home_city"; one active memory per key and user
    source: str = "user"              # "user" or "tool:<name>": where the fact came from
    created_at: int = 0               # the store's clock (one tick per ingested session) when written
    invalid_at: int | None = None     # set when a newer memory supersedes this one
    supersedes: str | None = None     # id of the memory this one replaced
    superseded_by: str | None = None

    @property
    def active(self) -> bool:
        return self.invalid_at is None


# --- indexing helpers (provided) ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize(text: str) -> str:
    """For duplicate checks: lowercase, words only, single spaces."""
    return " ".join(tokenize(text))


class BM25:
    """Okapi BM25 over a small list of documents. scores(query)[i] is document i's score (0 = no shared term)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / len(self.docs) if self.docs else 0.0
        self.df = Counter(t for d in self.docs for t in set(d))

    def scores(self, query: str) -> list[float]:
        n, out = len(self.docs), []
        for d in self.docs:
            tf, s = Counter(d), 0.0
            for t in set(tokenize(query)):
                if t not in tf:
                    continue
                idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                s += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


class HashEmbedder:
    """A deterministic, offline stand-in for an embedding model: hashed words and character trigrams.

    It matches spelling variants ("dog", "dogs") but not synonyms ("dog", "puppy"); a real embedding model
    (APIEmbedder) does both. Same interface: embed(texts) -> list of unit vectors.
    """

    name = "hash"

    def __init__(self, dims: int = 512):
        self.dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.dims
            for w in tokenize(text):
                feats = [w] + [f"#{w}#"[i:i + 3] for i in range(len(w))]
                for f in feats:
                    v[zlib.crc32(f.encode()) % self.dims] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class APIEmbedder:
    """Embeddings from an OpenAI-compatible /embeddings endpoint (a hosted API, or vLLM serving an embedding model).

    NANOOS_EMBED_MODEL names the model; NANOOS_EMBED_BASE_URL and NANOOS_EMBED_API_KEY default to
    NANOOS_BASE_URL and NANOOS_API_KEY.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        self.name = model or os.environ.get("NANOOS_EMBED_MODEL", "")
        self.base_url = (base_url or os.environ.get("NANOOS_EMBED_BASE_URL") or os.environ.get("NANOOS_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("NANOOS_EMBED_API_KEY") or os.environ.get("NANOOS_API_KEY", "")
        if not self.name or not self.base_url:
            raise ValueError("set NANOOS_EMBED_MODEL and NANOOS_EMBED_BASE_URL (or NANOOS_BASE_URL)")

    def embed(self, texts: list[str]) -> list[list[float]]:
        req = urllib.request.Request(
            f"{self.base_url}/embeddings", data=json.dumps({"model": self.name, "input": texts}).encode(),
            method="POST", headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)["data"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"embeddings HTTP {e.code}: {e.read()[:300].decode(errors='replace')}") from None
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


# --- the store ---------------------------------------------------------------------------------------------
class MemoryStore:
    def __init__(self, embedder: Any = None):
        self.embedder = embedder or HashEmbedder()
        self.memories: list[Memory] = []
        self.clock = 0          # advanced once per ingested session
        self.rejected = 0       # messages the trust gate kept away from the extractor
        self._next_id = 1
        self._vectors: dict[str, list[float]] = {}

    # provided helpers
    def new_id(self) -> str:
        mid = f"m{self._next_id}"
        self._next_id += 1
        return mid

    def active(self, user_id: str) -> list[Memory]:
        return [m for m in self.memories if m.user_id == user_id and m.active]

    def get(self, memory_id: str) -> Memory | None:
        return next((m for m in self.memories if m.id == memory_id), None)

    def vector(self, m: Memory) -> list[float]:
        if m.id not in self._vectors:
            self._vectors[m.id] = self.embedder.embed([m.text])[0]
        return self._vectors[m.id]

    def save(self, path: str | Path) -> None:
        data = {"clock": self.clock, "next_id": self._next_id, "rejected": self.rejected,
                "memories": [asdict(m) for m in self.memories]}
        Path(path).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, embedder: Any = None) -> "MemoryStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(embedder)
        store.clock, store._next_id, store.rejected = data["clock"], data["next_id"], data.get("rejected", 0)
        store.memories = [Memory(**m) for m in data["memories"]]
        return store

    # --- consolidation ---
    def apply(self, user_id: str, op: dict[str, Any], source: str = "user") -> Memory | None:
        """Apply one operation from an extractor to `user_id`'s memories (Mem0-style consolidation).

        Operations:

            {"op": "ADD", "text": ..., "key": optional slot name}
            {"op": "UPDATE", "target": memory id or key, "text": ...}
            {"op": "DELETE", "target": memory id or key}
            {"op": "NOOP"}

        Rules:

        - A target is resolved among this user's *active* memories: first by id, then by key. Never touch
          another user's memories.
        - ADD creates Memory(id=self.new_id(), user_id, text, key, source, created_at=self.clock) and returns it.
          If an active memory of this user has the same normalize(text), it is a duplicate: change nothing
          and return None. If the op has a key that an active memory already holds, it is an UPDATE of that
          memory.
        - UPDATE supersedes, it does not overwrite: the new memory takes the old one's key and has
          supersedes=old.id; the old one gets invalid_at=self.clock and superseded_by=new.id and stays in
          self.memories as history. Return the new memory. If the new text is the same as the target's
          (normalize), change nothing and return None. If the target matches nothing, add the text as a new
          memory without a key.
        - DELETE is "forget": remove the target and every older version it superseded (follow `supersedes`)
          from self.memories entirely, so not even the saved file keeps the text. Return None. A target that
          matches nothing is ignored.
        - NOOP (or an unknown op) changes nothing and returns None.
        """
        # TODO(lab 4)
        raise NotImplementedError("Lab 4: implement MemoryStore.apply")

    # --- write path ---
    def ingest(self, user_id: str, messages: list[dict[str, Any]], extractor: Extractor,
               trusted_tools: tuple[str, ...] = ()) -> list[Memory]:
        """The write path, run at the end of a session (or before compaction) over its messages.

        - Advance the clock by one first: everything written in this call gets the new self.clock.
        - Go through the messages in order. "user" messages are trusted. A "tool" message is trusted only when
          the tool that produced it (the name in the assistant tool call whose id matches its tool_call_id) is
          in trusted_tools. "system" and "assistant" messages are not evidence about the user: skip them.
        - An untrusted tool result never reaches the extractor (it might say "remember that ..."; that is
          memory poisoning). Count each one in self.rejected.
        - For each trusted message, call extractor(user_id, content, self.active(user_id)) and apply every
          operation it returns, in order, with source "user" or "tool:<name>". Ask the extractor again for
          each message, so it sees what the previous messages wrote.
        - Return the memories that were created (the return values of apply that are not None), in order.
        """
        # TODO(lab 4)
        raise NotImplementedError("Lab 4: implement MemoryStore.ingest")

    # --- read: hybrid search ---
    def search(self, user_id: str, query: str, k: int = 5) -> list[Memory]:
        """Hybrid retrieval over `user_id`'s active memories only.

        - Rank the candidates twice: by BM25(texts).scores(query), and by cosine(query vector, self.vector(m))
          with the query embedded by self.embedder. In each ranking keep only candidates with a score > 0,
          best first (rank 1 is the best). Ties, here and below, go to the newer memory (higher created_at),
          then to the one created first (earlier in self.memories).
        - Fuse with Reciprocal Rank Fusion: score(m) = sum over the rankings that contain m of
          1 / (RRF_K + rank). A memory in neither ranking is not returned.
        - Sort by fused score, highest first (ties as above). Return at most k.
        """
        # TODO(lab 4)
        raise NotImplementedError("Lab 4: implement MemoryStore.search")


# --- read path A: automatic injection ----------------------------------------------------------------------
class AutoRecall:
    """Read path A: before every model call, search with the latest user message and inject the hits.

    Use it as a context engine: ContextManagedModel(model, AutoRecall(store, user_id)).
    """

    name = "memory"
    compactions = 0

    def __init__(self, store: MemoryStore, user_id: str, k: int = 3):
        self.store = store
        self.user_id = user_id
        self.k = k
        self.injected: list[list[str]] = []   # per model call: the ids of the memories injected
        self.injected_text: list[str] = []    # every memory text injected, for the eval's recall and leak checks
        self.injected_tokens = 0

    def prepare(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the messages to send, with the user's relevant memories added to the system prompt.

        - The query is the content of the last message with role "user".
        - hits = self.store.search(self.user_id, query, self.k). Record the hit ids in self.injected (an empty
          list when there are none), so there is one entry per call.
        - No hits: return the messages unchanged.
        - Otherwise the first message must be the system prompt: return a new list in which it is a new dict
          with content = old content + MEMORY_HEADER + one line "- <text>" per hit (in search order, joined
          with "\\n"). Add each hit's text to self.injected_text and the tokens of the added block
          (estimate_tokens) to self.injected_tokens.
        - Never modify the input list or its dicts.
        """
        # TODO(lab 4)
        raise NotImplementedError("Lab 4: implement AutoRecall.prepare")


# --- read path B: a search tool (provided) -----------------------------------------------------------------
class MemoryTools:
    """Read path B: wraps a tool set and adds a search_memory tool over the store, scoped to one user."""

    SCHEMA = {"type": "function", "function": {
        "name": "search_memory",
        "description": ("Search what you remember about this user from earlier sessions (their facts, preferences "
                        "and plans). Call it whenever the question depends on something the user told you before."),
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}

    def __init__(self, tools: Any, store: MemoryStore, user_id: str, k: int = 5):
        self.tools, self.store, self.user_id, self.k = tools, store, user_id, k
        self.calls = 0
        self.returned_text: list[str] = []
        self.returned_tokens = 0

    def schemas(self) -> list[dict[str, Any]]:
        return self.tools.schemas() + [self.SCHEMA]

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        if name != "search_memory":
            return self.tools.call(name, arguments)
        query = arguments.get("query")
        if not isinstance(query, str):
            return "ERROR: search_memory needs a string 'query'", True
        self.calls += 1
        hits = self.store.search(self.user_id, query, self.k)
        out = "\n".join(f"[{m.id}] {m.text}" for m in hits) if hits else "(no memories found)"
        self.returned_text += [m.text for m in hits]
        self.returned_tokens += estimate_tokens(out)
        return out, False


# --- extractors (provided) ---------------------------------------------------------------------------------
class ScriptedExtractor:
    """Replays known operations per message text: the "gold" extraction for fixtures and tests."""

    def __init__(self, ops_by_text: dict[str, list[dict[str, Any]]]):
        self.ops_by_text = ops_by_text
        self.seen: list[str] = []

    def __call__(self, user_id: str, text: str, existing: list[Memory]) -> list[dict[str, Any]]:
        self.seen.append(text)
        return list(self.ops_by_text.get(text, []))


class ModelExtractor:
    """LLM extraction and consolidation in one call, as in the Mem0 paper: facts in, ADD/UPDATE/DELETE/NOOP out."""

    PROMPT = (
        "You maintain long-term memory about a user. Given the user's existing memories and one new message, "
        "return a JSON list of operations and nothing else. Operations: "
        '{"op": "ADD", "text": "<one short fact about the user, third person>", "key": "<snake_case slot, '
        'optional>"}, {"op": "UPDATE", "target": "<id or key>", "text": "<the new fact>"}, '
        '{"op": "DELETE", "target": "<id or key>"} when the user asks you to forget something. '
        "Only store durable facts, preferences and plans the user states about themselves. "
        "Return [] when there is nothing to remember.")

    def __init__(self, model: Model):
        self.model = model
        self.usage = Usage()

    def __call__(self, user_id: str, text: str, existing: list[Memory]) -> list[dict[str, Any]]:
        mem = "\n".join(f"{m.id} key={m.key or '-'}: {m.text}" for m in existing) or "(none)"
        resp = self.model.complete([{"role": "system", "content": self.PROMPT},
                                    {"role": "user", "content": f"Existing memories:\n{mem}\n\nNew message:\n{text}"}], [])
        self.usage += resp.usage
        return parse_ops(resp.content or "")


def parse_ops(text: str) -> list[dict[str, Any]]:
    """The first JSON list in a model reply, keeping only dict items; [] if there is none."""
    start = text.find("[")
    while start != -1:
        try:
            ops, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(ops, list):
                return [o for o in ops if isinstance(o, dict)]
        except json.JSONDecodeError:
            pass
        start = text.find("[", start + 1)
    return []
