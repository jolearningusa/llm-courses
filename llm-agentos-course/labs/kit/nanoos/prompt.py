"""Lab 5: prompt prefix stability for nanoos.

A prompt-prefix cache only helps when each request starts with exactly the tokens of an earlier one. vLLM finds
those tokens by hashing the prompt in fixed-size blocks, each hash chained to the one before it, so one changed
token invalidates every block after it (KV-cache deck, "Automatic prefix caching: hash chains"). This module
reproduces that on the harness side, so you can see which turn broke the cache and why before you pay for it:

    hashes = block_hashes(cache_tokens(render(prompt)))       # vLLM-style chained block hashes
    cache = PrefixCache(capacity_blocks=3072)
    hit_blocks = cache.access(hashes)                         # a simulated prefix cache with LRU eviction
    brk = diagnose(previous_prompt, prompt)                   # where the prefix broke, and the likely cause

Implement the three parts marked TODO (`make grade LAB=5`): `block_hashes`, `PrefixCache.access` and
`diagnose`. Rendering, the stand-in tokenizer and CacheAuditModel are provided. cache_experiment.py runs the
deck's "break the cache on purpose" exercise, offline or against a vLLM server.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from nanoos.types import Model, ModelResponse

BLOCK_SIZE = 16  # vLLM's default KV block size, in tokens
MEMORY_MARKER = "What you remember about this user"  # nanoos.memory.MEMORY_HEADER starts with this
TIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}")


@dataclass
class Prompt:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)


# --- rendering and tokens (provided) ----------------------------------------------------------------------
def segments(prompt: Prompt) -> list[tuple[str, str]]:
    """The prompt as (label, text) pieces in the order providers cache them: tools, then system, then messages.

    Labels are "tools", "system" and "message:<i>:<role>" (i is the index in prompt.messages). Rendering is
    deterministic (sorted JSON keys), so the same prompt always gives the same text.
    """
    out = [("tools", "<|tools|>" + json.dumps(prompt.tools, sort_keys=True, separators=(",", ":")))]
    for i, m in enumerate(prompt.messages):
        text = f"<|{m['role']}|>" + (m.get("content") or "")
        if m.get("tool_calls"):
            text += json.dumps(m["tool_calls"], sort_keys=True, separators=(",", ":"))
        if m.get("tool_call_id"):
            text += f"<|id={m['tool_call_id']}|>"
        out.append(("system" if i == 0 and m["role"] == "system" else f"message:{i}:{m['role']}", text))
    return out


def render(prompt: Prompt) -> str:
    return "".join(text for _, text in segments(prompt))


_TOKEN_RE = re.compile(r"\w+|[^\w\s]|\s+")


def cache_tokens(text: str) -> list[str]:
    """A stand-in tokenizer: words, single punctuation marks and runs of whitespace.

    Real tokenizers split differently but teach the same lesson: what matters is where the first changed token
    is. "".join(cache_tokens(text)) == text, and appending text never changes the earlier tokens as long as the
    appended part starts with punctuation, as every rendered message does.
    """
    return _TOKEN_RE.findall(text)


# --- the parts you write -----------------------------------------------------------------------------------
def block_hashes(tokens: list[str], block_size: int = BLOCK_SIZE, salt: str = "") -> list[str]:
    """vLLM-style chained hashes of the full blocks of `tokens`.

    - Only full blocks are hashed: len(result) == len(tokens) // block_size.
    - hash[i] = sha256 hex digest of (hash[i-1] + "\\x00" + "\\x00".join(tokens of block i)), encoded as UTF-8,
      where hash[-1] is the empty string for block 0 when there is no salt, and the salt (a per-tenant
      cache_salt) otherwise. So the salt changes every hash, and a changed token changes its block's hash
      and every later one.
    """
    # TODO(lab 5)
    raise NotImplementedError("Lab 5: implement block_hashes")


class PrefixCache:
    """A simulated engine prefix cache: a fixed number of KV blocks, keyed by chained block hash."""

    def __init__(self, capacity_blocks: int):
        self.capacity_blocks = capacity_blocks
        self.blocks: OrderedDict[str, None] = OrderedDict()  # least recently used first
        self.evictions = 0

    def access(self, hashes: list[str]) -> int:
        """Serve one request whose prompt has these block hashes; return how many blocks were cache hits.

        - Hits: walk the hashes from block 0 and stop at the first one not in the cache (as vLLM does). A block
          that is cached after a miss does not count, because its chained hash would not be looked up.
        - Then the request's blocks are all in the cache: insert the missing ones and mark every one as used.
          Mark them from the last block to block 0, so that block 0 is the most recently used and a request's
          tail is evicted before its head (vLLM frees a finished request's blocks in reverse order).
        - Finally evict least recently used blocks until at most capacity_blocks remain, counting each in
          self.evictions.
        """
        # TODO(lab 5)
        raise NotImplementedError("Lab 5: implement PrefixCache.access")


@dataclass
class Break:
    cause: str              # "first" | "none" | "tools" | "memory" | "timestamp" | "system" | "history"
    block: int | None       # index of the first block whose hash changed (None for "first" and "none")
    segment: str | None     # label of the segment (see segments) that holds the first changed token


def diagnose(prev: Prompt | None, cur: Prompt, block_size: int = BLOCK_SIZE) -> Break:
    """Compare a session's previous prompt with the current one and say where the prefix broke, and why.

    - No previous prompt: Break("first", None, None).
    - Tokenize both renders with cache_tokens. If every full block of prev is still a prefix of cur (the
      first (len(prev) // block_size) * block_size tokens are equal), nothing was lost: Break("none", None, None).
    - Otherwise find i, the first token index where they differ (or the length of the shorter one). The
      block is i // block_size. The segment is the label of cur's segment that contains token i; work in
      characters: token i starts at character pos = len("".join(tokens[:i])) of the render. Use prev's
      tokens and segments if cur has no token i.
    - The cause comes from that segment:
        "tools"      the tool list changed
        "system"     the system prompt changed. Look at the system text of cur before character position
                     (pos minus where the system segment starts): if it contains MEMORY_MARKER, the change is
                     inside the memory block, so the cause is "memory". Otherwise, if the line of cur's
                     system text that holds that position matches TIME_RE, the cause is "timestamp". Otherwise
                     it stays "system".
        message:...  an earlier message was rewritten or removed (compaction, tool-result clearing, or
                     unstable serialization): "history".
    """
    # TODO(lab 5)
    raise NotImplementedError("Lab 5: implement diagnose")


# --- a model wrapper that audits every call (provided) -----------------------------------------------------
class CacheAuditModel:
    """Wraps a model and records, for every call, the prompt's blocks, simulated hits and the break diagnosis.

    model = CacheAuditModel(model, PrefixCache(capacity_blocks=100_000))
    ... run the loop ...
    model.rows  -> one dict per call: prompt_tokens, hit_tokens, cause, block, segment
    """

    def __init__(self, model: Model, cache: PrefixCache | None = None, block_size: int = BLOCK_SIZE, salt: str = ""):
        self.model = model
        self.cache = cache or PrefixCache(capacity_blocks=10 ** 9)
        self.block_size = block_size
        self.salt = salt
        self.name = f"{model.name}+audit"
        self.rows: list[dict[str, Any]] = []
        self._prev: Prompt | None = None

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        prompt = Prompt(json.loads(json.dumps(messages)), json.loads(json.dumps(tools)))
        tokens = cache_tokens(render(prompt))
        hits = self.cache.access(block_hashes(tokens, self.block_size, self.salt))
        brk = diagnose(self._prev, prompt, self.block_size)
        self._prev = prompt
        self.rows.append({"call": len(self.rows), "prompt_tokens": len(tokens), "hit_tokens": hits * self.block_size,
                          "cause": brk.cause, "block": brk.block, "segment": brk.segment})
        return self.model.complete(messages, tools)
