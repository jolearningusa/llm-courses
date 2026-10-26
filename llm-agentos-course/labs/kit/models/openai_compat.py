"""A client for any OpenAI-compatible chat completions endpoint, using only the standard library.

Works with hosted APIs that speak this format and with a local vLLM server (`vllm serve ...` exposes
/v1/chat/completions; see labs/lab2_aios_kv_cache/run_vllm_lmcache.sh). Configure it with environment
variables, or pass the values to the constructor:

    NANOOS_BASE_URL   e.g. http://localhost:8000/v1 or your provider's base URL
    NANOOS_API_KEY    the API key (any string for a local vLLM server)
    NANOOS_MODEL      the model name the endpoint expects

There is deliberately no retry logic here: Lab 1 adds retries and fallback in the loop.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from models.scripted import ModelError
from nanoos.types import ModelResponse, ToolCall, Usage


class OpenAICompatModel:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None,
                 timeout_s: float = 120.0, temperature: float | None = None, max_tokens: int | None = None):
        self.base_url = (base_url or os.environ.get("NANOOS_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("NANOOS_API_KEY", "")
        self.name = model or os.environ.get("NANOOS_MODEL", "")
        if not self.base_url or not self.name:
            raise ValueError("set NANOOS_BASE_URL and NANOOS_MODEL (and NANOOS_API_KEY for a hosted API)")
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens  # Lab 5 sets 1 to time the prefill (time to first token)

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        body: dict[str, Any] = {"model": self.name, "messages": messages}
        if tools:
            body["tools"] = tools
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            raise ModelError(f"HTTP {e.code}: {e.read()[:500].decode(errors='replace')}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            raise ModelError(f"request failed: {e}") from None
        return parse_response(data)


def parse_response(data: dict[str, Any]) -> ModelResponse:
    """Turn a chat completions response body into a ModelResponse."""
    msg = data["choices"][0]["message"]
    calls = []
    for c in msg.get("tool_calls") or []:
        raw = c["function"].get("arguments") or "{}"
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            args = {"_unparsed": raw}  # a malformed call; the tool layer reports it to the model
        calls.append(ToolCall(id=c["id"], name=c["function"]["name"], arguments=args if isinstance(args, dict) else {}))
    u = data.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    usage = Usage(input_tokens=u.get("prompt_tokens", 0), cached_input_tokens=cached,
                  output_tokens=u.get("completion_tokens", 0))
    return ModelResponse(content=msg.get("content"), tool_calls=calls, usage=usage)
