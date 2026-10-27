"""Lab 13: OpenTelemetry spans for nanoos, following the GenAI semantic conventions, with content redacted.

Lab 0's JsonlTracer writes flat events. OtelTracer takes the same events (it is a drop-in tracer for every loop
in the kit) and turns each run into a tree of spans:

    invoke_agent nanoos                  the run (root span)
    ├── turn 0                           one loop iteration: a model call and the tools it asked for
    │   ├── chat demo:b07_two_hop        gen_ai.operation.name = chat, token usage, finish reason
    │   ├── retrieval memory             a memory or document search, if the loop reports one
    │   └── execute_tool read_file       gen_ai.tool.name, arguments and result (redacted), status
    └── turn 1
        └── ...

Span names and attributes follow the OpenTelemetry semantic conventions for generative AI (status: Development,
now in github.com/open-telemetry/semantic-conventions-genai): `{operation} {target}` names and `gen_ai.*`
attributes. Attributes the conventions do not cover use the `nanoos.` prefix. The spans are exported as OTLP JSON,
to a file or to any OTLP/HTTP endpoint that accepts JSON (Jaeger, an OpenTelemetry Collector).

Content (tool arguments and results, model text, retrieval queries) is captured **redacted** by default:
capture_content="redacted" runs every string through a Redactor before it reaches a span. "off" drops content
and "full" keeps it as is, as OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true does.

Your job (search for TODO): Redactor.redact, and OtelTracer.event, which builds the tree. `make grade LAB=13`
checks your work.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCOPE = "nanoos.otel"

# (kind, pattern) pairs for secrets that look like secrets. A match is replaced by "[REDACTED:<kind>]".
SECRET_PATTERNS: list[tuple[str, str]] = [
    ("private_key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    ("api_key", r"\bsk-[A-Za-z0-9_-]{16,}"),
    ("aws_key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("github_token", r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    ("bearer", r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
]
# Environment variables whose values count as secrets, by name.
SECRET_ENV = re.compile(r"(?i)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)")

# Event fields that carry content (they are redacted, or dropped when capture_content is "off").
CONTENT_FIELDS = {"arguments", "output", "content", "query", "prompt", "error", "call"}


def env_secrets(environ: dict[str, str] | None = None, min_len: int = 8) -> list[str]:
    """Values of environment variables whose names look secret (API_KEY, GH_TOKEN, ...)."""
    environ = os.environ if environ is None else environ
    return [v for k, v in environ.items() if SECRET_ENV.search(k) and len(v) >= min_len]


class Redactor:
    """Removes secrets from text before it reaches a span.

    - Every SECRET_PATTERNS match becomes "[REDACTED:<kind>]" (kinds in list order, so a private key block is
      replaced before the patterns inside it could match).
    - Every exact occurrence of a known secret value (the `secrets` argument; by default env_secrets()) becomes
      "[REDACTED:secret]". Replace longer values first, so a value that contains another is removed whole.
    - The result is cut to max_chars characters, with "...[truncated N chars]" appended, where N is the number
      of characters removed. max_chars=0 means no limit.
    """

    def __init__(self, secrets: list[str] | None = None, patterns: list[tuple[str, str]] | None = None,
                 max_chars: int = 2000):
        self.secrets = env_secrets() if secrets is None else list(secrets)
        self.patterns = [(k, re.compile(p)) for k, p in (SECRET_PATTERNS if patterns is None else patterns)]
        self.max_chars = max_chars

    def redact(self, text: str) -> str:
        # TODO(lab 13)
        raise NotImplementedError("Lab 13: implement Redactor.redact")

    def redact_value(self, value: Any) -> Any:
        """redact() applied to every string inside dicts, lists and tuples; other values unchanged."""
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, dict):
            return {k: self.redact_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.redact_value(v) for v in value]
        return value


@dataclass
class Span:
    name: str
    trace_id: str                      # 32 hex characters, shared by every span of a run
    span_id: str                       # 16 hex characters
    parent_id: str | None              # None for the root span
    start_ns: int
    end_ns: int | None = None          # None while the span is open
    kind: str = "INTERNAL"             # INTERNAL or CLIENT (model calls are CLIENT spans)
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)  # {"name", "time_ns", "attributes"}
    status: str = "UNSET"              # UNSET, OK or ERROR
    status_message: str = ""

    def end(self, end_ns: int) -> None:
        if self.end_ns is None:
            self.end_ns = max(end_ns, self.start_ns)


class IdGenerator:
    """Random trace and span ids; pass a seed for ids that repeat from run to run."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def trace_id(self) -> str:
        return f"{self._rng.getrandbits(128):032x}"

    def span_id(self) -> str:
        return f"{self._rng.getrandbits(64):016x}"


def _otlp_value(v: Any) -> dict[str, Any]:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, (list, tuple)):
        return {"arrayValue": {"values": [_otlp_value(x) for x in v]}}
    if isinstance(v, str):
        return {"stringValue": v}
    return {"stringValue": json.dumps(v, default=str)}


def _otlp_attrs(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _otlp_value(v)} for k, v in attrs.items() if v is not None]


_KIND = {"INTERNAL": 1, "SERVER": 2, "CLIENT": 3}
_STATUS = {"UNSET": 0, "OK": 1, "ERROR": 2}


def to_otlp(spans: list[Span], service_name: str = "nanoos",
            resource: dict[str, Any] | None = None) -> dict[str, Any]:
    """The OTLP/JSON body (ExportTraceServiceRequest) for a list of finished spans."""
    out = []
    for s in spans:
        d: dict[str, Any] = {"traceId": s.trace_id, "spanId": s.span_id, "name": s.name, "kind": _KIND[s.kind],
                             "startTimeUnixNano": str(s.start_ns), "endTimeUnixNano": str(s.end_ns or s.start_ns),
                             "attributes": _otlp_attrs(s.attributes),
                             "events": [{"name": e["name"], "timeUnixNano": str(e["time_ns"]),
                                         "attributes": _otlp_attrs(e.get("attributes", {}))} for e in s.events],
                             "status": {"code": _STATUS[s.status], **({"message": s.status_message}
                                                                      if s.status_message else {})}}
        if s.parent_id:
            d["parentSpanId"] = s.parent_id
        out.append(d)
    res = {"service.name": service_name, "telemetry.sdk.name": "nanoos", **(resource or {})}
    return {"resourceSpans": [{"resource": {"attributes": _otlp_attrs(res)},
                               "scopeSpans": [{"scope": {"name": SCOPE}, "spans": out}]}]}


def _from_value(v: dict[str, Any]) -> Any:
    if "intValue" in v:
        return int(v["intValue"])
    if "arrayValue" in v:
        return [_from_value(x) for x in v["arrayValue"].get("values", [])]
    return next(iter(v.values()), None)


def from_otlp(body: dict[str, Any]) -> list[Span]:
    """Spans back from an OTLP/JSON body (for tests, the dashboard and your own analysis)."""
    kinds = {v: k for k, v in _KIND.items()}
    codes = {v: k for k, v in _STATUS.items()}
    spans = []
    for rs in body.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for d in ss.get("spans", []):
                attrs = {a["key"]: _from_value(a["value"]) for a in d.get("attributes", [])}
                events = [{"name": e["name"], "time_ns": int(e["timeUnixNano"]),
                           "attributes": {a["key"]: _from_value(a["value"]) for a in e.get("attributes", [])}}
                          for e in d.get("events", [])]
                st = d.get("status", {})
                spans.append(Span(d["name"], d["traceId"], d["spanId"], d.get("parentSpanId"),
                                  int(d["startTimeUnixNano"]), int(d["endTimeUnixNano"]), kinds.get(d.get("kind"),
                                  "INTERNAL"), attrs, events, codes.get(st.get("code", 0), "UNSET"),
                                  st.get("message", "")))
    return spans


class InMemoryExporter:
    """Keeps exported spans in a list (tests, the dashboard)."""

    def __init__(self):
        self.spans: list[Span] = []

    def export(self, spans: list[Span]) -> None:
        self.spans.extend(spans)


class OtlpFileExporter:
    """Writes one OTLP/JSON file per export call (one run, since OtelTracer exports when it closes)."""

    def __init__(self, path: str | Path, service_name: str = "nanoos", resource: dict[str, Any] | None = None):
        self.path, self.service_name, self.resource = Path(path), service_name, resource

    def export(self, spans: list[Span]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(to_otlp(spans, self.service_name, self.resource), indent=1),
                             encoding="utf-8")


class OtlpHttpExporter:
    """POSTs OTLP/JSON to <endpoint>/v1/traces (for example Jaeger at http://localhost:4318).

    Fail-open, as telemetry must never break the agent: a failed export is counted in `failures` and printed
    once, and the run goes on.
    """

    def __init__(self, endpoint: str = "http://localhost:4318", service_name: str = "nanoos",
                 headers: dict[str, str] | None = None, timeout_s: float = 5.0, resource: dict[str, Any] | None = None):
        self.url = endpoint.rstrip("/") + ("" if endpoint.rstrip("/").endswith("/v1/traces") else "/v1/traces")
        self.service_name, self.headers, self.timeout_s = service_name, headers or {}, timeout_s
        self.resource = resource
        self.failures = 0

    def export(self, spans: list[Span]) -> None:
        body = json.dumps(to_otlp(spans, self.service_name, self.resource)).encode()
        req = urllib.request.Request(self.url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", **self.headers})
        try:
            urllib.request.urlopen(req, timeout=self.timeout_s).read()
        except Exception as e:  # noqa: BLE001 - fail open
            self.failures += 1
            if self.failures == 1:
                print(f"note: OTLP export to {self.url} failed ({e}); runs continue without it")


def span_tree(spans: list[Span]) -> str:
    """An indented text view of one trace: name, duration and status per span, children in start order."""
    children: dict[str | None, list[Span]] = {}
    ids = {s.span_id for s in spans}
    for s in spans:
        children.setdefault(s.parent_id if s.parent_id in ids else None, []).append(s)
    lines: list[str] = []

    def walk(parent: str | None, depth: int) -> None:
        for s in sorted(children.get(parent, []), key=lambda x: (x.start_ns, x.name)):
            ms = ((s.end_ns or s.start_ns) - s.start_ns) / 1e6
            flag = "  ERROR" if s.status == "ERROR" else ""
            lines.append(f"{'  ' * depth}{s.name}  {ms:.1f} ms{flag}")
            walk(s.span_id, depth + 1)

    walk(None, 0)
    return "\n".join(lines)


def _ns(ts: float) -> int:
    return int(ts * 1e9)


class OtelTracer:
    """A tracer (same `event(kind, **fields)` interface as Lab 0's) that builds one span tree per run.

    Spans, from the events every loop in the kit emits (see nanoos/trace.py for their fields):

    - **Root.** `invoke_agent {agent_name}`, opened by the first event of the run (normally run_start) and
      ended by run_end. Attributes: gen_ai.operation.name "invoke_agent", gen_ai.agent.name, and
      gen_ai.conversation.id = run_id; from run_start, gen_ai.request.model (its `model` field) and every other
      field as `nanoos.<field>` (task, trial, loop, ...); from run_end, nanoos.stopped_by and nanoos.steps.
      A run that stopped with "error" gets status ERROR.
    - **Turns.** `turn {step}` under the root, one per distinct `step` value, opened by the first event that
      carries that step. Attribute nanoos.step. A turn ends when a later step starts or the run ends.
    - **Model calls.** `chat {model}` (kind CLIENT) under the current turn, for each model_call event. `model`
      comes from the event's `model` field, or else the run's gen_ai.request.model. The call happened *before*
      the event, so the span starts at the event time minus latency_s and ends at the event time. Attributes:
      gen_ai.operation.name "chat", gen_ai.request.model, gen_ai.usage.input_tokens, gen_ai.usage.output_tokens,
      gen_ai.usage.cache_read.input_tokens (from cached_input_tokens), and gen_ai.response.finish_reasons:
      ["tool_calls"] when n_tool_calls > 0, else ["stop"]. A `content` field is content (gen_ai.output.messages).
    - **Tool calls.** `execute_tool {name}` under the current turn, timed the same way. Attributes:
      gen_ai.operation.name "execute_tool", gen_ai.tool.name, gen_ai.tool.type "function"; content
      gen_ai.tool.call.arguments (from `arguments`) and gen_ai.tool.call.result (from `output`). is_error=True
      sets status ERROR and error.type "tool_error".
    - **Retrieval.** `retrieval {source}` under the current turn, timed the same way, for a retrieval event
      (fields: source, query, n_results, latency_s). Attributes: gen_ai.operation.name "retrieval",
      gen_ai.data_source.id = source, nanoos.n_results; content gen_ai.retrieval.query.text (from `query`).
    - **Everything else** (retry, fallback, steer, nudge, interrupt, ...) is a span event named after its kind,
      on the current turn, or on the root when no turn is open, with its fields as attributes (content fields
      handled as below).

    Content: the fields in CONTENT_FIELDS. With capture_content "redacted" (default) a content value is passed
    through self.redactor.redact_value; with "off" it is left out; with "full" it is kept. Non-string content is
    serialized with json.dumps first. Span times are nanoseconds (event ts * 1e9). Parents always contain their
    children: when a child starts before its turn or root, move that parent's start back.

    close() ends every open span at the last event's time and exports all spans of the run in one call, in the
    order they were opened. It may be called more than once; only the first call exports.
    """

    def __init__(self, exporter: Any, run_id: str = "run", agent_name: str = "nanoos",
                 capture_content: str = "redacted", redactor: Redactor | None = None, ids: IdGenerator | None = None,
                 clock: Any = time.time):
        if capture_content not in ("off", "redacted", "full"):
            raise ValueError("capture_content must be off, redacted or full")
        self.exporter, self.run_id, self.agent_name = exporter, run_id, agent_name
        self.capture_content = capture_content
        self.redactor = redactor or Redactor()
        self.ids = ids or IdGenerator()
        self.clock = clock
        self.trace_id = self.ids.trace_id()
        self.spans: list[Span] = []          # every span of the run, in the order they were opened
        self.root: Span | None = None
        self.turn: Span | None = None
        self.model = ""
        self.last_ns = 0
        self.closed = False

    # -- provided helpers -----------------------------------------------------------------------------------

    def _content(self, value: Any) -> Any:
        """A content value as it should appear on a span, or None when content is not captured."""
        if self.capture_content == "off" or value is None:
            return None
        if not isinstance(value, str):
            value = json.dumps(value, default=str)
        return value if self.capture_content == "full" else self.redactor.redact(value)

    def _open(self, name: str, parent: Span | None, start_ns: int, kind: str = "INTERNAL",
              attributes: dict[str, Any] | None = None) -> Span:
        span = Span(name, self.trace_id, self.ids.span_id(), parent.span_id if parent else None, start_ns,
                    kind=kind, attributes=dict(attributes or {}))
        self.spans.append(span)
        return span

    def score(self, name: str, value: float | bool | str, comment: str = "") -> None:
        """Attach a grader's result to the run, on the root span: attribute nanoos.score.<name>, plus a span
        event "score" (as Langfuse and Phoenix attach scores to traces). Call it before close()."""
        if self.root is None:
            return
        self.root.attributes[f"nanoos.score.{name}"] = value
        self.root.events.append({"name": "score", "time_ns": self.last_ns or self.root.start_ns,
                                 "attributes": {"name": name, "value": value, "comment": comment}})

    # -- your part ------------------------------------------------------------------------------------------

    def event(self, kind: str, **fields: Any) -> None:
        # TODO(lab 13): build the span tree described in the class docstring. Keep the open root and turn in
        # self.root and self.turn, open spans with self._open(), and pass content through self._content().
        raise NotImplementedError("Lab 13: implement OtelTracer.event")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        end = self.last_ns or _ns(self.clock())
        for s in self.spans:
            s.end(end)
        if self.spans:
            self.exporter.export(list(self.spans))
