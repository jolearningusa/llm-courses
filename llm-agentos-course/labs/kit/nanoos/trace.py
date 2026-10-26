"""Lab 0: a JSONL trace logger, and two ways to read a run back.

Every loop in the kit reports what it does through a tracer:

    tracer.event("model_call", step=0, input_tokens=812, cached_input_tokens=0, output_tokens=40,
                 latency_s=0.9, n_tool_calls=1)

The baseline loop emits four kinds of events:

    run_start   task, model
    model_call  step, input_tokens, cached_input_tokens, output_tokens, latency_s, n_tool_calls
    tool_call   step, name, arguments, is_error, output_chars, latency_s
    run_end     stopped_by, steps

Your job (search for TODO): write the events to a file, then read a run back two ways. `make grade LAB=0`
checks your work. Lab 13 replaces this logger with OpenTelemetry spans that carry the same information.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TextIO

from nanoos.types import estimate_tokens


class NullTracer:
    """A tracer that records nothing. The suite falls back to it until JsonlTracer works."""

    def event(self, kind: str, **fields: Any) -> None:
        pass

    def close(self) -> None:
        pass


class JsonlTracer:
    """Writes one JSON object per line to `path`.

    Every record has these keys, plus the keyword arguments passed to event():

        seq     0, 1, 2, ... in the order events were recorded in this run
        ts      time.time() when the event was recorded
        run_id  the run_id given to the constructor
        kind    the event kind
    """

    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self.path.open("w", encoding="utf-8")
        self._seq = 0

    def event(self, kind: str, **fields: Any) -> None:
        # TODO(lab 0): build the record described in the class docstring, write it as one line of JSON
        # and flush, so the file can be read while the run is still going. Field values that JSON cannot
        # encode (Paths, sets, ...) should be written with str() rather than crash the run.
        raise NotImplementedError("Lab 0: implement JsonlTracer.event")

    def close(self) -> None:
        self._file.close()


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Load a trace file written by JsonlTracer."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals for one run, from its events.

    Return a dict with exactly these keys:

        model_calls, tool_calls, tool_errors                      counts
        input_tokens, cached_input_tokens, output_tokens          sums over model_call events
        tools_used                                                {tool name: number of calls}
        wall_s                                                    ts of the last event minus ts of the first
                                                                  (0.0 for fewer than two events)
    """
    # TODO(lab 0)
    raise NotImplementedError("Lab 0: implement summarize_trace")


def token_breakdown(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, int]:
    """Where the tokens of one prompt go, estimated with nanoos.types.estimate_tokens.

    Return a dict with exactly these keys:

        system        content of "system" messages
        tools         the tool schemas, serialized with json.dumps(tools)
        tool_results  content of "tool" messages
        history       everything else: content of "user" and "assistant" messages, plus the
                      json.dumps() of each assistant message's "tool_calls" list when it has one

    Estimate each piece of text separately and add the estimates up (do not concatenate first).
    Treat a missing or None content as empty.
    """
    # TODO(lab 0)
    raise NotImplementedError("Lab 0: implement token_breakdown")
