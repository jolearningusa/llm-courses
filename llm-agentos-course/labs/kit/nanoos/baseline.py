"""The baseline loop: the smallest loop that can finish a basic-tier task.

It is provided so that Lab 0 can run the task suite before anyone has written a loop. Lab 1 writes a
better one in nanoos/loop.py (budgets, parallel tools, retries, interrupts, hooks) with the same
signature, and run_suite.py --loop nanoos switches to it.
"""

from __future__ import annotations

import json
import time
from typing import Any

from nanoos.trace import NullTracer
from nanoos.types import Model, RunResult, Usage

SYSTEM_PROMPT = (
    "You are a careful assistant working in a workspace of files. Use the tools to look at the files "
    "before you answer. When the task asks a question, end with a short final answer and no tool calls."
)


def run(model: Model, task_prompt: str, tools: Any, tracer: Any = None, max_steps: int = 20) -> RunResult:
    tracer = tracer or NullTracer()
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT},
                                      {"role": "user", "content": task_prompt}]
    schemas = tools.schemas()
    usage, tool_errors = Usage(), 0
    for step in range(max_steps):
        t0 = time.time()
        resp = model.complete(messages, schemas)
        usage += resp.usage
        tracer.event("model_call", step=step, input_tokens=resp.usage.input_tokens,
                     cached_input_tokens=resp.usage.cached_input_tokens, output_tokens=resp.usage.output_tokens,
                     latency_s=round(time.time() - t0, 4), n_tool_calls=len(resp.tool_calls))
        messages.append(resp.to_message())
        if not resp.tool_calls:
            tracer.event("run_end", stopped_by="answer", steps=step + 1)
            return RunResult(resp.content or "", step + 1, usage, tool_errors, 0, "answer", messages)
        for call in resp.tool_calls:  # one at a time, in order
            t0 = time.time()
            out, is_error = tools.call(call.name, call.arguments)
            tool_errors += is_error
            tracer.event("tool_call", step=step, name=call.name, arguments=json.dumps(call.arguments)[:500],
                         is_error=is_error, output_chars=len(out), output=out[:2000],
                         latency_s=round(time.time() - t0, 4))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
    tracer.event("run_end", stopped_by="budget", steps=max_steps)
    return RunResult("", max_steps, usage, tool_errors, 0, "budget", messages)
