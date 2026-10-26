"""Lab 1: the nanoos agent loop.

Start from nanoos/baseline.py (the provided minimal loop) and the exercise slide of the agent-loop deck, and
build a loop that survives real models and real tools. `make grade LAB=1` checks each behavior below against
the scripted model; `python run_suite.py --loop nanoos` runs the task suite with your loop.

What to build (search for TODO):

1. Budget and stop conditions. Stop after `max_steps` model calls with stopped_by="budget" and an empty
   answer. When exactly `wrap_up_at` steps are left, append WRAP_UP_MSG (formatted with left=...) as a user
   message before that model call, once. 0 turns it off.
2. Tool calls. Run all calls of one assistant turn at the same time when `parallel_tools` is on, one by one in
   order when it is off. Either way, append the tool results in the order of the calls, each linked to its
   call with "tool_call_id". Count every result that is an error (or blocked) in RunResult.tool_errors.
3. Retries and fallback. A model call that raises ModelError is retried up to `max_retries` times, sleeping
   backoff_s * 2**attempt before retry number attempt+1 (0.5 s, 1 s, 2 s, ...). Then the same for `fallback`,
   if set. If every attempt fails, stop with stopped_by="error".
4. Interrupts. When `interrupt` is set (before or during a model call), stop at once with
   stopped_by="interrupted", without waiting for a slow model call to return.
5. Steering. Before each model call, drain `steering` (a queue of strings) and append each one as a user
   message, after the tool results of the previous step.
6. Repetition. When the last `repeat_limit` tool calls (counted across steps, in order) all have the same
   signature(), append REPEAT_MSG (formatted with n=repeat_limit) as a user message after that step's tool
   results. Nudge once per streak: a further nudge needs `repeat_limit` more identical calls. 0 turns it off.
7. Hooks. Before each tool call, run the `pre_tool_call` hooks in list order: a hook that returns a string
   blocks the call (the tool does not run, its result is "BLOCKED: <string>", and it counts as an error) and
   the remaining pre hooks are skipped. After a call that ran, pass (call, output, is_error) through the
   `post_tool_call` hooks in order; a hook that returns a string replaces the output.

Emit the same trace events as the baseline (model_call, tool_call, run_end), plus "retry", "fallback",
"interrupt", "steer" and "nudge" events with whatever fields help you debug.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from models.scripted import ModelError  # noqa: F401 - raised by model clients; you will catch it
from nanoos.baseline import SYSTEM_PROMPT  # noqa: F401 - start the conversation the same way
from nanoos.trace import NullTracer  # noqa: F401
from nanoos.types import Model, ModelResponse, RunResult, ToolCall, Usage  # noqa: F401

WRAP_UP_MSG = "You have {left} steps left. Wrap up now: give your best final answer without more tool calls."
REPEAT_MSG = ("You have made the same tool call {n} times in a row with the same arguments. "
              "Stop repeating it: change approach or give your final answer.")

PreHook = Callable[[ToolCall], "str | None"]
PostHook = Callable[[ToolCall, str, bool], "str | None"]


@dataclass
class LoopConfig:
    max_steps: int = 20                  # model calls per run
    wrap_up_at: int = 3                  # steps left when WRAP_UP_MSG is sent (0: never)
    max_retries: int = 2                 # retries per model, per call
    backoff_s: float = 0.5               # first retry delay; doubles each retry
    fallback: Model | None = None        # tried after the primary model's retries are used up
    parallel_tools: bool = True
    repeat_limit: int = 3                # identical calls in a row before REPEAT_MSG (0: never)
    pre_tool_call: list[PreHook] = field(default_factory=list)
    post_tool_call: list[PostHook] = field(default_factory=list)
    steering: "queue.Queue[str] | None" = None
    interrupt: threading.Event | None = None


class Interrupted(Exception):
    """Raise this from your model-call code when the interrupt event is set."""


def signature(call: ToolCall) -> str:
    """Two calls are "the same" when their signatures are equal."""
    return call.name + ":" + json.dumps(call.arguments, sort_keys=True)


def call_model(model: Model, messages: list[dict[str, Any]], schemas: list[dict[str, Any]], cfg: LoopConfig,
               tracer: Any, step: int) -> ModelResponse:
    """One model call with retries, backoff, fallback (3) and interrupts (4).

    Raise Interrupted when cfg.interrupt is set, and ModelError when every attempt failed. Hint for (4): run
    model.complete in a background thread and wait on the interrupt event while the thread is alive.
    """
    # TODO(lab 1)
    raise NotImplementedError("Lab 1: implement call_model")


def run_tools(calls: list[ToolCall], tools: Any, cfg: LoopConfig, tracer: Any, step: int) -> list[tuple[str, bool]]:
    """Run one turn's tool calls (2) through the hooks (7); return (output, is_error) per call, in call order."""
    # TODO(lab 1)
    raise NotImplementedError("Lab 1: implement run_tools")


def run(model: Model, task_prompt: str, tools: Any, tracer: Any = None, max_steps: int = 20,
        config: LoopConfig | None = None) -> RunResult:
    """Run one task. Same signature as nanoos.baseline.run, plus `config`.

    Without a config, use LoopConfig(max_steps=max_steps); with one, its max_steps wins.
    """
    # TODO(lab 1): the loop itself: budget and wrap-up (1), steering (5), repetition (6), and the
    # bookkeeping RunResult needs (answer, steps, usage, tool_errors, stopped_by, messages).
    raise NotImplementedError("Lab 1: implement run")
