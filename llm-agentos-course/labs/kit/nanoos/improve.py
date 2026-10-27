"""Lab 14: a harness that improves itself. Propose edits from failed runs, probe each one, compose what holds.

The model's weights stay fixed; the harness changes, as in Harness² (Module 7). A HarnessConfig is the editable
part of nanoos: the system prompt, the tool descriptions and the settings. One improvement round:

    1. Evaluate the current harness on the DEV tasks; collect the failed runs.
    2. Propose: a proposer (a model, or the scripted one offline) reads the failures and suggests edits.
    3. Probe: run the failing DEV tasks with each edit applied alone. An edit that fixes none is rejected.
    4. Compose: add the surviving edits one at a time, best first. Keep an edit only if, together with the edits
       already kept, it raises DEV success without lowering it, and does not lower success on the HELDOUT tasks,
       which the proposer never sees.
    5. Log every decision as a lesson record (schema llm-agentos-course/lesson/v1, as in
       labs/lab3_google_harness2/sample_lesson_entry.jsonl).

Edit targets:

    system_prompt          op "append": add one line to the system prompt
    tool:<name>            op "set": replace that tool's description
    setting:max_steps      op "set": an integer from 1 to 50

Your job (search for TODO): parse_edits, probe, compose and improve. `make grade LAB=14` checks your work.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from nanoos import baseline
from nanoos.tools_basic import WorkspaceTools
from nanoos.types import ModelResponse

SCHEMA = "llm-agentos-course/lesson/v1"
TOOL_NAMES = ("list_files", "read_file", "write_file", "search")
SETTINGS = {"max_steps": (1, 50)}


@dataclass
class HarnessConfig:
    system_prompt: str = "You are an assistant with tools for a workspace of files."
    tool_descriptions: dict[str, str] = field(default_factory=dict)   # overrides; missing = the default
    settings: dict[str, Any] = field(default_factory=lambda: {"max_steps": 4})

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Edit:
    target: str                 # system_prompt, tool:<name> or setting:<name>
    op: str                     # append (system_prompt) or set
    value: Any
    rationale: str = ""
    failure_signature: str = ""  # the failure it was proposed for

    def describe(self) -> str:
        return f"{self.target} {self.op} {json.dumps(self.value)}"


@dataclass
class RunOutcome:
    task_id: str
    trial: int
    success: bool
    answer: str
    steps: int
    stopped_by: str
    tool_calls: list[dict[str, Any]]           # [{"name", "arguments"}] in order
    tokens: int


@dataclass
class Probe:
    edit: Edit
    tasks: list[str]                            # the failing DEV tasks it was probed on
    fixed: list[str]                            # tasks that pass with the edit (in every trial)
    still_failing: list[str]
    runs: int                                   # agent runs the probe cost
    adopted: bool


# -- provided: applying edits and running the harness ------------------------------------------------------------

def apply_edit(config: HarnessConfig, edit: Edit) -> HarnessConfig:
    """A new config with the edit applied (the old one is unchanged)."""
    new = copy.deepcopy(config)
    if edit.target == "system_prompt":
        new.system_prompt = new.system_prompt.rstrip() + "\n" + str(edit.value).strip()
    elif edit.target.startswith("tool:"):
        new.tool_descriptions[edit.target[5:]] = str(edit.value)
    elif edit.target.startswith("setting:"):
        new.settings[edit.target[8:]] = edit.value
    else:
        raise ValueError(f"unknown target {edit.target!r}")
    return new


def conflicts(a: Edit, b: Edit) -> bool:
    """Two edits conflict when both set the same target (appends to the system prompt never conflict)."""
    return a.target == b.target and "set" in (a.op, b.op)


class DescribedTools:
    """Tools with some descriptions replaced, as a HarnessConfig asks."""

    def __init__(self, tools: Any, descriptions: dict[str, str]):
        self.tools, self.descriptions = tools, descriptions

    def schemas(self) -> list[dict[str, Any]]:
        out = copy.deepcopy(self.tools.schemas())
        for s in out:
            if s["function"]["name"] in self.descriptions:
                s["function"]["description"] = self.descriptions[s["function"]["name"]]
        return out

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        return self.tools.call(name, arguments)


class PromptedModel:
    """A model whose system message is the config's system prompt, whatever the loop put there."""

    def __init__(self, model: Any, system_prompt: str):
        self.model, self.system_prompt = model, system_prompt
        self.name = getattr(model, "name", "model")

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        msgs = [dict(m, content=self.system_prompt) if m.get("role") == "system" else m for m in messages]
        return self.model.complete(msgs, tools)


def run_task(config: HarnessConfig, task: Any, model: Any, workspace: Path, trial: int = 0,
             loop: Any = baseline) -> RunOutcome:
    """Run one task under a harness config and check it."""
    workspace.mkdir(parents=True, exist_ok=True)
    task.setup(workspace)
    tools = DescribedTools(WorkspaceTools(workspace, task.tools), config.tool_descriptions)
    result = loop.run(PromptedModel(model, config.system_prompt), task.prompt, tools, None,
                      max_steps=int(config.settings.get("max_steps", 20)))
    calls = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
             for m in result.messages for c in m.get("tool_calls") or []]
    return RunOutcome(task.id, trial, bool(task.check(workspace, result.answer)), result.answer, result.steps,
                      result.stopped_by, calls, result.usage.input_tokens + result.usage.output_tokens)


class Evaluator:
    """Runs task sets under configs, counting the agent runs and tokens the improvement loop spends.

    make_model(task_id) returns a fresh model for one run (HabitModel offline, your API model otherwise).
    """

    def __init__(self, tasks: list[Any], make_model: Callable[[str], Any], workdir: Path, trials: int = 1,
                 loop: Any = baseline):
        self.tasks = {t.id: t for t in tasks}
        self.make_model, self.workdir, self.trials, self.loop = make_model, Path(workdir), trials, loop
        self.runs = 0
        self.tokens = 0

    def run(self, config: HarnessConfig, task_ids: list[str] | tuple[str, ...]) -> dict[str, list[RunOutcome]]:
        """{task_id: [one RunOutcome per trial]} for the given tasks under `config`."""
        out: dict[str, list[RunOutcome]] = {}
        for tid in task_ids:
            for trial in range(self.trials):
                self.runs += 1
                ws = self.workdir / f"run{self.runs:05d}"
                o = run_task(config, self.tasks[tid], self.make_model(tid), ws, trial, self.loop)
                self.tokens += o.tokens
                out.setdefault(tid, []).append(o)
        return out


def passed(results: dict[str, list[RunOutcome]]) -> set[str]:
    """Tasks that passed in every trial."""
    return {tid for tid, runs in results.items() if all(r.success for r in runs)}


def success(results: dict[str, list[RunOutcome]]) -> float:
    """Share of tasks that passed in every trial (pass^k over the task set)."""
    return len(passed(results)) / len(results) if results else 0.0


def signature(o: RunOutcome) -> str:
    """A short failure signature for a failed run, from what the run did."""
    if o.stopped_by == "budget":
        return "step_budget"
    if not o.tool_calls:
        return "answered_without_tools"
    if o.tool_calls[-1]["name"] == "write_file":
        return "bad_write"
    return "wrong_answer"


def failure_report(results: dict[str, list[RunOutcome]], prompts: dict[str, str]) -> list[dict[str, Any]]:
    """What the proposer is shown about each failed DEV task: prompt, signature, calls and answer of a failed run."""
    out = []
    for tid, runs in sorted(results.items()):
        bad = next((r for r in runs if not r.success), None)
        if bad is None:
            continue
        out.append({"task_id": tid, "prompt": prompts.get(tid, ""), "signature": signature(bad),
                    "steps": bad.steps, "stopped_by": bad.stopped_by, "tool_calls": bad.tool_calls[:12],
                    "answer": bad.answer[:500]})
    return out


def lesson_record(lesson_id: str, edit: Edit, task_ids: list[str], decision: str, contrast: str,
                  evidence: list[str], **probe: Any) -> dict[str, Any]:
    """One decision in the course's lesson schema (see labs/lab3_google_harness2/sample_lesson_entry.jsonl)."""
    component = edit.target.split(":")[0]
    component = {"system_prompt": "system_prompt", "tool": "tool_guidance", "setting": "settings"}[component]
    return {"schema": SCHEMA, "lesson_id": lesson_id, "task_id": ",".join(task_ids), "target_component": component,
            "failure_signature": edit.failure_signature, "proposed_edit": edit.describe(),
            "probe": {"contrast": contrast, **probe}, "evidence": evidence, "decision": decision,
            "notes": edit.rationale}


# -- provided: proposers -----------------------------------------------------------------------------------------

PROPOSER_SYSTEM = """You improve an AI agent's harness: its system prompt, its tool descriptions and its settings.
The model behind the agent cannot change. You are shown the current harness and the agent's failed runs.
Propose up to {n} small edits that would make the agent succeed on these tasks and on similar ones. Do not
mention task-specific answers or file contents. Reply with a JSON array; each item has:
  "target": "system_prompt" | "tool:<name>" | "setting:max_steps"
  "op": "append" (system_prompt: one line added) or "set" (tool description, or setting value)
  "value": the line, the new description, or an integer
  "rationale": one sentence on the failure it addresses
  "failure_signature": the signature of the failure it addresses"""


class ModelProposer:
    """Asks a model for edits; the reply is parsed with parse_edits."""

    def __init__(self, model: Any, n: int = 3):
        self.model, self.n = model, n
        self.calls = 0
        self.seen: list[str] = []          # task ids the proposer was shown

    def propose(self, config: HarnessConfig, failures: list[dict[str, Any]],
                tried: list[str]) -> list[Edit]:
        self.calls += 1
        self.seen += [f["task_id"] for f in failures]
        user = json.dumps({"harness": config.to_json(), "tools": list(TOOL_NAMES), "failures": failures,
                           "already_tried": tried}, indent=1)
        resp = self.model.complete([{"role": "system", "content": PROPOSER_SYSTEM.format(n=self.n)},
                                    {"role": "user", "content": user}], [])
        return parse_edits(resp.content or "")[:self.n]


READ_FIRST = "Read the relevant files before answering; never answer from a guess."
EVERY_MATCH = "When a search returns several matches, look at every match, not only the first."
WRITE_STRICT = ("Creates a new text file. Paths are relative to the workspace root. Never use it on a file that "
                "already exists.")
WRITE_GOOD = "Create or overwrite a text file. Paths are relative to the workspace root (for example src/a.py)."

SCRIPTED_EDITS: dict[str, list[dict[str, Any]]] = {
    "answered_without_tools": [
        {"target": "system_prompt", "op": "append", "value": "Think step by step.",
         "rationale": "More deliberate answers."},
        {"target": "system_prompt", "op": "append", "value": READ_FIRST,
         "rationale": "The agent answered without opening any file."}],
    "bad_write": [
        {"target": "tool:write_file", "op": "set", "value": WRITE_STRICT,
         "rationale": "The written file had wrong paths; say what paths mean, and keep writes to new files."},
        {"target": "tool:write_file", "op": "set", "value": WRITE_GOOD,
         "rationale": "The written file had wrong paths; say what paths are relative to."}],
    "wrong_answer": [
        {"target": "system_prompt", "op": "append", "value": EVERY_MATCH,
         "rationale": "The agent opened one search hit and answered."},
        {"target": "setting:max_steps", "op": "set", "value": 8,
         "rationale": "Opening every hit takes more steps than the budget allows."}],
    "step_budget": [
        {"target": "setting:max_steps", "op": "set", "value": 8, "rationale": "The run hit the step budget."}],
}


class ScriptedProposer:
    """The offline proposer: for each failure signature, the next edit from SCRIPTED_EDITS not yet tried.

    Its list for each signature holds a mix of edits that help, that do nothing and that do harm, in an order that
    makes the loop show each kind of decision on the simulated agent (tasks/improve_sim.py).
    """

    def __init__(self, table: dict[str, list[dict[str, Any]]] | None = None):
        self.table = table or SCRIPTED_EDITS
        self.calls = 0
        self.seen: list[str] = []

    def propose(self, config: HarnessConfig, failures: list[dict[str, Any]], tried: list[str]) -> list[Edit]:
        self.calls += 1
        self.seen += [f["task_id"] for f in failures]
        out: list[Edit] = []
        for sig in dict.fromkeys(f["signature"] for f in failures):
            for spec in self.table.get(sig, []):
                e = Edit(spec["target"], spec["op"], spec["value"], spec.get("rationale", ""), sig)
                if e.describe() not in tried and all(e.describe() != x.describe() for x in out):
                    out.append(e)
                    break
        return out


# -- your part ---------------------------------------------------------------------------------------------------

def parse_edits(text: str) -> list[Edit]:
    """Edits from a proposer's reply.

    Accept a JSON array of edit objects, or an object with an "edits" array, possibly inside a ```json fence or
    after some prose (use the first "[" or "{" that starts valid JSON). Keep an item only when it is valid:

    - target "system_prompt" with op "append" and a non-empty string value;
    - target "tool:<name>" for a name in TOOL_NAMES, op "set", non-empty string value;
    - target "setting:<name>" for a name in SETTINGS, op "set", a value that int() accepts and that lies in the
      setting's (low, high) range, stored as an int.

    rationale and failure_signature default to "". Unparseable text gives [].
    """
    # TODO(lab 14)
    raise NotImplementedError("Lab 14: implement parse_edits")


def probe(evaluator: Evaluator, config: HarnessConfig, edit: Edit, failing: list[str]) -> Probe:
    """Run the failing DEV tasks with the edit applied to `config` (one evaluator.run call). fixed: the tasks
    that now pass in every trial; adopted when at least one is fixed. runs: the agent runs this probe cost."""
    # TODO(lab 14)
    raise NotImplementedError("Lab 14: implement probe")


def compose(evaluator: Evaluator, config: HarnessConfig, probes: list[Probe], dev: list[str],
            heldout: list[str], dev_score: float, heldout_score: float
            ) -> tuple[HarnessConfig, float, float, list[tuple[Probe, str, float, float]]]:
    """Add adopted probes' edits to `config` one at a time, most tasks fixed first (ties: proposal order).

    For each edit:
      - "conflict" when it conflicts() with an edit already kept in this call (not evaluated);
      - otherwise evaluate config + edit on DEV and on HELDOUT (two evaluator.run calls), then
        "kept" when DEV success rises and HELDOUT success does not fall (the config, dev_score and
        heldout_score move on), "regressed_heldout" when DEV rises but HELDOUT falls, and
        "no_gain" when DEV does not rise.
    Return (config, dev_score, heldout_score, [(probe, decision, dev, heldout)]) where dev and heldout are the
    scores measured for that edit (the current scores for "conflict").
    """
    # TODO(lab 14)
    raise NotImplementedError("Lab 14: implement compose")


@dataclass
class ImproveResult:
    config: HarnessConfig
    lessons: list[dict[str, Any]]
    history: list[dict[str, Any]]      # one entry per evaluation: {"round", "dev", "heldout"}
    runs: int
    tokens: int
    proposer_calls: int


def improve(config: HarnessConfig, evaluator: Evaluator, proposer: Any, dev: list[str], heldout: list[str],
            rounds: int = 3) -> ImproveResult:
    """The improvement loop (module docstring).

    Start with history [{"round": 0, "dev": .., "heldout": ..}] for `config`. In each round (1..rounds):
      1. Evaluate `config` on DEV (reuse the round-0 results in round 1). Stop when nothing fails.
      2. failures = failure_report(DEV results, the DEV tasks' prompts). Ask
         proposer.propose(config, failures, tried) where tried lists edit.describe() of every edit proposed so
         far. Stop when it returns no edits. The proposer never sees HELDOUT tasks.
      3. probe() each edit on the failing DEV tasks. A probe that fixes nothing gets the lesson decision
         "rejected_probe".
      4. compose() the adopted ones; their lesson decisions are compose()'s ("kept" becomes "adopted").
      5. Append {"round": r, "dev": .., "heldout": ..} with the scores after compose.
    One lesson per proposed edit, in the order decided, via lesson_record with lesson_id "les_r{round}_{n}"
    (n counts from 1 within the round), task_id the failing DEV tasks, contrast a one-line summary, evidence
    the fixed tasks, and the probe fields fixed, still_failing and runs (plus dev and heldout for composed ones).
    """
    # TODO(lab 14)
    raise NotImplementedError("Lab 14: implement improve")
