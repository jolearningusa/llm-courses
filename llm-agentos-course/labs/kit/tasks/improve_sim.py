"""Lab 14: a simulated agent with bad habits that the right harness edit removes, to run the improvement loop offline.

A real model's behavior depends on the harness: its system prompt, its tool descriptions, its step budget. The
scripted models of the other labs ignore all of that, so they cannot test a loop that edits the harness.
HabitModel can. For each basic-tier task it replays the task's scripted solution, unless one of the task's
*habits* is active, in which case it replays that habit's bad script instead. A habit is active when its *cue*
(a regular expression) does not match the harness the model sees; a few harmful habits are the other way round
and switch on when an edit adds a phrase. The cues stand for what a real model would need to read.

    guess          b01, b04, b07   answers from a guess, without opening a file. Removed by a system prompt line
                                   that says to read or open files before answering.
    first_match    b09             opens only the first search hit. Removed by a system prompt line about every
                                   (each, all) match.
    bare_paths     b08             writes paths without the src/ prefix. Removed by a write_file or search
                                   description that says paths are relative to the workspace root.
    no_overwrite   b03, b06        harmful: refuses to change an existing file. Switched ON by a write_file
                                   description that forbids writing to existing files.

b09's scripted solution also needs five model calls, more than the starting harness's max_steps of 4, so it
takes two edits together to fix. The split: DEV tasks are the ones the improvement loop sees and probes;
HELDOUT tasks are only used to check a composed harness, never shown to the proposer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from models.scripted import ScriptedModel
from nanoos.types import ModelResponse
from tasks.basic import TASKS, call, say

DEV = ("b01_count_lines", "b02_top_error", "b04_csv_sum", "b05_json_lookup", "b08_collect_todos",
       "b09_urgent_refund")
HELDOUT = ("b03_fix_typo", "b06_config_edit", "b07_two_hop", "b10_low_stock")


@dataclass
class Habit:
    name: str
    tasks: tuple[str, ...]
    cue: str                  # regular expression searched in the harness text (see harness_text)
    where: str                # "system", "tool:<name>" or "tools" (any tool description)
    harmful: bool = False     # True: the habit is ON when the cue matches (an edit made it worse)


HABITS = [
    Habit("guess", ("b01_count_lines", "b04_csv_sum", "b07_two_hop"),
          r"(?i)\b(read|open)\b[^.]{0,60}\bbefore\b", "system"),
    Habit("first_match", ("b09_urgent_refund",), r"(?i)\b(every|each|all)\b[^.]{0,20}\b(match|result|hit)", "system"),
    Habit("bare_paths", ("b08_collect_todos",), r"(?i)relative to the workspace root", "tools:write_file,search"),
    Habit("no_overwrite", ("b03_fix_typo", "b06_config_edit"),
          r"(?i)(never|do not|don't)\b[^.]{0,40}\b(existing|already exists)", "tool:write_file", harmful=True),
]

BAD_SCRIPTS: dict[tuple[str, str], list[dict[str, Any]]] = {
    ("guess", "b01_count_lines"): [say("About 36 lines.")],
    ("guess", "b04_csv_sum"): [say("The West region totals roughly 4110.")],
    ("guess", "b07_two_hop"): [say("Lena Fischer")],
    ("first_match", "b09_urgent_refund"): [call("search", pattern="(?i)refund", path="tickets"),
                                           call("read_file", path="tickets/T-1003.json"), say("T-1003")],
    ("bare_paths", "b08_collect_todos"): [call("search", pattern="TODO", path="src"),
                                          call("write_file", path="todos.txt", content="a.py:5\nb.py:2\nb.py:9\n"),
                                          say("Wrote 3 TODOs.")],
    ("no_overwrite", "b03_fix_typo"): [call("read_file", path="notes.md"),
                                       say("write_file must not touch an existing file, so notes.md is unchanged.")],
    ("no_overwrite", "b06_config_edit"): [call("read_file", path="config.json"),
                                          say("I cannot overwrite config.json: write_file is only for new files.")],
}


def harness_text(messages: list[dict[str, Any]], tools: list[dict[str, Any]], where: str) -> str:
    """The part of the harness a habit's cue is matched against."""
    if where == "system":
        return "\n".join(m.get("content") or "" for m in messages if m.get("role") == "system")
    names = None
    if ":" in where:
        names = set(where.split(":", 1)[1].split(","))
    return "\n".join(t["function"].get("description", "") for t in tools
                     if names is None or t["function"]["name"] in names)


def active_habits(task_id: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[str]:
    out = []
    for h in HABITS:
        if task_id not in h.tasks:
            continue
        found = re.search(h.cue, harness_text(messages, tools, h.where)) is not None
        if found == h.harmful:
            out.append(h.name)
    return out


class HabitModel:
    """Replays a task's scripted solution, or a habit's bad script when the harness leaves that habit on.

    The script is chosen on the first call of a run, from the system prompt and tool schemas the harness sends,
    and then replayed turn by turn like ScriptedModel. Use one instance per run.
    """

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.task = next(t for t in TASKS if t.id == task_id)
        self.name = f"habits:{task_id}"
        self.habits: list[str] | None = None
        self._model: ScriptedModel | None = None

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        if self._model is None:
            self.habits = active_habits(self.task_id, messages, tools)
            script = BAD_SCRIPTS[(self.habits[0], self.task_id)] if self.habits else self.task.demo
            self._model = ScriptedModel(json.loads(json.dumps(script)), name=self.name)
        return self._model.complete(messages, tools)
