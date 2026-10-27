"""Lab 13: graders for agent runs: exact match, outcome checks, trajectories, convergence and an LLM judge.

A grader scores one aspect of one trial (the deck's vocabulary). The kit's graders:

    exact_match(answer, expected)          the final answer equals a reference, after normalize()
    outcome(task, workspace, answer)       the task's own checker: the environment's final state (provided)
    trajectory(actual, expected, match)    the tool-call sequence against an expected one (ADK's
                                           tool_trajectory_avg_score, with EXACT, IN_ORDER and ANY_ORDER)
    convergence(steps)                     how close each run's path is to the shortest one seen (Arize)
    LLMJudge(model).grade(...)             a model grades the answer against a reference with a yes/no rubric
    agreement(judge, human)                how well a judge agrees with human labels: accuracy, Cohen's kappa,
                                           true positive and true negative rates

The judge is calibrated on tasks/judge_cases.json, 30 answers labeled pass or fail by a person.

Your job (search for TODO): normalize, exact_match, trajectory, convergence, parse_verdict, LLMJudge.grade and
agreement.
`make grade LAB=13` checks your work.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CASES = Path(__file__).resolve().parent.parent / "tasks" / "judge_cases.json"


@dataclass
class Grade:
    name: str
    score: float                 # 0.0 to 1.0
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)


# -- provided ---------------------------------------------------------------------------------------------------

def outcome(task: Any, workspace: Path, answer: str) -> Grade:
    """The task's own checker, which looks at the final answer and the files the run left behind."""
    ok = bool(task.check(Path(workspace), answer))
    return Grade("outcome", float(ok), ok)


def tool_calls(messages: list[dict[str, Any]]) -> list[str]:
    """Tool names called in a run, in order, from its final message list."""
    out = []
    for m in messages:
        for c in m.get("tool_calls") or []:
            out.append(c["function"]["name"])
    return out


def demo_trajectory(demo: list[dict[str, Any]]) -> list[str]:
    """The tool names a task's scripted solution calls: a reference trajectory for trajectory()."""
    return [c["name"] for turn in demo for c in turn.get("tool_calls", [])]


def demo_answer(demo: list[dict[str, Any]]) -> str:
    """The final text of a task's scripted solution: a reference answer for exact_match() and the judge."""
    texts = [t["content"] for t in demo if t.get("content") and not t.get("tool_calls")]
    return texts[-1] if texts else ""


def load_cases(path: Path = CASES) -> list[dict[str, Any]]:
    """The judge calibration set: [{"id", "question", "reference", "answer", "label": "pass"|"fail", "why"}]."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


JUDGE_SYSTEM = (
    "You grade the final answer of an AI agent. Compare the candidate answer with the reference answer and "
    "apply the rubric. Reply with one short sentence of reasoning, then a last line that is exactly "
    "VERDICT: PASS or VERDICT: FAIL."
)

DEFAULT_RUBRIC = [
    "The candidate gives the same final value or result as the reference (formatting, units written out, and "
    "extra explanation do not matter).",
    "The candidate commits to one answer; listing several options, or saying it is unsure, fails.",
    "The candidate does not contradict the reference anywhere, including in caveats.",
]


def judge_messages(question: str, reference: str, answer: str, rubric: list[str] = DEFAULT_RUBRIC) -> list[dict]:
    """The judge prompt: a system message and one user message with the case and a yes/no rubric."""
    checks = "\n".join(f"{i}. {r}" for i, r in enumerate(rubric, 1))
    user = (f"Task given to the agent:\n{question}\n\nReference answer:\n{reference}\n\n"
            f"Candidate answer:\n{answer}\n\nRubric (all must hold for PASS):\n{checks}")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


# -- your part --------------------------------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase, drop thousands separators in numbers, turn punctuation other than . - / : @ into spaces, strip
    trailing dots, and collapse whitespace."""
    # TODO(lab 13)
    raise NotImplementedError("Lab 13: implement normalize")


def exact_match(answer: str, expected: str, normalized: bool = True) -> Grade:
    """1.0 when the answer equals the expected text: compared after normalize() when `normalized`, else
    character for character. detail: {"answer": <compared form>, "expected": <compared form>}."""
    # TODO(lab 13)
    raise NotImplementedError("Lab 13: implement exact_match")


def trajectory(actual: list[str], expected: list[str], match: str = "in_order") -> Grade:
    """Compare a run's tool-call sequence with an expected one.

    match:
        exact      the same calls in the same order, nothing else
        in_order   every expected call appears in that order; extra calls in between are allowed
        any_order  every expected call appears (with multiplicity), in any order; extra calls allowed

    score is 1.0 or 0.0. detail: {"missing": [expected calls not matched, in expected order],
    "extra": [actual calls not used for a match, in actual order]}. For exact, a mismatch reports the
    in_order missing and extra lists.
    """
    # TODO(lab 13)
    raise NotImplementedError("Lab 13: implement trajectory")


def convergence(steps: list[int], optimal: int | None = None) -> float:
    """Arize's convergence score for one task: the mean of optimal / steps over its runs, where optimal is
    the shortest run (or the given optimal length). 1.0 means every run took the shortest path. Runs with
    0 steps are ignored; no runs gives 0.0."""
    # TODO(lab 13)
    raise NotImplementedError("Lab 13: implement convergence")


def parse_verdict(text: str | None) -> bool | None:
    """The judge's verdict from its reply: True for pass, False for fail, None when there is none.

    Accept "VERDICT: PASS" / "VERDICT: FAIL" (any case, optional spaces, optional ** around the word) anywhere
    in the reply, and also a JSON object with a "verdict" key ("pass"/"fail"). When the reply has several
    verdict lines, the last one counts.
    """
    # TODO(lab 13)
    raise NotImplementedError("Lab 13: implement parse_verdict")


class LLMJudge:
    """Grades an answer against a reference with a model and a yes/no rubric."""

    def __init__(self, model: Any, rubric: list[str] | None = None, samples: int = 1):
        self.model = model
        self.rubric = list(rubric or DEFAULT_RUBRIC)
        self.samples = samples
        self.calls = 0

    def grade(self, question: str, reference: str, answer: str) -> Grade:
        """Ask the model `samples` times (with judge_messages and no tools) and take the majority verdict;
        a tie fails. Replies with no verdict count as fail votes. detail: {"votes": [True/False/None, ...],
        "unparsed": <number of None votes>, "reason": <the last reply's text>}."""
        # TODO(lab 13)
        raise NotImplementedError("Lab 13: implement LLMJudge.grade")


def agreement(judge: list[bool], human: list[bool]) -> dict[str, Any]:
    """How well judge verdicts agree with human labels (True = pass), case by case.

    Return {"n", "accuracy", "kappa", "tpr", "tnr", "tp", "fp", "tn", "fn"}: tpr is the share of human passes
    the judge passed, tnr the share of human fails the judge failed (None when there are none), and kappa is
    Cohen's kappa, (p_o - p_e) / (1 - p_e), with kappa = 1.0 when p_e == 1. Raise ValueError when the lists
    differ in length or are empty.
    """
    # TODO(lab 13)
    raise NotImplementedError("Lab 13: implement agreement")
