"""A naive judge "model" for Lab 13: PASS when the reference appears in the candidate answer.

It reads the prompt built by nanoos.graders.judge_messages and answers like an LLM judge would, so the judge
pipeline and its calibration run offline. Its rule is the one a quick string check uses: every comma-separated
part of the reference must appear as a whole word in the answer, ignoring case and thousands separators. That
rule is wrong in instructive ways (a hedged or contradicted answer that mentions the reference passes; a correct
answer written in other words fails), which is what `obs_eval.py --calibrate` measures against the 30 labeled
cases in tasks/judge_cases.json.
"""

from __future__ import annotations

import re
from typing import Any

from nanoos.types import ModelResponse, Usage, estimate_tokens


def _section(text: str, title: str) -> str:
    m = re.search(rf"{title}:\n(.*?)(?:\n\n[A-Z][^\n]*:\n|\Z)", text, re.S)
    return m.group(1).strip() if m else ""


def _has(answer: str, value: str) -> bool:
    norm = re.sub(r"(?<=\d),(?=\d{3}\b)", "", answer).lower()
    return re.search(rf"(?<![\w.-]){re.escape(value.lower())}(?![\w-])", norm) is not None


class SubstringJudge:
    name = "substring-judge"

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        prompt = messages[-1]["content"] or ""
        reference, answer = _section(prompt, "Reference answer"), _section(prompt, "Candidate answer")
        parts = [p.strip() for p in reference.split(",") if p.strip()]
        found = bool(parts) and all(_has(answer, p) for p in parts)
        text = (f"The reference {'appears' if found else 'does not appear'} in the candidate answer.\n"
                f"VERDICT: {'PASS' if found else 'FAIL'}")
        usage = Usage(sum(estimate_tokens(m.get("content") or "") for m in messages), 0, estimate_tokens(text))
        return ModelResponse(text, [], usage)
