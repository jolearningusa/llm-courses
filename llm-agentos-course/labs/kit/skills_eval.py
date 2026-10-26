#!/usr/bin/env python3
"""Lab 8: does a skill help, and what does the index cost? (skills deck exercise), using nanoos/skills.py.

    python skills_eval.py --skills none                 # offline: a scripted model that does not know the format
    python skills_eval.py --skills all                  # ... and one that opens report-fill (50 skills installed)
    python skills_eval.py --skills all --model api --loop nanoos --trials 10 --out runs/lab8-all

Five tasks (tasks/skills/): four incident reports in a house format that only the report-fill skill describes,
and one control question that needs no skill. --skills none installs no skills, one installs report-fill, all
installs it with 49 distractors. --root adds another skill root (for example your own skills folder) after the
workspace's .agents/skills. --shell adds the Lab 7 run_shell tool, so the skill's script can run.
Outputs under --out: metrics.csv (one row per task and trial) and summary.json: pass@1, pass^k, activation rate,
false activations on the control task, and index tokens.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.scripted import ModelError, ScriptedModel  # noqa: E402
from nanoos.context import ContextManagedModel  # noqa: E402
from nanoos.skills import DEFAULT_BUDGET_CHARS, SkillIndexEngine, SkillTools, discover, render_index  # noqa: E402
from nanoos.tools_basic import WorkspaceTools  # noqa: E402
from tasks.skills import TASKS, install  # noqa: E402

COLUMNS = ["task", "skills", "trial", "success", "activated", "references_read", "other_skills_opened", "steps",
           "input_tokens", "output_tokens", "index_tokens"]
SKILL = "report-fill"


class ReadLog:
    """Passes calls through and records read_file paths (reading SKILL.md directly also activates a skill)."""

    def __init__(self, tools: Any):
        self.tools, self.paths = tools, []

    def schemas(self) -> list[dict[str, Any]]:
        return self.tools.schemas()

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        if name == "read_file" and isinstance(arguments, dict):
            self.paths.append(str(arguments.get("path", "")))
        return self.tools.call(name, arguments)


class ShellTools:
    """Adds Lab 7's sandboxed run_shell (writable workspace, so a script can write its output)."""

    def __init__(self, tools: Any, workspace: Path):
        from nanoos.sandbox import best_available, shell_tool
        from nanoos.tools import ToolRegistry
        self.tools, self.reg = tools, ToolRegistry(workspace)
        self.reg.register(shell_tool(best_available(workspace, writable=True)))

    def schemas(self) -> list[dict[str, Any]]:
        return self.tools.schemas() + self.reg.schemas()

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        return self.reg.call(name, arguments) if name == "run_shell" else self.tools.call(name, arguments)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skills", choices=["none", "one", "all"], default="all")
    ap.add_argument("--model", choices=["demo", "api"], default="demo")
    ap.add_argument("--loop", default="baseline", help="baseline, nanoos (your Lab 1 loop) or a module with run()")
    ap.add_argument("--root", action="append", default=[], help="another skill root, after the workspace's")
    ap.add_argument("--budget-chars", type=int, default=DEFAULT_BUDGET_CHARS, help="index budget in characters")
    ap.add_argument("--shell", action="store_true", help="add Lab 7's run_shell tool")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--out", default="runs/lab8")
    args = ap.parse_args(argv)

    loop = importlib.import_module({"baseline": "nanoos.baseline", "nanoos": "nanoos.loop"}.get(args.loop, args.loop))
    api_model = None
    if args.model == "api":
        from models.openai_compat import OpenAICompatModel
        api_model = OpenAICompatModel()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rows, warnings, index_tokens = [], [], 0
    for task in TASKS:
        for trial in range(args.trials):
            with tempfile.TemporaryDirectory() as d:
                ws = Path(d)
                root = install(ws, args.skills)
                skills, warnings = discover([root, *args.root])
                engine = SkillIndexEngine(render_index(skills, args.budget_chars))
                index_tokens = engine.index_tokens
                reads = ReadLog(WorkspaceTools(ws))
                tools: Any = SkillTools(reads, skills)
                if args.shell:
                    tools = ShellTools(tools, ws)
                st = tools.tools if args.shell else tools
                scripted = task.demo_without if args.skills == "none" else task.demo_with
                model = ContextManagedModel(api_model or ScriptedModel(scripted), engine)
                try:
                    r = loop.run(model, task.prompt, tools, None, max_steps=args.max_steps)
                    answer, steps, usage = r.answer, r.steps, r.usage
                except ModelError as e:
                    print(f"{task.id}: model error: {e}", file=sys.stderr)
                    answer, steps, usage = "", 0, None
                activated = SKILL in st.opened or any(p.rstrip("/").endswith(f"{SKILL}/SKILL.md") for p in reads.paths)
                rows.append({"task": task.id, "skills": args.skills, "trial": trial,
                             "success": int(task.check(ws, answer)), "activated": int(activated),
                             "references_read": sum(1 for n, _ in st.files_read if n == SKILL),
                             "other_skills_opened": sum(1 for n in st.opened if n != SKILL), "steps": steps,
                             "input_tokens": usage.input_tokens if usage else 0,
                             "output_tokens": usage.output_tokens if usage else 0, "index_tokens": index_tokens})
            r_ = rows[-1]
            opened = "opened" if r_["activated"] else "-"
            print(f"{task.id:12s} {'PASS' if r_['success'] else 'fail'}  skill={opened:6s} in={r_['input_tokens']}")

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    needs = {t.id for t in TASKS if t.needs_skill}
    skill_rows = [r for r in rows if r["task"] in needs]
    control = [r for r in rows if r["task"] not in needs]
    by_task: dict[str, list[int]] = {}
    for r in skill_rows:
        by_task.setdefault(r["task"], []).append(r["success"])
    summary = {"skills": args.skills, "model": args.model if not api_model else api_model.name,
               "trials": args.trials, "skills_installed": len(skills), "warnings": warnings,
               "index_tokens": index_tokens,
               "pass_at_1": sum(r["success"] for r in skill_rows) / len(skill_rows),
               "pass_hat_k": sum(all(v) for v in by_task.values()) / len(by_task),
               "activation_rate": sum(r["activated"] for r in skill_rows) / len(skill_rows),
               "control_success": sum(r["success"] for r in control) / max(1, len(control)),
               "false_activations": sum(r["activated"] + r["other_skills_opened"] for r in control),
               "input_tokens_per_task": sum(r["input_tokens"] for r in rows) / len(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nskills={args.skills} ({len(skills)} installed, index {index_tokens} tokens): pass@1 "
          f"{summary['pass_at_1']:.0%}, pass^{args.trials} {summary['pass_hat_k']:.0%}, activation "
          f"{summary['activation_rate']:.0%}, false activations {summary['false_activations']}  -> {out}")
    for w_ in warnings:
        print("warning:", w_)
    return 0


if __name__ == "__main__":
    sys.exit(main())
