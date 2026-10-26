"""Lab 8 autograder tests: nanoos/skills.py (validation, discovery, the index, progressive disclosure)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from models.scripted import ScriptedModel
from nanoos import baseline
from nanoos.context import ContextManagedModel
from nanoos.skills import (INDEX_HEADER, Skill, SkillIndexEngine, SkillTools, discover, render_index, validate)
from nanoos.tools_basic import WorkspaceTools
from skills_eval import main as skills_eval
from tasks.skills import TASKS, install


def write_skill(root: Path, rel: str, name: str | None = None, description: str = "Does a thing. Use for things.",
                body: str = "# Steps\n\n1. Do it.\n", raw: str | None = None) -> Path:
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    text = raw if raw is not None else (f"---\nname: {name or Path(rel).name}\ndescription: {description}\n---\n\n"
                                        + body)
    (d / "SKILL.md").write_text(text, encoding="utf-8")
    return d / "SKILL.md"


class TestValidate(unittest.TestCase):
    def test_valid(self):
        meta = {"name": "pdf-tools", "description": "Work with PDFs.", "license": "MIT", "metadata": {"a": "1"},
                "allowed-tools": "Read", "platforms": "linux"}
        self.assertEqual(validate(meta, "pdf-tools"), [])
        self.assertEqual(validate({"name": "a1", "description": "x"}, "a1"), [])

    def test_name_rules(self):
        for bad in ("PDF", "-pdf", "pdf-", "pdf--tools", "pdf_tools", "a" * 65, ""):
            with self.subTest(name=bad):
                self.assertTrue(validate({"name": bad, "description": "x"}, bad), bad)
        self.assertEqual(validate({"name": "a" * 64, "description": "x"}, "a" * 64), [])
        problems = validate({"name": "pdf", "description": "x"}, "pdf-tools")
        self.assertEqual(len(problems), 1)
        self.assertIn("pdf-tools", problems[0], "say which directory it should match")
        self.assertTrue(validate({"description": "x"}, "pdf"))

    def test_description_and_compatibility(self):
        self.assertTrue(validate({"name": "a"}, "a"))
        self.assertTrue(validate({"name": "a", "description": "  "}, "a"))
        self.assertTrue(validate({"name": "a", "description": "x" * 1025}, "a"))
        self.assertEqual(validate({"name": "a", "description": "x" * 1024}, "a"), [])
        self.assertTrue(validate({"name": "a", "description": "x", "compatibility": "y" * 501}, "a"))
        self.assertEqual(validate({"name": "a", "description": "x", "compatibility": "y" * 500}, "a"), [])


class TestDiscover(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.base = Path(self.dir.name)
        self.project, self.user = self.base / "project", self.base / "user"

    def tearDown(self):
        self.dir.cleanup()

    def test_precedence_nesting_and_order(self):
        p = write_skill(self.project, "zeta", description="Project zeta.")
        write_skill(self.project, "ops/alpha")
        write_skill(self.project, "a/b/c/d/e/deep")
        write_skill(self.user, "zeta", description="User zeta.")
        write_skill(self.user, "beta")
        skills, warnings = discover([self.project, self.user, self.base / "missing"])
        self.assertEqual([s.name for s in skills], ["alpha", "beta", "deep", "zeta"])
        zeta = skills[-1]
        self.assertEqual((zeta.description, zeta.path, zeta.root), ("Project zeta.", p, self.project))
        self.assertEqual(len(warnings), 1)
        self.assertIn(str(self.user / "zeta" / "SKILL.md"), warnings[0], "the shadowed file is named")
        self.assertEqual([s.name for s in discover([self.project], max_depth=2)[0]], ["alpha", "zeta"])
        self.assertEqual(discover([self.project / "zeta"])[0], [], "a SKILL.md must be in a folder below the root")

    def test_bad_skills_are_skipped_with_a_warning(self):
        write_skill(self.project, "good")
        bad = {
            "nofront": write_skill(self.project, "nofront", raw="# no frontmatter\n"),
            "unclosed": write_skill(self.project, "unclosed", raw="---\nname: unclosed\ndescription: x\n"),
            "badline": write_skill(self.project, "badline", raw="---\nname: badline\njust words\n---\n"),
            "mismatch": write_skill(self.project, "mismatch", name="other"),
            "nodesc": write_skill(self.project, "nodesc", raw="---\nname: nodesc\n---\nbody\n"),
        }
        skills, warnings = discover([self.project])
        self.assertEqual([s.name for s in skills], ["good"])
        self.assertEqual(len(warnings), len(bad))
        for path in bad.values():
            self.assertTrue(any(str(path) in w for w in warnings), path)

    def test_symlink_out_of_the_root_is_skipped(self):
        outside = write_skill(self.base / "elsewhere", "sneaky")
        (self.project / "sneaky").mkdir(parents=True)
        os.symlink(outside, self.project / "sneaky" / "SKILL.md")
        skills, warnings = discover([self.project])
        self.assertEqual(skills, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("sneaky", warnings[0])

    def test_frontmatter_values(self):
        write_skill(self.project, "folded", raw="---\nname: folded\ndescription: >\n  Two lines\n  joined.\n"
                                                "metadata:\n  version: \"1.0\"\n---\n\nBody.\n")
        (skill,), _ = discover([self.project])
        self.assertEqual(skill.description, "Two lines joined.")
        self.assertEqual(skill.meta["metadata"], {"version": "1.0"})


def make_skills(n: int, desc_len: int = 60) -> list[Skill]:
    return [Skill(f"skill-{i:02d}", ("d" * desc_len), Path(f"/x/skill-{i:02d}/SKILL.md"), Path("/x"))
            for i in range(n)]


class TestIndex(unittest.TestCase):
    def test_full_form(self):
        s = [Skill("a", "Does a.\n  Use  for a.", Path("/r/a/SKILL.md"), Path("/r")),
             Skill("b", "Does b.", Path("/r/b/SKILL.md"), Path("/r"))]
        self.assertEqual(render_index(s), INDEX_HEADER + "- a: Does a. Use for a.\n- b: Does b.")
        self.assertEqual(render_index([]), "")

    def test_over_budget_drops_descriptions_then_names(self):
        skills = make_skills(30)
        full = render_index(skills, 100_000)
        self.assertIn("d" * 60, full)
        names_only = render_index(skills, len(full) - 1)
        self.assertEqual(names_only, INDEX_HEADER + "\n".join(f"- skill-{i:02d}" for i in range(30)))
        for budget in (len(names_only) - 1, 300, len(INDEX_HEADER) + 40):
            with self.subTest(budget=budget):
                text = render_index(skills, budget)
                self.assertLessEqual(len(text), budget)
                kept = [x for x in text.splitlines() if x.startswith("- skill-")]
                self.assertEqual(kept, [f"- skill-{i:02d}" for i in range(len(kept))])
                self.assertTrue(text.endswith(f"- ({30 - len(kept)} more; call skills_list)"), text[-60:])
        self.assertEqual(render_index(skills, 300), render_index(skills, 300), "deterministic")

    def test_fifty_installed_skills_fit_the_default_budget(self):
        with tempfile.TemporaryDirectory() as d:
            skills, warnings = discover([install(Path(d), "all")])
        self.assertEqual((len(skills), warnings), (50, []))
        index = render_index(skills)
        self.assertLessEqual(len(index), 8000)
        self.assertIn("- report-fill: Write an incident report", index)
        self.assertLess(SkillIndexEngine(index).index_tokens, 2000)


class TestSkillView(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.ws = Path(self.dir.name)
        root = self.ws / ".agents" / "skills"
        write_skill(root, "notes", body="# Notes\n\nRead references/style.md first.\n")
        (root / "notes" / "references").mkdir()
        (root / "notes" / "references" / "style.md").write_text("Use short lines.\n")
        (root / "notes" / "scripts").mkdir()
        (root / "notes" / "scripts" / "go.py").write_text("print(1)\n")
        write_skill(root, "bare")
        (self.ws / "secret.txt").write_text("not for skills\n")
        os.symlink(self.ws / "secret.txt", root / "notes" / "references" / "link.md")
        skills, _ = discover([root])
        self.tools = SkillTools(WorkspaceTools(self.ws), skills)

    def tearDown(self):
        self.dir.cleanup()

    def test_level_two_returns_the_body_and_lists_files(self):
        out, err = self.tools.call("skill_view", {"name": "notes"})
        self.assertFalse(err)
        self.assertTrue(out.startswith("# Notes"), "no frontmatter")
        self.assertNotIn("description:", out)
        self.assertTrue(out.endswith("Files in this skill: references/link.md, references/style.md, scripts/go.py"))
        self.assertEqual(self.tools.call("skill_view", {"name": "bare"})[0], "# Steps\n\n1. Do it.")
        self.assertEqual(self.tools.opened, ["notes", "bare"])

    def test_level_three_reads_files_inside_the_skill_only(self):
        self.assertEqual(self.tools.call("skill_view", {"name": "notes", "path": "references/style.md"}),
                         ("Use short lines.\n", False))
        self.assertEqual(self.tools.files_read, [("notes", "references/style.md")])
        for path in ("../bare/SKILL.md", "../../../secret.txt", str(self.ws / "secret.txt"), "references/link.md"):
            with self.subTest(path=path):
                out, err = self.tools.call("skill_view", {"name": "notes", "path": path})
                self.assertTrue(err)
                self.assertIn("outside", out)
        out, err = self.tools.call("skill_view", {"name": "notes", "path": "references/missing.md"})
        self.assertTrue(err)
        self.assertIn("references/style.md", out, "list the files that exist")
        self.assertEqual(self.tools.opened, [], "level-3 reads do not count as opening the skill")
        self.assertEqual(len(self.tools.files_read), 1)

    def test_unknown_skill_and_other_tools(self):
        out, err = self.tools.call("skill_view", {"name": "nope"})
        self.assertTrue(err)
        self.assertTrue(out.startswith("ERROR: "))
        self.assertIn("skills_list", out)
        self.assertEqual(self.tools.call("read_file", {"path": "secret.txt"}), ("not for skills\n", False))
        out, err = self.tools.call("skills_list", {})
        self.assertFalse(err)
        self.assertIn("- notes: Does a thing.", out)


class TestProgressiveDisclosure(unittest.TestCase):
    def test_body_enters_the_prompt_only_after_skill_view(self):
        task = TASKS[0]
        with tempfile.TemporaryDirectory() as d:
            ws = Path(d)
            skills, _ = discover([install(ws, "all")])
            scripted = ScriptedModel(task.demo_with)
            model = ContextManagedModel(scripted, SkillIndexEngine(render_index(skills)))
            r = baseline.run(model, task.prompt, SkillTools(WorkspaceTools(ws), skills), max_steps=12)
            self.assertTrue(task.check(ws, r.answer))
        body_marker, ref_marker = "## Pitfalls", "Use the first row that matches"
        prompts = ["\n".join(str(m.get("content")) for m in c["messages"]) for c in scripted.calls]
        self.assertIn("report-fill: Write an incident report", prompts[0], "the index is in the system prompt")
        self.assertNotIn(body_marker, prompts[0])
        self.assertIn(body_marker, prompts[1])
        self.assertNotIn(ref_marker, prompts[1])
        self.assertIn(ref_marker, prompts[2])


class TestSkillsEval(unittest.TestCase):
    def test_the_skill_helps_and_the_index_is_small(self):
        s = {}
        for v in ("none", "all"):
            with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
                skills_eval(["--skills", v, "--out", str(Path(d) / "r")])
                s[v] = json.loads((Path(d) / "r" / "summary.json").read_text())
        self.assertEqual((s["none"]["pass_at_1"], s["none"]["activation_rate"]), (0.0, 0.0))
        self.assertEqual((s["all"]["pass_at_1"], s["all"]["activation_rate"]), (1.0, 1.0))
        self.assertEqual((s["all"]["skills_installed"], s["all"]["false_activations"]), (50, 0))
        self.assertEqual(s["all"]["control_success"], 1.0)
        self.assertLess(s["all"]["index_tokens"], 2000)


if __name__ == "__main__":
    unittest.main()
