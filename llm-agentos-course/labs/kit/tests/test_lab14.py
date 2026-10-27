"""Lab 14 autograder tests: nanoos/improve.py (parse_edits, probe, compose, improve).

The agent is HabitModel (tasks/improve_sim.py), a simulated agent whose bad habits the right edits remove, so the
tests are offline and deterministic. They use the provided baseline loop, not your Lab 1 loop.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nanoos.improve import (READ_FIRST, SCHEMA, WRITE_GOOD, WRITE_STRICT, Edit, Evaluator, HarnessConfig, Probe,
                            ScriptedProposer, apply_edit, compose, improve, parse_edits, probe)
from tasks.basic import TASKS
from tasks.improve_sim import DEV, HELDOUT, HabitModel

EVERY = "When a search returns several matches, look at every match, not only the first."


def evaluator(trials: int = 1) -> Evaluator:
    return Evaluator(TASKS, HabitModel, Path(tempfile.mkdtemp()), trials=trials)


def sp(text: str) -> Edit:
    return Edit("system_prompt", "append", text)


class Recording:
    """A proposer that returns fixed edits and records what it was given."""

    def __init__(self, batches):
        self.batches, self.calls, self.tried, self.seen = list(batches), 0, [], []

    def propose(self, config, failures, tried):
        self.calls += 1
        self.tried.append(list(tried))
        self.seen += [f["task_id"] for f in failures]
        return self.batches.pop(0) if self.batches else []


class TestParseEdits(unittest.TestCase):
    def test_array_in_prose_and_fence(self):
        text = """Here are my edits:
```json
[{"target": "system_prompt", "op": "append", "value": "Read files first.", "rationale": "guessing",
  "failure_signature": "answered_without_tools"},
 {"target": "setting:max_steps", "op": "set", "value": "8"}]
```"""
        edits = parse_edits(text)
        self.assertEqual([(e.target, e.op, e.value) for e in edits],
                         [("system_prompt", "append", "Read files first."), ("setting:max_steps", "set", 8)])
        self.assertEqual((edits[0].rationale, edits[0].failure_signature), ("guessing", "answered_without_tools"))
        self.assertEqual((edits[1].rationale, edits[1].failure_signature), ("", ""))

    def test_object_form_and_invalid_items_dropped(self):
        text = """{"edits": [
          {"target": "tool:write_file", "op": "set", "value": "Paths are relative to the workspace root."},
          {"target": "tool:delete_file", "op": "set", "value": "x"},
          {"target": "tool:read_file", "op": "append", "value": "x"},
          {"target": "system_prompt", "op": "set", "value": "replace everything"},
          {"target": "system_prompt", "op": "append", "value": "   "},
          {"target": "setting:max_steps", "op": "set", "value": 99},
          {"target": "setting:max_steps", "op": "set", "value": "many"},
          {"target": "setting:temperature", "op": "set", "value": 1},
          {"target": "memory", "op": "set", "value": "x"},
          "not an object"]}"""
        edits = parse_edits(text)
        self.assertEqual([e.target for e in edits], ["tool:write_file"])

    def test_garbage(self):
        self.assertEqual(parse_edits("I would change the prompt."), [])
        self.assertEqual(parse_edits('{"edits": "none"}'), [])
        self.assertEqual(parse_edits("[broken json"), [])


class TestProbe(unittest.TestCase):
    def test_a_helpful_edit_fixes_some_failing_tasks(self):
        ev, cfg = evaluator(), HarnessConfig()
        p = probe(ev, cfg, sp(READ_FIRST), ["b01_count_lines", "b04_csv_sum", "b09_urgent_refund"])
        self.assertIsInstance(p, Probe)
        self.assertEqual(p.fixed, ["b01_count_lines", "b04_csv_sum"])
        self.assertEqual(p.still_failing, ["b09_urgent_refund"])
        self.assertTrue(p.adopted)
        self.assertEqual((p.runs, ev.runs), (3, 3))
        self.assertEqual(cfg.system_prompt, HarnessConfig().system_prompt, "probing must not change the config")

    def test_a_useless_edit_is_not_adopted_and_trials_count(self):
        ev = evaluator(trials=2)
        p = probe(ev, HarnessConfig(), sp("Think step by step."), ["b01_count_lines", "b04_csv_sum"])
        self.assertFalse(p.adopted)
        self.assertEqual(p.fixed, [])
        self.assertEqual(p.runs, 4)

    def test_edits_that_only_work_together(self):
        ev, cfg, b09 = evaluator(), HarnessConfig(), ["b09_urgent_refund"]
        self.assertFalse(probe(ev, cfg, sp(EVERY), b09).adopted, "alone, the step budget still stops it")
        self.assertFalse(probe(ev, cfg, Edit("setting:max_steps", "set", 8), b09).adopted)
        both = apply_edit(cfg, Edit("setting:max_steps", "set", 8))
        self.assertTrue(probe(ev, both, sp(EVERY), b09).adopted)


class TestCompose(unittest.TestCase):
    def setUp(self):
        self.ev = evaluator()
        self.cfg = HarnessConfig()
        self.dev, self.held = list(DEV), list(HELDOUT)

    def p(self, edit, fixed):
        return Probe(edit, ["b01_count_lines", "b04_csv_sum", "b08_collect_todos"], fixed, [], 1, bool(fixed))

    def test_best_first_and_heldout_regression(self):
        strict = self.p(Edit("tool:write_file", "set", WRITE_STRICT), ["b08_collect_todos"])
        read = self.p(sp(READ_FIRST), ["b01_count_lines", "b04_csv_sum"])
        cfg, dev, held, log = compose(self.ev, self.cfg, [strict, read], self.dev, self.held, 2 / 6, 3 / 4)
        self.assertEqual([(x[0].edit.target, x[1]) for x in log],
                         [("system_prompt", "kept"), ("tool:write_file", "regressed_heldout")])
        self.assertAlmostEqual(dev, 4 / 6)
        self.assertAlmostEqual(held, 1.0)
        self.assertAlmostEqual(log[1][2], 5 / 6)
        self.assertAlmostEqual(log[1][3], 2 / 4)
        self.assertIn(READ_FIRST, cfg.system_prompt)
        self.assertNotIn("write_file", cfg.tool_descriptions)
        self.assertEqual(self.ev.runs, 2 * (6 + 4))

    def test_conflicts_no_gain_and_unadopted(self):
        good = self.p(Edit("tool:write_file", "set", WRITE_GOOD), ["b08_collect_todos"])
        strict = self.p(Edit("tool:write_file", "set", WRITE_STRICT), ["b08_collect_todos"])
        vague = self.p(sp("Think step by step."), ["b01_count_lines"])  # its probe claims a fix; composing shows none
        skipped = self.p(sp(READ_FIRST), [])
        cfg, dev, held, log = compose(self.ev, self.cfg, [good, strict, vague, skipped], self.dev, self.held,
                                      2 / 6, 3 / 4)
        self.assertEqual([x[1] for x in log], ["kept", "conflict", "no_gain"])
        self.assertEqual(cfg.tool_descriptions["write_file"], WRITE_GOOD)
        self.assertEqual((log[1][2], log[1][3]), (dev, held), "a conflict is logged with the current scores")
        self.assertEqual(self.ev.runs, 2 * (6 + 4), "a conflict is not evaluated")


class TestImprove(unittest.TestCase):
    def test_offline_loop(self):
        ev, proposer = evaluator(), ScriptedProposer()
        res = improve(HarnessConfig(), ev, proposer, list(DEV), list(HELDOUT), rounds=3)
        self.assertEqual([x["decision"] for x in res.lessons],
                         ["rejected_probe", "rejected_probe", "regressed_heldout", "rejected_probe", "adopted",
                          "adopted"])
        self.assertEqual([x["lesson_id"] for x in res.lessons],
                         ["les_r1_1", "les_r1_2", "les_r1_3", "les_r2_1", "les_r2_2", "les_r2_3"])
        self.assertEqual([(h["round"], round(h["dev"], 3), h["heldout"]) for h in res.history],
                         [(0, 0.333, 0.75), (1, 0.333, 0.75), (2, 0.833, 1.0)])
        self.assertIn(READ_FIRST, res.config.system_prompt)
        self.assertEqual(res.config.tool_descriptions, {"write_file": WRITE_GOOD})
        self.assertEqual(res.config.settings["max_steps"], 4)
        self.assertEqual(proposer.calls, 3)
        self.assertEqual(set(proposer.seen) & set(HELDOUT), set(), "the proposer never sees HELDOUT tasks")
        self.assertEqual((res.runs, res.proposer_calls), (ev.runs, 3))
        for x in res.lessons:
            self.assertEqual(x["schema"], SCHEMA)
            for key in ("task_id", "target_component", "failure_signature", "proposed_edit", "probe", "evidence",
                        "decision", "notes"):
                self.assertIn(key, x)
            for key in ("contrast", "fixed", "still_failing", "runs"):
                self.assertIn(key, x["probe"])
        adopted = [x for x in res.lessons if x["decision"] == "adopted"]
        self.assertEqual(adopted[0]["evidence"], ["b01_count_lines", "b04_csv_sum"])
        self.assertIn("dev", adopted[0]["probe"])

    def test_tried_edits_are_passed_back(self):
        rec = Recording([[sp("Think step by step.")], [sp("Be careful.")]])
        res = improve(HarnessConfig(), evaluator(), rec, list(DEV), list(HELDOUT), rounds=2)
        self.assertEqual(rec.tried, [[], ['system_prompt append "Think step by step."']])
        self.assertEqual(len(res.history), 3)
        self.assertEqual(res.config.system_prompt, HarnessConfig().system_prompt)

    def test_stops_when_nothing_fails_or_nothing_is_proposed(self):
        good = HarnessConfig(system_prompt=HarnessConfig().system_prompt + "\n" + READ_FIRST + "\n" + EVERY,
                             tool_descriptions={"write_file": WRITE_GOOD}, settings={"max_steps": 8})
        rec = Recording([[sp("unused")]])
        res = improve(good, evaluator(), rec, list(DEV), list(HELDOUT), rounds=3)
        self.assertEqual(rec.calls, 0)
        self.assertEqual(res.history, [{"round": 0, "dev": 1.0, "heldout": 1.0}])
        rec = Recording([])
        res = improve(HarnessConfig(), evaluator(), rec, list(DEV), list(HELDOUT), rounds=3)
        self.assertEqual(rec.calls, 1)
        self.assertEqual((len(res.history), res.lessons), (1, []))


if __name__ == "__main__":
    unittest.main()
