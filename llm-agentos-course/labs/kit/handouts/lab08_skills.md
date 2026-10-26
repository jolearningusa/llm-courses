# Lab 8: skills

**Lecture:** [skills](../../../slides/05_skills_deck.html), Module 5 · **Week:** 8 ·
**Autograder:** `make grade LAB=8` · **Budget:** Part A is free. Part B needs 5 tasks × 10 trials × 3
configurations, which is about 150 short runs. Part C needs a Hermes Agent or OpenClaw install.

A skill is a procedure on disk: a folder with a `SKILL.md` and, optionally, references and scripts. Skills cost
almost nothing until they are used, because only their names and descriptions sit in the prompt. The body is
paged in when the model decides it needs it. This lab builds that loading path, then measures whether a skill
actually helps and what the index costs.

## Part A: discovery, the index and progressive disclosure (graded)

Implement the four TODOs in `nanoos/skills.py`, following their docstrings:

1. **`validate`.** Check the frontmatter against the [Agent Skills specification](https://agentskills.io/specification).
   - `name`: 1–64 lowercase letters, digits and single hyphens, and the same as the folder name.
   - `description`: required, at most 1,024 characters.
   - `compatibility`: optional, at most 500 characters.
2. **`discover`.** Find skills under several roots in precedence order, the way OpenClaw and Codex do.
   - Category folders are allowed above a skill.
   - The first root wins a name clash; warn about the shadowed one.
   - Skip a malformed or invalid `SKILL.md` with a warning, never an exception.
   - Skip a `SKILL.md` whose real path leaves its root (OpenClaw's realpath containment).
3. **`render_index`.** Level 1: one line per skill for the system prompt, within a character budget. Over the
   budget, drop the descriptions first, then the names that do not fit. The output must be deterministic, so
   the prompt prefix stays cacheable (Lab 5).
4. **`SkillTools.skill_view`.** Level 2 returns the body without the frontmatter and lists the skill's files.
   Level 3 returns one file, which must stay inside the skill folder: no `..`, no absolute path, no symlink
   that leads out.

The frontmatter parser, `skills_list`, and `SkillIndexEngine` (which adds the index to the system prompt) are
provided. `make grade LAB=8` runs 15 tests. One of them checks that the skill body reaches the model only after
`skill_view`, and a reference file only after it is asked for.

## Part B: does the skill help? (report)

`tasks/skills/` holds one skill worth having and 49 that are not. **`report-fill`** describes the team's
incident-report format: four fixed sections, times in UTC, a severity table in `references/severity.md`, and
action items with owners. It also has a script that prints the report. The four report tasks are checked
strictly, the way a script downstream of the agent would read the report. A fifth task is a control question
that needs no skill.

Offline, the scripted models show the two ends:

```bash
python skills_eval.py --skills none --out runs/lab8-demo-none    # does not know the format: 0%
python skills_eval.py --skills all --out runs/lab8-demo-all      # opens report-fill: 100%
```

Then run your model through your Lab 1 loop, 10 trials each:

```bash
for s in none one all; do
  python skills_eval.py --skills $s --model api --loop nanoos --trials 10 --out runs/lab8-$s
done
```

Report:

- **pass@1 and pass^10** on the report tasks, for none, one and all. pass^k is the share of tasks that pass in
  every trial. Does the skill make the agent reliable, or only better on average?
- **Activation rate**: how often the model opened `report-fill` when it needed it. Also report false
  activations on the control task, and whether the model read `references/severity.md`.
- **Index tokens** with 1 and with 50 skills installed. Does activation drop when the skill is one line among
  50?
- **The budget.** Rerun `--skills all` with `--budget-chars 600`, so that the descriptions are dropped. What
  happens to activation when the model sees only names?
- **Your own skill.** Write a second skill for a task your agent fails often (`--root` adds your skill folder),
  validate it with `skills-ref validate` from [agentskills/agentskills](https://github.com/agentskills/agentskills),
  and report 10 trials without and with it.

With your Lab 7 `tools.py` and `sandbox.py`, add `--shell` so the model can run the skill's script in the
sandbox. Does it run the script or write the report by hand?

## Part C: the same skill in real harnesses (report)

Hermes Agent and OpenClaw both read `<project>/.agents/skills/`. Write the incidents and the skill into a
project folder:

```bash
python -c "from pathlib import Path; from tasks.skills import install; install(Path('/tmp/lab8'), 'one')"
```

1. Open the harness in that folder and run the four report tasks 10 times each. Score each report with
   `check_report(Path('/tmp/lab8'), 'INC-2041')` from `tasks/skills/`.
2. Turn on Hermes's `skills.write_approval: true`. Start from a weaker version of the skill (delete the
   Pitfalls section) and let Hermes patch it after failures. Review each staged diff with `/skills diff`
   before you approve it. Report the pass rate of the hand-written skill against the agent-patched one, and
   compare with the deck: self-generated skills helped little in SkillsBench, while Voyager kept a skill only
   after it worked.

**Supply chain.** A skill body is a prompt your agent trusts, and a skill script is code it runs. Pick one
third-party skill from a public hub. Before you install it, review it and report:

- what its scripts do;
- what they download;
- which network or credential instructions its body contains;
- what your sandbox from Lab 7 would and would not stop.

Do not run untrusted skills outside the sandbox. This lab does not plant an instruction in a skill file. To test
an attack, use Lab 6's red-team fixtures under its safety rules.

Hand in one page from [`report_template.md`](../report_template.md).

## Grading

| Part | Weight |
|------|-------:|
| Autograder (`make grade LAB=8`) | 50% |
| Report: pass@1 and pass^10, activation, index tokens and budget, your own skill | 25% |
| Report: real harnesses, the hand-written skill against the agent-patched one, the supply-chain review | 15% |
| Code quality of `skills.py` | 10% |

## Stretch

- **Gating.** Read OpenClaw's `metadata.openclaw.requires` (`bins`, `env`) and leave out skills whose
  requirements are not met. Report how many of your installed skills would be hidden on a fresh machine.
- **Explicit invocation.** Support `/report-fill` at the start of a task to open a skill without relying on the
  description. Compare activation with implicit selection.
- **A snapshot.** Freeze the index at session start, like OpenClaw, and show with Lab 5's `CacheAuditModel` that
  installing a skill mid-session breaks the cache only if you rebuild the index.
