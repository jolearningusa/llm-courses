# Lab 14: A self-improving harness

**Lecture:** [Master deck, Module 7](../../../slides/00_master_agentos_deck.html) and the Harness² slides in the
[harness deep-dives](../../../slides/05_opensource_plugin_apis_slides.html) · **Week:** 14 ·
**Autograder:** `make grade LAB=14` · **Budget:** Part A and the offline run are free. Part B runs the loop with
your model: about 80 to 150 short agent runs and a few proposer calls per configuration (the summary reports the
exact count). Part C's demo is free; its LAB task needs Vertex AI credentials.

Harness² keeps the model fixed and changes the harness. It proposes harness edits, probes them on the task, and
composes a refined harness from what the probes show. It then keeps the adopted harness and the lessons the
evidence supports. This lab builds the same loop over `nanoos`. It ends with the question every improvement loop
has to answer: did the edits help on tasks the loop never saw?

## Part A: propose, probe, compose (graded)

[`nanoos/improve.py`](../nanoos/improve.py) treats three parts of the harness as editable:

| Target | Op | What it changes |
|--------|----|-----------------|
| `system_prompt` | `append` | Adds one line to the system prompt |
| `tool:<name>` | `set` | Replaces a tool's description (`list_files`, `read_file`, `write_file`, `search`) |
| `setting:max_steps` | `set` | The step budget, 1 to 50 |

A `HarnessConfig` holds them, and `run_task` runs a task under one config with any loop. The `Evaluator` counts
every agent run and token the loop spends.

The tasks are split. The six DEV tasks are the ones the loop sees: it evaluates them, shows their failures to the
proposer and probes edits on them. The four HELDOUT tasks are only used to check a composed harness.

Implement four TODOs:

1. **`parse_edits`.** Turn a proposer's reply into edits.
   - Accept a JSON array or `{"edits": [...]}`, inside prose or a code fence.
   - Drop every item that is not a valid edit for the table above.
   - Store `max_steps` as an int.
2. **`probe`.** Run the failing DEV tasks with one edit applied. The edit is adopted if it fixes at least one of
   them in every trial.
3. **`compose`.** Add the adopted edits one at a time, the one that fixed the most tasks first. Keep an edit only
   when DEV success rises and HELDOUT success does not fall. Skip an edit that sets a target an earlier kept edit
   already set.
4. **`improve`.** Run the rounds.
   - Evaluate, report failures, propose, probe and compose.
   - Stop when nothing fails or nothing new is proposed.
   - Log one lesson per proposed edit in the course's lesson schema
     ([`sample_lesson_entry.jsonl`](../../lab3_google_harness2/sample_lesson_entry.jsonl)).

**The simulated agent.** Scripted models ignore the harness, so they cannot test a loop that edits it.
[`tasks/improve_sim.py`](../tasks/improve_sim.py) provides `HabitModel`, a scripted agent with bad habits:

- it guesses answers without opening files;
- it opens only the first search hit;
- it writes paths without their folder.

The right edit removes each habit. One edit is harmful: a stricter `write_file` description fixes a DEV task but
makes the agent refuse to edit existing files on two HELDOUT tasks. The offline proposer
(`ScriptedProposer`) suggests a mix of edits that help, edits that do nothing, and that harmful one. On this
setup, a correct loop raises DEV from 33% to 83% and HELDOUT from 75% to 100%, rejects the harmful edit at
compose, and leaves one DEV task unsolved, because that task needs two edits together.

`make grade LAB=14` runs 11 tests:

- the parser on good and bad replies;
- probes that fix, do nothing, or only work in combination;
- compose's ordering, its held-out check and conflicts;
- the whole loop offline, including that the proposer never sees a HELDOUT task.

## Part B: run it on your model (report)

```bash
python improve_eval.py                                                 # offline, as the tests run it
python improve_eval.py --model api --proposer api --trials 2 --out runs/lab14-api
python improve_eval.py --model api --proposer demo --trials 2 --out runs/lab14-scripted
python improve_eval.py --model api --proposer api --loop nanoos --trials 2 --out runs/lab14-nanoos
```

`--trials 2` counts a task as passed only when both trials pass, as pass^2 does. A loop with one trial adopts
edits that only got lucky. Each run writes:

- `config_after.json`;
- `lessons.jsonl`;
- `history.json`;
- `summary.json`, with the agent runs, tokens and proposer calls the loop spent.

Report:

- **Before and after.** DEV and HELDOUT success per round, for your model with the model proposer and with the
  scripted one.
- **What it kept.** Each adopted edit, the failure it was proposed for, and whether it held on HELDOUT. Quote the
  final system prompt.
- **What it threw away.** The rejected edits, grouped by decision (`rejected_probe`, `no_gain`,
  `regressed_heldout`, `conflict`). Pick one rejected edit and say whether you agree with the decision.
- **Cost.** Agent runs, tokens and proposer calls the loop spent, against what the improvement saves per task
  afterwards. How many runs of the improved harness pay back the loop?
- **Noise.** Run the API configuration twice. Do the same edits get adopted?
- **Interacting edits.** One DEV task needs two edits together (see `tasks/improve_sim.py`). Did your model's
  loop hit a case like that? How would you change `probe` to find edit pairs, and what would it cost?

## Part C: Harness² (report)

Run Google Research's Harness² demo from the course's
[`lab3_google_harness2/`](../../lab3_google_harness2/) folder:

```bash
cd ../../lab3_google_harness2 && ./run_harness2_demo.sh          # offline demo, recursion budget k=2
./run_harness2_demo.sh --lab                                      # plus one LAB task (needs Vertex AI)
```

Read the demo's outputs under `harness2/runs/demo`. For each proposed edit, fill in one lesson record in the
course schema ([`sample_lesson_entry.jsonl`](../../lab3_google_harness2/sample_lesson_entry.jsonl)): target
component, failure signature, proposed edit, the probe's contrast, decision. Then compare Harness² with your loop:

- **Components.** Harness² edits eight components (system prompt, guardrails, memory, tool guidance, sub-agents,
  skills, scripts, plugins) against your three. Which of its adopted edits could your loop express?
- **Evidence.** Harness² has no ground-truth checker during the task: it probes edits by comparing behavior. Your
  loop uses each task's checker. What would your loop adopt without one?
- **Generalization.** Harness² carries lessons to later tasks. How many of your adopted edits held on HELDOUT, and
  how many of its lessons apply outside the task they came from?
- **Cost.** Compare the cost of both loops per adopted edit.

Hand in one page from [`report_template.md`](../report_template.md), with your filled-in lesson records attached.

## Grading

| Part | Weight |
|------|-------:|
| Autograder (`make grade LAB=14`) | 50% |
| Report: the loop on your model (before and after, kept and rejected edits, cost, noise) | 30% |
| Report: Harness² lesson records and the comparison | 10% |
| Code quality of `improve.py` | 10% |

## Stretch

- **Pairs.** When no single edit fixes a task, probe pairs of the round's rejected edits, and report how many
  runs that costs.
- **More components.** Make the Lab 3 context engine (`setting:context`) or the Lab 8 skill index editable, and
  let the proposer edit them.
- **A guard against overfitting.** Reject an edit that mentions a DEV task's answer or file names (the proposer
  prompt asks for this, but check it), and measure how often your model's proposals break the rule.
- **Lessons across runs.** Start a new loop from `config_after.json` and the earlier `lessons.jsonl`, pass the
  lessons to the proposer, and see whether it stops proposing edits that were already rejected.
