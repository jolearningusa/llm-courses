# Lab 7: tools and the sandbox

**Lecture:** [tools and the sandbox](../../../slides/05_tools_sandbox_deck.html), Module 5 · **Week:** 7 ·
**Autograder:** `make grade LAB=7` · **Budget:** Part A is free. Part B needs about 5 tasks × 3 trials × 4
configurations, which is roughly 60 short runs. Part C needs Docker or Linux user namespaces, and a Hermes or
OpenClaw install.

Tools are the agent's system calls. The model reads a tool's name, description and parameter schema the way a
program reads an API reference, except that it cannot ask what you meant. And every command the model runs
should run inside a boundary, because the arguments come from a model that untrusted input can steer (Lab 6).
This lab builds both halves: a tool registry that is pleasant for a model to use, and a sandbox for the shell.

## Part A: a tool registry and a sandbox (graded)

Implement the TODOs, following their docstrings.

In `nanoos/tools.py`:

1. **`json_type` and `schema_from_function`.** Build an OpenAI-style tool schema from a plain Python function:
   types from the type hints (`str`, `int`, `float`, `bool`, `list[X]`, `Literal[...]` as an enum, `Optional`),
   descriptions from the docstring's `Args:` section, defaults, and the required list. A parameter without a
   type hint is an error, because the model needs a type.
2. **`ToolRegistry.call`.** Check the model's arguments before calling the function, and turn every failure into
   a message the model can act on: an unknown tool lists the real ones, an unknown argument lists the valid
   ones, a missing argument gives its description, and a wrong type or enum value says what was expected and
   what arrived. A `ToolError` raised by the tool becomes its message; any other exception is reported, not
   raised. A result longer than `max_chars` is cut, and the full text is saved under `.tool_outputs/` with a
   pointer the model can follow with `read_file`.

In `nanoos/sandbox.py`:

3. **`Sandbox.run`.** Run one command with a timeout, a clean environment and resource limits. A timeout kills
   the whole process group and comes back as a result, not an exception. Output is capped.
4. **`DockerSandbox.argv`.** The `docker run` command line: no network, read-only root file system, a small
   `/tmp`, memory, CPU and PID limits, no capabilities, and the workspace mounted read-only unless you ask for a
   writable one.
5. **`run_shell`** inside `shell_tool`. Format a result for the model. A timeout is a `ToolError` with a hint;
   a non-zero exit code is ordinary output that the model reads.

Three backends share one interface:

| Backend | Isolation | Needs |
|---------|-----------|-------|
| `HostSandbox` | None. Limits and a clean environment only | Nothing |
| `NamespaceSandbox` | No network; the whole file system read-only except the workspace (read-only too unless `writable=True`). The idea behind bubblewrap | Linux with unprivileged user namespaces (`unshare`) |
| `DockerSandbox` | A throwaway container with no network, a read-only root and the limits above | A Docker daemon |

`best_available()` picks Docker, then namespaces, then the host. `make grade LAB=7` runs 21 tests. The two
namespace tests check that the network is unreachable and that a write outside the workspace fails; they are
skipped where `unshare` is not allowed (macOS, some containers), so run them on a Linux machine before you hand
in.

## Part B: what tool design costs (report)

`tasks/tickets.py` is a support-ticket tracker: 60 tickets in `tickets.json` and five questions about them. Two
tool designs read it:

- **v1:** `search(q)`, documented only as "Search.", returns every matching ticket as raw JSON, bodies included.
- **v2:** `search_tickets(query, status, priority, page, response_format)` returns 10 tickets per page, one line
  each in the concise format, and says how many pages there are. `get_ticket(ticket_id)` returns one ticket.

Both get `read_file` and a sandboxed `run_shell`. Offline, the scripted solutions show the token difference
before any model is involved:

```bash
python tools_eval.py --tools v1 --out runs/lab7-demo-v1
python tools_eval.py --tools v2 --out runs/lab7-demo-v2
```

Then run your model through your Lab 1 loop, 3 trials each:

```bash
python tools_eval.py --tools v1 --model api --loop nanoos --trials 3 --out runs/lab7-v1
python tools_eval.py --tools v2 --model api --loop nanoos --trials 3 --out runs/lab7-v2
```

Report:

- **Tokens per task and success** for v1 and v2. Where does v1 spend its tokens? Look for the truncation
  pointers in the traces, and whether the model followed them with `read_file` or `run_shell`.
- **Concise against detailed.** Change the default of `response_format` to `"detailed"` and rerun v2. What did
  the bodies cost, and did any answer need them?
- **Tool-call errors before and after rewriting descriptions.** `summary.json` counts the calls the registry
  rejected. Find the errors in the traces (a wrong argument name, a bad page, an id in the wrong form), rewrite
  the descriptions or error messages that caused them, and rerun. Keep a diff of what you changed.
- **The truncation limit.** Rerun v1 with `--max-chars 2000` and `--max-chars 30000`. What happens to tokens and
  success?

## Part C: sandbox backends and real harnesses (report)

1. **Your backends.** Run the same probes on each backend you have, with `HostSandbox(ws).run(...)`,
   `NamespaceSandbox` and `DockerSandbox` from a Python shell:
   - reach the network (for example, open a socket to a public address);
   - write a file outside the workspace, and one inside it with and without `writable=True`;
   - print the environment, and check that your API key is not in it;
   - run a command that never ends.

   Give a table of which probe each backend stopped, and what each cost in start-up time per command
   (`duration_s`).
2. **A real harness.** Expose the v2 ticket tools as an MCP server (for example with FastMCP, wrapping the same
   functions) and run the five questions from Hermes with `terminal.backend: docker`, or from OpenClaw with
   `sandbox.mode: all` and `workspaceAccess: ro`. Repeat the four probes by asking the agent to run them.
   Report which probes the harness's sandbox stopped and how many approval prompts each task needed. Check
   the current configuration keys in each harness's documentation; the deck lists the ones it was written
   against.

This lab does not plant an injection in the tickets. The deck's exercise asks whether an injected command could
reach the network; the network probe above answers the same question directly. To test it with a real attack,
rerun Lab 6's red-team fixtures (`tasks/redteam/`) with `run_shell` inside the sandbox, under the safety rules in
[`tasks/redteam/README.md`](../tasks/redteam/).

Hand in one page from [`report_template.md`](../report_template.md).

## Grading

| Part | Weight |
|------|-------:|
| Autograder (`make grade LAB=7`) | 50% |
| Report: v1 against v2, concise against detailed, errors before and after rewriting descriptions | 25% |
| Report: the probe table and the real harness run | 15% |
| Code quality of `tools.py` and `sandbox.py` | 10% |

## Stretch

- **An output-size budget per session.** Count the characters every tool returns in one run and refuse (with
  an actionable error) once a budget is spent. Compare with plain truncation.
- **Seccomp or Landlock.** Add a backend that uses a seccomp filter or Landlock rules instead of namespaces,
  and run the probes on it.
- **Tool search.** Register 50 tools, put only their names in the prompt, and add a `find_tool` tool that
  returns full schemas on demand. Measure tokens and errors against putting all 50 schemas in the prompt.
