# Lab 10: the extension layer

**Lecture:** [the extension layer](../../../slides/05_extension_layer_deck.html), Module 5 · **Week:** 10 ·
**Autograder:** `make grade LAB=10` · **Budget:** Part A and the offline runs are free. Part B needs about 15 short
runs with your model. Part C needs Hermes Agent, OpenClaw (Node.js) and an MCP SDK.

Plugins are the agent's kernel modules: code that runs inside the harness, described by a manifest, hooked into
fixed points of the loop, and in some cases the only owner of a role (one context engine, one memory provider).
MCP servers are the user-space daemons: separate processes that offer tools over a protocol. This lab builds a
small plugin host for `nanoos`, then builds one feature three ways and compares them.

## Part A: a plugin host (graded)

A plugin is a folder with a `plugin.json` manifest (id, kind, config schema) and a `plugin.py` with
`register(api)`. The kit ships four in [`plugins/`](../plugins/). Implement the three TODOs in
`nanoos/plugins.py`:

1. **`validate_config`.** Check a plugin's config against the schema in its manifest and fill in defaults.
   Report unknown keys, missing required keys, and values of the wrong type or outside an enum.
2. **`PluginHost.load`.** Load the enabled plugins, in a defined order:
   - Unknown ids are an error, and `deny` wins over `enabled`.
   - The config is checked **before any plugin code runs**, as OpenClaw does with its manifests.
   - Slots: a plugin of kind `context-engine` or `memory` is loaded only if it is the slot's selection, or
     the only one of its kind. A second provider for a filled slot is an error at load time, not a surprise
     at run time.
3. **`HookBus.emit`.** Run one event's handlers by descending priority, with ties in registration order.
   - `before_tool_call` **decides**: the first block wins, and lower-priority handlers do not run.
   - `tool_result` **transforms**: each handler's output feeds the next.
   - `after_tool_call` **observes**.

   A handler that crashes disables its plugin. The loop goes on, and a crash in a decide hook blocks that one
   call (fail closed).

`read_manifest`, `discover`, `PluginAPI`, slot filling and `HookedTools`, which runs every tool call through the
hooks, are provided. `make grade LAB=10` runs 12 tests.

The kit's `threshold-compaction` and `tool-result-clearing` plugins put your Lab 3 engines behind the
`context-engine` slot: `host.slot("context-engine")` replaces the hard-coded engine in `run_suite.py`. Do the same
for your Lab 4 `AutoRecall` as a `memory` plugin.

## Part B: the shell guard in nanoos (report)

The deck's feature is "log every shell command the agent runs and block `curl | sh`". The kit's `shell-guard`
plugin does it with two hooks. `tasks/shell_ops.py` has five tasks on a **simulated shell that never executes
anything**:

- two ordinary commands;
- the vendor installer piped from `curl` into `sh`;
- the same installer piped from `wget` into `bash`;
- the same installer downloaded first and then run in two steps.

Run them offline:

```bash
python plugins_eval.py --enable none --out runs/lab10-none      # every command reaches the shell
python plugins_eval.py --out runs/lab10-guard                    # shell-guard: two of three installers stopped
python plugins_eval.py --set shell-guard.crash_on=wc --out runs/lab10-crash
python plugins_eval.py --set shell-guard.log_pat=x               # a typo in the config: fails at load
```

Then run the guarded configuration with your model (`--model api --loop nanoos --trials 3`) and report:

- **What the pattern misses.** The two-step installer gets through. Change `block_patterns`, or write a better
  check than a regex, so that it is stopped without blocking the ordinary commands. When your model is blocked,
  does it try another way of writing the command?
- **The crash.** In the crash run, the guard crashes on the first command and is disabled, so every installer
  after that reaches the shell. Should a crashed guard fail closed for the rest of the session instead? Change
  `HookBus` to support that and say what it costs.
- **Overhead.** The hook time per call is in `summary.json`. Compare it with the model's latency per call.
- **Ordering.** Enable `secret-redactor` as well, and give a tool result an `sk-`-style key. Show where the
  redaction happens relative to the guard's log, and explain the priorities you chose.

## Part C: the same feature three ways (report)

Build the shell guard as:

1. a **Hermes plugin** with a `pre_tool_call` hook and a `plugin.yaml`;
2. an **OpenClaw plugin** in TypeScript with `api.on("before_tool_call", ...)` and an `openclaw.plugin.json`
   with a `configSchema`;
3. an **MCP server** that offers its own `run_shell` tool, logs each command, and refuses piped installers.

Test each one against commands that cannot do harm: point the installer at `https://tools.example/install.sh`
(the `.example` domain never resolves), and run the harness inside your Lab 7 sandbox with the network off. For
each version, give it a bad config and see what fails and when. Then install a second plugin that also hooks tool
calls, and check the order.

Compare in a table:

- lines of code and setup steps;
- **whether the model can go around it.** A hook sees every tool call. The MCP wrapper sees only calls to its
  own tool, so it is bypassed whenever the harness also has a native shell tool. Show this happening;
- the latency each one adds per tool call;
- what happens when the guard crashes.

Hand in one page from [`report_template.md`](../report_template.md).

## Grading

| Part | Weight |
|------|-------:|
| Autograder (`make grade LAB=10`) | 50% |
| Report: the nanoos guard (the bypass and your fix, the crash, overhead, ordering) | 20% |
| Report: the three implementations compared | 20% |
| Code quality of `plugins.py` | 10% |

## Stretch

- **Isolation.** Run each plugin's hooks in a separate process (like Hermes's `plugins.isolation: host`), with a
  time limit per call. Measure the added latency and show that a plugin that hangs no longer stalls the loop.
- **Versioned API.** Add an `api_version` to the manifest and refuse plugins written for a newer host. What
  should happen to a plugin written for an older one?
