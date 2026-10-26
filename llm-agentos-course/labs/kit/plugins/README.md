# Example plugins for Lab 10

Each folder is one plugin for `nanoos/plugins.py`: a `plugin.json` manifest (id, kind, config schema) and a
`plugin.py` with `register(api)`.

| Plugin | Kind | What it does |
|--------|------|--------------|
| [`shell-guard`](shell-guard/) | hooks | Logs every `run_shell` command to a JSONL file and blocks installers piped from curl or wget into a shell |
| [`secret-redactor`](secret-redactor/) | hooks | Replaces strings that look like API keys in tool results |
| [`threshold-compaction`](threshold-compaction/) | `context-engine` slot | Lab 3's `ThresholdCompaction` behind a slot |
| [`tool-result-clearing`](tool-result-clearing/) | `context-engine` slot | Lab 3's `ToolResultClearing` behind a slot |

The two context engines share one slot, so a host can load only one of them. `plugins_eval.py` uses
`shell-guard`; see the [Lab 10 handout](../handouts/lab10_extension_layer.md).
