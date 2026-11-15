# OpenClaw Python host: visible compaction errors — 2026-11-14 UTC

This narrow AGT-1 correction follows the [actual npm registry probe](openclaw_context_runtime_2026_11.md). It changes the course Python teaching host, not OpenClaw itself. Base: `1d981c5`; CPython 3.12.14 in the previously qualified isolated CPU environment. No model service, Gateway or GPU is used.

## Primary evidence and contract

The unchanged npm [2026.10.7](https://registry.npmjs.org/openclaw/2026.10.7) and [2026.11.3](https://registry.npmjs.org/openclaw/2026.11.3) registry modules were executed in the preceding probe. Their `compact` error guard awaits quarantine recording and throws the original error; assembly errors use immediate fallback. The prior [protocol](../labs/kit/openclaw_context_protocol.json) pins the modules and tar integrity, with full raw evidence in the preceding note. The 2026.11.3 primary npm metadata was fetched again on this date; its integrity equals that retained probe artifact. This is not a fresh full-registry rerun or a claim of full Python/TypeScript equivalence.

`OpenClawHost._engine` formerly caught every engine exception and returned `None`, allowing a failed compaction to proceed to a model reply. It now records its existing per-host quarantine string and uses bare `raise` for `compact` exceptions. Other methods keep their existing immediate fallback. The caller receives the identical exception before any model request.

## Executed regression controls

[Three end-to-end controls](../labs/kit/tests/test_openclaw_host_errors.py) use a deterministic context fixture, `ScriptedModel` and no tools. The same tests were run before and after the correction, plus the existing kit suite:

| Control | Before | After |
| --- | --- | --- |
| First compact failure is the same exception; no model call; quarantine set; later explicit run can fall back | FAIL: RuntimeError not raised | PASS |
| Assembly failure falls back on the same attempt | PASS | PASS |
| Successful compaction gives normal reply and counts one compression | PASS | PASS |
| Existing kit (`grade.py kit`) | Not rerun before | 36/36 PASS |

The first control also checks the later request fits the 64-token budget, the failing compactor is not called again, and fallback does not count a successful compression. The successful control checks two assemblies and one compact call.

Commands, from `labs/kit/` (using the isolated interpreter above):

```sh
python -m unittest discover -s tests -p test_openclaw_host_errors.py -v
python grade.py kit
```

Raw [before](openclaw_host_errors_2026_11/before.log), [after](openclaw_host_errors_2026_11/after.log), [kit](openclaw_host_errors_2026_11/kit.log) logs and [source/log identities](openclaw_host_errors_2026_11/validation.json) are retained. The old failure is deliberate regression evidence, not an unresolved final failure.

## Limits and reading

Quarantine belongs to this Python host object; it is not the actual process-wide registry. A later explicit `run()` starts a new task/local transcript: it does not automatically resume the failed turn, replay prior history, clear quarantine or prove durable recovery. Caller-abort mapping, all other engine-method policies, full installation/public SDK/Gateway, provider/budget accounting, clone/profile/persistence and live model compaction remain unqualified. No tracker or default dependency pin advances. AGT-1 stays open; PR #120 supplied only AGT-2 teaching/protocol, with later measured pilot evidence separately dated.

Context-manager slide 14 now has an original EN/ZH before/after flow; its old reference list is retained in hidden reading notes. [Review captures](../../docs/reviews/agentos-openclaw-host-errors/README.md) show both versions.
