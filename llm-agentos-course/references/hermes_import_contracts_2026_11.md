# Hermes optional imports and clone contracts — 2026-11-14 UTC

## A broken installation must stay visible

The Lab 12 adapter previously caught **every `ImportError`** while importing Hermes' `ContextEngine` and `MemoryProvider`, then silently defined offline stand-ins. A missing helper/dependency or removed symbol inside an existing Hermes installation could therefore look like a successful import of the course adapter while using neither upstream base class.

`nanoos/ports.py` now permits offline stand-ins only for `ModuleNotFoundError` whose missing module is exactly **`agent`**. Nested missing modules, partial/conflicting `agent` packages, and missing imported symbols propagate with their original traceback. This is a narrow loading correction. The code still does not certify any arbitrary package named `agent` as Hermes, and an importable interface alone is not a qualified agent session.

## Before / after controls

[Five fresh-process tests](../labs/kit/tests/test_hermes_imports.py) execute synthetic import-boundary fixtures, not models or installed fake Hermes releases:

| Fixture | Previous loader | Fixed loader |
|---|---|---|
| No `agent` package | Offline stand-ins allowed | Offline stand-ins allowed |
| Present interface classes | Those classes used | Those classes used |
| Missing nested dependency | Hidden by stand-ins | Original error propagates |
| Existing `agent` package missing a helper | Hidden by stand-ins | Original error propagates |
| Removed imported symbol | Hidden by stand-ins | Original error propagates |

All five tests pass after the fix. Running the same assertions against an isolated copy of the old adapter gives exactly three failed assertions, identifying the three false-fallback cases. Raw before/after logs are in the [result record](hermes_import_results_2026_11.json). No student context/memory TODO or private reference solution was filled in or published.

## Actual upstream imports, two isolated cohorts

This additional probe imports the **unmodified upstream source files** through ordinary Python imports and constructs the existing course adapter against their real abstract base classes. No stubs, AST rewriting or import mocks are used in these two cohorts.

| Cohort | Peeled commit | Actual observation |
|---|---|---|
| Hermes v0.22.5 | `bf0203a794a07aac3ff764972ab9087baa4e4f4e` | Six import/construction/clone assertions pass |
| Hermes v0.23.0 | `2bd378ed1b8d6bf2192cb26e0a55ff3091e6bd19` | Six import/construction/clone assertions pass |

Annotated tag objects were resolved separately from their peeled commits. Commit-addressed GitHub source archives were downloaded and SHA-256 hashed. The [source ledger](hermes_import_sources_2026_11.json) records archive URLs/digests and the complete observed Python import closure: seven upstream files in the old cohort, ten in the new cohort. Its hashes are checked before the probe, and every imported upstream module is required to appear there. Archive hashes identify the downloaded bytes, not signed release authenticity or a built/installable distribution.

Each cohort used a new **CPython 3.12.14 / macOS 27.0.1 arm64** venv, `include-system-site-packages = false`, with only bootstrap pip 25.0.1 installed. Source roots and the course kit were explicitly added to `sys.path` under Python `-I`; `HERMES_HOME` was a fresh temporary directory. The full Hermes distribution/dependency graph was **not installed**. Upstream's optional `jiter` preload returned false with `ModuleNotFoundError`, recorded rather than suppressed by our adapter. No SDK parser/CLI/model-provider support follows from these imports.

[The replayable probe](../labs/kit/hermes_import_eval.py) checks:

1. The adapter's immediate context base is the imported upstream `ContextEngine` object.
2. Its immediate memory base is the imported upstream `MemoryProvider` object.
3. Both adapter classes satisfy the upstream abstract-method requirements and construct.
4. The adapter's provided clone has distinct adapter and view-engine objects.
5. Updating the child's budget and token counters leaves the parent's values intact.
6. The provided memory-discovery method and tool schema are callable.

The parent retains `(context length, threshold, total tokens) = (1000, 750, 604)` after the child changes to `(200, 150, 15)`. The view engine is the provided no-op course `ContextEngine`, not an implemented summarizer. Memory construction/discovery does not initialize a store, persist data or test background profile propagation. These are six bounded assertion groups per cohort, not six end-to-end agent tasks.

The source-inventory guard initially rejected one unlisted v0.23.0 CLI helper. The actual imported helper files were added with their hashes, then both final cohorts reran successfully; the guard was not removed. The selected upstream modules were inspected directly, including base-class clone/defaults, optional preload and memory abstract methods.

## Reproduce

From the repository root, extract the two commit-addressed archives listed in the source ledger into separate directories using a safe archive extractor. Create separate CPython 3.12 venvs without system site-packages; no third-party package installation is needed for this selected import closure. Use the exact recorded Python build to compare this cohort.

```sh
python3.12 -m venv /tmp/hermes-old-env
python3.12 -m venv /tmp/hermes-new-env
/tmp/hermes-old-env/bin/python -I llm-agentos-course/labs/kit/hermes_import_eval.py \
  --source-root /tmp/hermes-old-source \
  --revision bf0203a794a07aac3ff764972ab9087baa4e4f4e --output /tmp/hermes-old-result.json
/tmp/hermes-new-env/bin/python -I llm-agentos-course/labs/kit/hermes_import_eval.py \
  --source-root /tmp/hermes-new-source \
  --revision 2bd378ed1b8d6bf2192cb26e0a55ff3091e6bd19 --output /tmp/hermes-new-result.json
/tmp/hermes-new-env/bin/python -I -m unittest discover \
  -s llm-agentos-course/labs/kit/tests -p test_hermes_imports.py -v
# Existing infrastructure checks, including local loopback test servers:
(cd llm-agentos-course/labs/kit && /tmp/hermes-new-env/bin/python grade.py kit)
```

Choose new output paths; the probe refuses to overwrite records. The kit check initially gave **34/36** because the restricted environment refused two loopback-server operations. Rerunning with local-network permission produced **36/36**; both logs are preserved. This is test-infrastructure access, not a Hermes provider or external service call.

## Scope and remaining AGT-1 gates

The selected real-base import/construction/provided-clone portion is now executed, and the silent broken-import fallback is fixed. **Compaction and failed/retried sessions remain unqualified**: the probe deliberately calls the existing student `compress()` scaffold and records its `NotImplementedError` as a TODO, never as a successful compaction. It does not run the Lab 12 grader with private solutions, Hermes' CLI/plugin registry or `AIAgent`, a full package installation, model/API calls, persistence, profile isolation or OpenClaw. No dependency tracker or `verified` date is advanced by this narrower result. AGT-1 remains open for those runtime/session gates; AGT-2's measured matched-memory benchmark is still pending beyond PR #120's teaching/protocol work.

Primary source scope: [v0.22.5 context base](https://github.com/NousResearch/hermes-agent/blob/bf0203a794a07aac3ff764972ab9087baa4e4f4e/agent/context_engine.py), [v0.23.0 context base](https://github.com/NousResearch/hermes-agent/blob/2bd378ed1b8d6bf2192cb26e0a55ff3091e6bd19/agent/context_engine.py), [v0.23.0 memory base](https://github.com/NousResearch/hermes-agent/blob/2bd378ed1b8d6bf2192cb26e0a55ff3091e6bd19/agent/memory_provider.py) and the additional exact imported paths in the source ledger. The course's [earlier contract comparison](research_audit_2026_11.md#hermes-agent-v0230-contract-check--2026-11-11) remains source-reading evidence, distinct from this executed subset.
