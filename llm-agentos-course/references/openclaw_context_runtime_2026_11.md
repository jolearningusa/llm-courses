# OpenClaw context-registry runtime controls — 2026-11-14 UTC

AGT-1 now includes execution of selected **actual bundled context-registry modules** from npm OpenClaw2026.10.7 and2026.11.3 in isolated fresh Node processes. Ten control groups pass in each cohort. Deterministic course callbacks supply both the custom engine and the fallback registered with ID`legacy`; the real LegacyContextEngine summarizer, Gateway, public plugin-SDK composition root and provider are not invoked. This measures registry/proxy contract behavior, not a full OpenClaw installation or LLM compaction result.

## Artifact and execution identity

[Protocol](../labs/kit/openclaw_context_protocol.json) frozen at `7c288d7`; [probe](../labs/kit/openclaw_context_eval.mjs) committed at `874dbf1` before control execution. Both npm archives were downloaded from their version-specific registry tarball URLs; SHA512 integrity and SHA1 shasum matched published metadata. SHA256 and selected packaged file hashes are in the protocol. Registry signatures/attestations were listed, not independently validated. npm metadata has no `gitHead`, so no full tag-to-bundle provenance claim follows from this archive check.

Actual runtime: Node24.22.1, macOSarm64. The old artifact declares `>=24.22.1 <25 || >=26.12.1`; the new artifact declares `>=24.16.0 <25 || >=26.1.0`. This Node satisfies both selected ranges. Isolated minimal dependency roots contain six/eight selected direct dependencies and locked transitives (29/54 lock entries). Install scripts were disabled; each package's existing bundled node_modules was retained and missing dependencies linked to its own temporary cohort. No global OpenClaw installation, account setup, daemon, live Gateway or package upgrade ran. These locks are the selected import closure's dependencies, not a complete supported OpenClaw environment lock.

The probe verifies packaged source hashes before import, verifies exported function identities, registers its deterministic fixtures using the packaged registry and imports the real resolver/proxy/health/runtime code unchanged. A Node resolution hook records main-process resolved file paths and SHA256 digests, rechecked afterward; dynamically spawned health-store worker closure is not instrumented. Private synthetic state and absent-config paths are explicitly selected. External HTTP/fetch/socket entry points are blocked by the trusted test harness; both runs report zero attempts. This is a local test control, not a hardened network security boundary.

| npm artifact | Control groups | External network attempts | Main-process resolved files | Probe seconds |
|---|---|---|---|---|
| 2026.10.7 | 10/10 | 0 | 635 | 0.382743 |
| 2026.11.3 | 10/10 | 0 | 681 | 0.514161 |

[Old raw result](openclaw_context_artifacts_2026_11/2026.10.7-results.json), [new raw result](openclaw_context_artifacts_2026_11/2026.11.3-results.json), [independent verification/setup record](openclaw_context_artifacts_2026_11/verification.json). Raw npm metadata, dependency locks, install logs, expected-error stderr and kit results are in the same [artifact folder](openclaw_context_artifacts_2026_11/). All20 groups and both cohorts are retained. Timings are descriptive short local execution/import intervals, not Gateway throughput, end-to-end install time, provider latency or a speed comparison. File-resolution counts include dependencies; they are not coverage percentages.

## Observed contracts

| Control | Actual behavior in both pinned cohorts |
|---|---|
| Registration ownership | A non-core owner cannot replace reserved`legacy`; another owner and an unapproved same-owner duplicate are rejected. |
| Host-parameter projection | An engine accepting only sessionKey receives ordinary messages/custom input and that key, but not synthetic prompt/runtimeSettings/sessionTarget/runtimeContext host fields. Compact always receives its abortSignal even with an empty accepted-host list. This projection filters the named host fields, not every arbitrary user field. |
| Fresh factories | Two real resolver calls create separate callback instances; ingesting session-A-only in one does not alter the other's compaction fixture memory. This is factory isolation, not a Gateway clone, persistence or actual application session test. |
| Assembly failure | The failing custom callback is invoked once, quarantined, and the same call gets the course fallback's result. Further calls/new ordinary resolver selections use fallback without reinvoking the failed factory. |
| Compaction failure | The **first compact call rethrows the original error object**, while recording quarantine. A later call on that wrapper uses the course fallback. Explicit same-owner refresh re-registration clears quarantine; a fresh resolver then runs the recovered custom callback. Recovery is explicit, not an automatic same-engine retry. |
| Invalid factory | Missing compact triggers contract-validation quarantine and fallback. |
| Caller abort before invocation | Original abort reason propagates; custom callback is not called and no engine quarantine is recorded. |
| Caller abort during compact | The signal reaches compact, its original reason propagates without quarantine, and a later un-aborted call on the same engine succeeds. No arbitrary sleep or live provider cancellation is involved. |
| Read-only discovery | Discovery-only registration falls back without constructing the custom engine or quarantining it. |
| Cleanup | Repeated disposal of one resolved engine calls the raw callback disposer exactly once. |

Three expected deliberate failures per cohort (assembly, compact, invalid contract) appear in stderr. They are exercised controls, not unexpected production incidents. Separate guard/abort cases establish that an ordinary failure, a discovery-only entry and caller cancellation do not share one fallback policy. Do not equate a final fallback result with success of the original custom compaction.

Private SQLite state files were created by the imported runtime. The probe checks in-process quarantine/re-registration behavior; crash recovery, cross-process health-record visibility, worker module coverage and durable cleanup are not qualified. An additional read-only Python SQLite inspection could not open the file and supplied no schema/row evidence. No such stronger claim is inferred. The original kit passes36/36 with local loopback access; that result remains separate from these20 actual registry control groups.

## Primary sources and limits

Read shipped `docs/concepts/context-engine.md`, `docs/plugins/architecture-internals/context-engines.md`, `docs/plugins/manifest.md` and selected bundled registry/runtime/empty-registry modules at their protocol hashes. Version-specific npm metadata: [2026.10.7](https://registry.npmjs.org/openclaw/2026.10.7), [2026.11.3](https://registry.npmjs.org/openclaw/2026.11.3); [official v2026.11.3 release](https://github.com/openclaw/openclaw/releases/tag/v2026.11.3). Public docs explain registration, accepted host parameters and compaction ownership; actual proxy execution supplies the first-error/cancellation distinctions above.

The pinned registry wraps guarded methods, projects host parameters, treats caller abort separately, quarantines other failures, and rethrows a first compact failure before later fallback. The runtime also contains a distinct logical-turn resolver whose source describes per-turn construction retry; that path is **not** exercised here. The observed ordinary-resolver quarantine behavior must not be generalized to every host/harness path. `ownsCompaction` does not exempt a real outgoing request from host budget checks; no tokenizer, provider boundary or pre-prompt-budget guard is executed here.

Initial setup attempts are retained in the verification record: an import without dependencies raised the expected missing proc-safe package; a directory assumption encountered bundled partial node_modules; a nonexistent bundled npm launcher was corrected to the installed npm11.19.1 with Node24.22.1. No expected control outcome or protocol changed. Neither published bundle was patched. The deterministic callbacks are fixtures, not copied upstream compaction code or a model summary. Original Lab12 student TODOs and the Python host/reference implementations are unchanged. The existing [Python OpenClawHost._engine](../labs/kit/nanoos/hosts.py) catches every engine exception and returns None, so its broad reply-path fallback does not represent the tested first-compaction-error or caller-abort distinctions. Use the actual registry probe for those operation-level claims; complete Python-host/production-host alignment remains a separate unqualified gate.

## Reproduce the selected closure

Extract the two npm tarballs with integrity verification into separate directories. Install the protocol's selected direct dependencies in separate temporary dependency roots with `npm install --ignore-scripts --no-audit --no-fund`; use the retained lock files, and link missing dependency packages without overwriting shipped bundles. This does not install all OpenClaw features. Under Node24.22.1, from the repository root:

```bash
OPENCLAW_STATE_DIR=/private/tmp/new-openclaw-state OPENCLAW_CONFIG_PATH=/private/tmp/new-openclaw-state/absent.json node llm-agentos-course/labs/kit/openclaw_context_eval.mjs 2026.11.3 /private/tmp/new-openclaw-package /private/tmp/new-openclaw-result.json
```

Use fresh state/output paths per cohort; results cannot be overwritten. Each output retains exact file identities. Complete installation and public plugin loading, real Gateway failed/retried sessions, actual Legacy/model compaction, provider/token budget, profile clone and persistent recovery remain open AGT-1 gates. This completes a bounded packaged-registry compatibility supplement; it does not close those gates or AGT-2's broader measured benchmark. PR#120 still completes only AGT-2 teaching/protocol.

## Dated follow-up — 2026-11-14 UTC

The subsequent [Python host correction](openclaw_host_errors_2026_11.md) now rethrows its first compact error before the model request. Earlier descriptions of its broad catch refer to the pre-fix host. Abort/full-registry equivalence remains unqualified.
