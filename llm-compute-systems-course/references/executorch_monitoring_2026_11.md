# ExecuTorch: version monitoring and qualification — 2026-11-07

This fixes the monitoring blind spot identified by the [compute audit](research_audit_2026_11.md). It does not migrate the L11 runtime or re-certify historical results.

## Observed evidence

| Evidence lane | Observation | What it does and does not establish |
|---|---|---|
| Qualified course baseline | ExecuTorch 1.0.0 + torch 2.9.1, unquantized XNNPACK CPU; original `verified` date 2026-10-27 | Historical small-fixture export/lower/load/execute evidence in [L11](../labs/kit/handouts/lab11_deployment.md); no new compatibility run in this change |
| Registry | [PyPI JSON](https://pypi.org/pypi/executorch/json) reported `info.version = 1.6.0` and 20 listed files; [version-specific project page](https://pypi.org/project/executorch/1.6.0/) accessible | An upstream release/artifact discovery signal. No candidate wheel installed or producer/loader/backend behavior qualified |
| Documentation | [Moving stable getting-started page](https://docs.pytorch.org/executorch/stable/getting-started.html) rendered the title “ExecuTorch 1.5 documentation” in the web reader | Record the documentation version separately from the registry version. Matching versioned 1.6 documentation was not established; no claim of 1.6 support for the old API |
| Historical workflow source | [Versioned 1.0 getting started](https://docs.pytorch.org/executorch/1.0/getting-started.html) accessible in the web reader | Documents installation, export and runtime steps for that release. Does not imply interoperability with newer runtimes |

The direct registry request succeeded. A direct scripted docs request returned HTTP 403; stable and 1.0 were read through the web reader. Attempts to retrieve versioned 1.5/1.6 pages did not establish accessible matching documentation. Retrieval failure is not evidence of runtime incompatibility. Dates here describe source observations, not a new runtime `verified` date.

## What changed in monitoring

`tracked_sources.json` now uses `compare: 2` for the ExecuTorch entry. The existing checker compares the first two dot-separated components:

| Comparison rule | Baseline | Observed upstream | Significant values | Report a change? |
|---|---|---|---|---|
| Old `compare: 1` | 1.0.0 | 1.6.0 | 1 / 1 | No: the minor movement was hidden |
| New `compare: 2` | 1.0.0 | 1.6.0 | 1.0 / 1.6 | Yes: request a compatibility review |
| New rule, hypothetical patch control | 1.0.0 | 1.0.1 | 1.0 / 1.0 | No: this threshold deliberately suppresses patch-only movement |
| New rule, hypothetical major control | 1.0.0 | 2.0.0 | 1.0 / 2.0 | Yes |

This is an alert-granularity rule, not a semantic-versioning safety promise. Patch releases can also change behavior and warrant a manual review. All old pin/version/verification fields stay intact; the tracker lists the bilingual closing slide and this note as review targets. No other source threshold is changed.

To inspect the normal report from the repository root:

```sh
python scripts/check_freshness.py --no-links --report /tmp/freshness-new.md
```

Exit 1 means the scan found review work, including the now-visible ExecuTorch movement; it does not mean this monitoring correction failed. Other package movements remain independent work. This pass does not close the shared freshness issue or assert all course dependencies are current.

## What remains before a candidate becomes the baseline

Follow [L11's candidate-environment worksheet](../labs/kit/handouts/lab11_deployment.md#e-version-discovery-is-not-a-qualified-environment): freeze input/oracle constraints, exact package/build/source identities and matching documentation; export and lower; verify actual delegation/fallback; reload in a fresh process; check eligible outputs and an invalid-shape rejection. Record artifact identity and cold/warm timing boundaries. Cross-version loading is a separate matrix cell. A `.pte` suffix is not a compatibility contract.

The new original SVG on deck 23 slide 18 separates these evidence lanes and gates. The English/Chinese twins retain 18 slides. Their gate chain is a teaching protocol, not a claim that the 1.6 pilot ran. CMP-1's isolated old/new environment matrix remains pending.

## Verification record for this correction

The normal `--no-links` scan completed with **6 changed, 0 due, 0 unchecked** and exit 1, now including `L11 ExecuTorch CPU lab pin: 1.0.0 → 1.6.0`. The old threshold would suppress that row. Offline controls using the actual `significant()` implementation confirmed that 1.0.1 stays below this threshold, while 1.6.0 and 2.0.0 trigger review. All non-ExecuTorch tracker entries, the ExecuTorch baseline/date and both requirements files were checked unchanged against the parent main revision. No claim that other freshness findings are resolved.

## Later qualified CPU cohort — 2026-11-14 UTC

The [two-environment CPU matrix](executorch_cpu_compatibility_2026_11.md) now supplies the previously missing candidate export/lower/load/execute evidence for one frozen MLP and four actual producer/consumer cells. It uses pinned source documentation when the web-reader 1.6 page remains inaccessible. The earlier monitoring observations above are historical; the tested default remains 1.0.0 and the candidate has a separate scoped lock/report.
