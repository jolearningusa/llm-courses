# vLLM 0.33.0 — source identity before runtime qualification

Observed 2026-11-08. CMP-2 follow-up, first evidence gate only. The course's historical vLLM 0.32.0 comparison cohort, tracker `verified` date and dependency pins remain unchanged. This is not a migration or a qualified deployment.

## What was actually checked

The [exact-version PyPI metadata](https://pypi.org/pypi/vllm/0.33.0/json) lists three unyanked artifacts. The [release file page](https://pypi.org/project/vllm/0.33.0/#files) lists two Linux `manylinux_2_28` wheels (x86-64 and AArch64) and one source distribution. Only the source archive was downloaded; wheels were neither downloaded nor installed. The [machine-readable record](vllm_source_identity_2026_11.json) preserves file URLs, sizes, registry digests and inspection scope.

| Evidence | Observed result |
|---|---|
| Downloaded `vllm-0.33.0.tar.gz` | 47,101,396 bytes; computed SHA-256 matches PyPI: `f4c33c2503d52bb411146afa9618a4f48bed68912f617e6803ac8d76e4c74112` |
| Archive `PKG-INFO` | Name `vllm`, Version `0.33.0`, Requires-Python `<3.15,>=3.11` |
| Archive `vllm/_version.py`, read as text | Version `0.33.0`; declared commit `g4ba787ec9` |
| [Git tag reference](https://api.github.com/repos/vllm-project/vllm/git/ref/tags/v0.33.0) | Annotated tag object `3a4d82f569a9f57054ba81633e9b6c73b34b51fe` |
| [Dereferenced tag object](https://api.github.com/repos/vllm-project/vllm/git/tags/3a4d82f569a9f57054ba81633e9b6c73b34b51fe) | Commit `4ba787ec9959e566d3b587a796dff07673324da5`, matching the archive's declared short ID |
| [Release REST endpoint](https://api.github.com/repos/vllm-project/vllm/releases/tags/v0.33.0) | HTTP 404 at observation time; this is separate from the existing Git tag and does not establish a changelog |

The wheel filename's `cp38-abi3` tag does **not** override the package's Python requirement. Published artifacts are candidates for an eligible environment, not evidence that installation/dependency resolution will succeed on a particular host. The inspection host was macOS; no Linux/GPU runtime was qualified.

Two archive members were compared byte-for-byte with raw files at the exact commit, with equality for both:

- [NIXL connector facade](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py): SHA-256 `36dbc6a182ba753caf06fb40d40baa3b218371c62323f3ade29f8dd50514bd8a`.
- [NIXL compatibility documentation](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/docs/features/nixl_connector_compatibility.md): SHA-256 `0c9590d05a0e22b06b5f4763f4704e80b794a7476bdaa31066434ca070db6c50`.

These samples establish a concrete starting point for the connector walkthrough. They do not establish whole-tree equivalence, wheel contents, an authenticated build chain or reproduced binary builds. Version metadata is a declaration; a digest proves equality with the observed registry record, not independent trust in its publisher. No new engine-feature or performance claim is inferred from a version number.

## Repeat the read-only inspection

Resolve `git/ref/tags/v0.33.0`, then dereference its tag object using the two API links above. Save the full commit. Fetch the exact-version PyPI JSON and select its `sdist` row; save the artifact URL, size and SHA-256. Hash the downloaded bytes **before** reading archive members. Do not install or import the downloaded package to inspect its version.

For a locally saved archive, the standard library can read only the required members without extracting or executing them:

```python
import email, hashlib, tarfile
from pathlib import Path

path = Path('/tmp/vllm-0.33.0.tar.gz')  # downloaded from the recorded PyPI URL
expected = 'f4c33c2503d52bb411146afa9618a4f48bed68912f617e6803ac8d76e4c74112'
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
with tarfile.open(path) as archive:
    prefix = 'vllm-0.33.0/'
    metadata = email.message_from_bytes(archive.extractfile(prefix + 'PKG-INFO').read())
    print(metadata['Version'], metadata['Requires-Python'])
    print(archive.extractfile(prefix + 'vllm/_version.py').read().decode())
```

Compare each sampled member's bytes with its raw GitHub URL at the **full commit**, as recorded in the JSON manifest. If metadata, digest, tag or file equality differs, retain the discrepancy and stop this identity gate rather than silently replacing the snapshot. The manifest is a dated observation, not a continuously refreshed compatibility verdict.

## Remaining CMP-2 execution gates

1. **Environment:** select a documented Linux/hardware lane; freeze Python, driver, accelerator runtime, dependencies, wheel digest, model revision and request data. Resolve/install in isolation; retain resolver and import logs. A source archive alone is not an installability result.
2. **Contract:** walk the pinned NIXL scheduler/worker delegation, configuration and version compatibility. Map producer/consumer, transfer ownership, completion, cancellation and failure/retry semantics to exact methods. Reading source is not executing the connector.
3. **Correctness:** exercise baseline generation and transfer, cancellation during transfer and injected peer failure; assert no stale KV reuse, stranded blocks or request leakage. Save upstream test selections, actual target logs and unsupported cases separately.
4. **Performance:** run cold, warm and shared-prefix controls with fixed model/quality and request arrival pattern; report TTFT, ITL, goodput, transfer volume and peak memory. Compare equivalent states and include failed requests. No simulator timing substitutes for engine measurements.

**Not run:** wheel inspection, complete-tree comparison, dependency resolution, installation/build/import, model loading, upstream connector tests, GPU/network transfer, cancellation/failure injection and benchmarks. The first source-identity gate is complete; CMP-2 as a whole remains open. The matched AgentOS memory benchmark is also still pending independently.
