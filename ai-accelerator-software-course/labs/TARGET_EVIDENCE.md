# Target qualification and HIL evidence template

Use this as a report template; filling it does not establish authenticity or acceptance. The default CPU lab leaves real-device rows **not run**. Replace them only after executing a documented target path.

| Evidence | Status and required record |
|---|---|
| Target | Vendor/model/board revision, device ID and access owner |
| Software | Firmware/image hash, driver/runtime/SDK/compiler versions and supported combination |
| Interface | ABI, buffers/addressing, queue, sync, cancellation and reset contracts |
| Build | Source/artifact hashes, build logs and dependency identity |
| Functional | Reference outputs, errors/tolerances, invalid input rejection, negative controls |
| Execution coverage | Harness owner/revision, exact intended workload/target identity, per-run correlation, observed start/completion counts, filters and raw evidence channel; candidate stdout is not coverage proof |
| Concurrency/lifetime | Sanitizer/tool scope, adverse schedule, failure/recovery and raw traces |
| Sustained operation | Duration, arrivals, memory growth, error/reset counts, latency population |
| Physical | Real telemetry/instrument, power boundary, sampling interval, temperature/clocks and operating conditions |
| Recovery | Tested quiesce/reset, known-good artifact and rollback procedure |
| Decision | Passed / failed / unsupported / not-run per required property; reviewer and limits |

For every row, also record execution stage (`built_only`, `simulator_run`, `device_run`, `not_run` or `unsupported`), check outcomes, required-check list, source/build/target/run identity, raw log locations and reasons for missing coverage. Stage is not a correctness verdict. Missing hardware means not-run, not unsupported. A failed check remains failed even if another field is invalid. Do not accept physical qualification unless the designated device path actually ran and all declared physical requirements have valid passing evidence. See [L6's executed CPU matrix and policy controls](handouts/lab06_qualification.md#executed-matrix-and-policy-controls); its hashes and log references are not authenticated evidence.

For an initial HIL path, reserve one designated board; identify/recover it; deploy a known artifact through the supported runner; execute bounded tests; archive serial/device/runtime logs; restore and verify known-good operation. Zephyr Twister is a reference for the orchestration pattern, not a drop-in AI-chip adapter. Do not flash arbitrary firmware or inject faults on production devices.

Simulation, FPGA and final silicon have different evidence scopes. PVT means process, voltage and temperature; power/current telemetry is additional measurement. A functional emulator cannot validate final-silicon physical corners. Device temperature is not an energy meter; TDP is not measured task energy. Software counters/test hooks do not replace silicon scan/BIST or DFT.

If hardware is unavailable, submit a dry-run plan with all target execution rows marked not-run and the exact prerequisites. Do not convert a plan, manifest, signature or model pass into a hardware pass.


## Execution coverage before a clean result

Record coverage, numerical correctness and diagnostics separately. Missing/malformed coverage or missing required tool output makes a run invalid, even when output values happen to match. A valid execution record does not excuse a known finding or wrong result. Preserve all dimensions when a run is both missing evidence and known to have failed. Only a complete eligible run can earn a bounded pass.

The [L5 CPU protocol controls](handouts/lab05_state_exploration.md#execution-coverage-controls) exercise these distinctions using the existing virtual worker. They do not invoke Compute Sanitizer or prove real kernel launch coverage. An actual GPU adapter needs a trusted harness channel and target-specific observations, matching source/build/run identity and complete raw tool logs. Keep correlation data away from candidate-controlled output channels; a nonce is not an authenticity proof. In-process Python controls provide no hostile-code isolation.
