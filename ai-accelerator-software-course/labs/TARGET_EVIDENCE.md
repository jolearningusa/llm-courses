# Target qualification and HIL evidence template

Use this as a report template; filling it does not establish authenticity or acceptance. The default CPU lab leaves real-device rows **not run**. Replace them only after executing a documented target path.

| Evidence | Status and required record |
|---|---|
| Target | Vendor/model/board revision, device ID and access owner |
| Software | Firmware/image hash, driver/runtime/SDK/compiler versions and supported combination |
| Interface | ABI, buffers/addressing, queue, sync, cancellation and reset contracts |
| Build | Source/artifact hashes, build logs and dependency identity |
| Functional | Reference outputs, errors/tolerances, invalid input rejection, negative controls |
| Concurrency/lifetime | Sanitizer/tool scope, adverse schedule, failure/recovery and raw traces |
| Sustained operation | Duration, arrivals, memory growth, error/reset counts, latency population |
| Physical | Real telemetry/instrument, power boundary, sampling interval, temperature/clocks and operating conditions |
| Recovery | Tested quiesce/reset, known-good artifact and rollback procedure |
| Decision | Passed / failed / unsupported / not-run per required property; reviewer and limits |

For an initial HIL path, reserve one designated board; identify/recover it; deploy a known artifact through the supported runner; execute bounded tests; archive serial/device/runtime logs; restore and verify known-good operation. Zephyr Twister is a reference for the orchestration pattern, not a drop-in AI-chip adapter. Do not flash arbitrary firmware or inject faults on production devices.

Simulation, FPGA and final silicon have different evidence scopes. PVT means process, voltage and temperature; power/current telemetry is additional measurement. A functional emulator cannot validate final-silicon physical corners. Device temperature is not an energy meter; TDP is not measured task energy. Software counters/test hooks do not replace silicon scan/BIST or DFT.

If hardware is unavailable, submit a dry-run plan with all target execution rows marked not-run and the exact prerequisites. Do not convert a plan, manifest, signature or model pass into a hardware pass.
