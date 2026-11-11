# Qualification matrix: stages, outcomes and evidence — 2026-11-10

ACC-2's bounded reporting/policy portion, based on main `9c8de10` after PR #134. The [L6 handout](../labs/handouts/lab06_qualification.md#executed-matrix-and-policy-controls) explains the exercise; [raw report](../labs/examples/qualification_report.json) retains actual CPU evidence. Deck 06 slides 2/4 (EN/ZH) distinguish execution stage from check outcome and show the executed matrix.

## Primary-source verification

Rechecked [Zephyr Twister's introduction](https://docs.zephyrproject.org/latest/develop/twister/index.html): it reports how tests ran or whether they were built only; missing devices or harness configuration can prevent execution. The [native status reference](https://docs.zephyrproject.org/latest/develop/twister/twister_statuses.html) distinguishes not-run, filtered, skipped, blocked, error, fail and pass. Its NOTRUN specifically concerns successfully built but unexecuted tests. Our broader course `not_run` can include an absent target build, so it is **not** a direct translation of Twister statuses. No Twister run was performed.

[Linux KUnit's introduction](https://docs.kernel.org/dev-tools/kunit/index.html) describes in-kernel tests and its UML/QEMU runner. This Python model is not KUnit, a kernel execution, or device qualification. These sources motivate scoped reporting; they do not certify this course runner or establish a universal physical acceptance suite.

## Actual execution

Python 3.9.6, standard library only, assertions enabled, on the platform recorded in JSON. From the lab directory:

```sh
python3 -m unittest discover -s tests -v
PYTHONHASHSEED=0 python3 qualification_matrix.py --output examples/qualification_report.json
```

41 tests passed: 24 original model tests, eight execution-coverage tests, nine new qualification-policy tests. The runner compiled three Python sources into checked-hash bytecode in a temporary directory, retained source/artifact hashes and compilation messages, and ran `experiments.report()`. Temporary bytecode is not distributed; reproduce it with the recorded interpreter, `PYTHONHASHSEED=0` and embedded filenames. Fixing the startup hash seed avoids differences in serialization of unordered code constants. No accelerator binary was built. The simulator label means the course's deterministic functional model, not an instruction-level or timing simulator.

The actual model checks pass; build-only execution is incomplete; physical execution is not-run and its decision incomplete. Device functionality, recovery and power/thermal all remain not-run. Missing target build and board identity remain null rather than invented. The nested raw report keeps scenarios, exploration results, seeded checks and simulated ticks. Required model checks are frozen by the runner; target worksheet checks are examples to refine for a future supported adapter.

The new unit tests exercise fabricated records, including an eligible policy positive control. They do not produce real device measurements. The key negative control rejects claimed device execution when the board is unavailable. Other controls retain failures alongside missing evidence and reject build/model promotion, identity mismatches and incomplete check sets. Two fresh runner invocations with the fixed seed matched after excluding the new run ID; bytecode hashes and all check results matched.

## Limits and remaining work

`decide()` verifies local record consistency and completeness; evidence references are not opened or authenticated. In-process Python is not hostile-code isolation. Identity expected by the harness and the required checks must never come from the candidate. A fabricated complete record can satisfy schema policy, which is why its positive unit control is not a device pass.

ACC-2 reporting and unavailable-board acceptance controls are implemented. A real build/runner adapter, supported board, observed deployment/execution logs, telemetry/instruments and target-specific recovery/physical requirements remain pending, alongside ACC-3. No GPU, sanitizer, KUnit, Twister, firmware, HIL or physical experiment ran. Existing tracker versions and older reports are unchanged.
