# AI-assisted development and verification of inference software

Primary sources reviewed 2026-10-31; firmware/runtime testing update rechecked 2026-11-01. Supports decks 37–39 and the standard-library [CPU lab](../labs/kit/handouts/lab_ai_verification.md). The three lectures connect existing optimization projects P08/P09/P22 to independent evidence gates. They do not claim that kernel search, differential testing or deployment validation are new.

## Sources and scope

- [NVIDIA ComputeEval](https://github.com/NVIDIA/compute-eval): existing generated-CUDA evaluation and sanitizer integration. Source/documentation reviewed; not installed or executed for this unit.
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html): memory, shared-memory hazards, initialization and synchronization tools have distinct supported scopes. A clean run is evidence on exercised cases, not a race-free proof of the full software stack.
- [NVIDIA floating-point guide](https://docs.nvidia.com/cuda/floating-point/index.html): CPU/GPU comparisons require attention to operation order, FMA and implementation choices. Numerical tolerances and task quality are different from bit-exact protocol requirements.
- [NNSmith, arXiv:2207.13066 (v2)](https://arxiv.org/html/2207.13066v2), §§3–5: valid graph generation and differential compiler testing. Invalid-format rejection is a separate input-validation exercise.
- [Linux fault injection](https://docs.kernel.org/fault-injection/fault-injection.html): existing fault mechanisms motivate bounded failure/cleanup tests. The course does not run these privileged mechanisms on the authoring host.
- [Zephyr Twister](https://docs.zephyrproject.org/latest/develop/twister/index.html): distinguishes build-only, simulation and device execution; useful HIL orchestration reference, not evidence of an AI-accelerator driver implementation.
- [NVIDIA telemetry](https://docs.nvidia.com/deploy/nvidia-smi/): supported power, temperature, clock and error counters can support target observations, subject to device capability and sampling limits.
- [Synopsys PVT definition](https://www.synopsys.com/glossary/what-are-pvt-sensors.html): process, voltage and temperature. A functional emulator or FPGA prototype does not itself reproduce the final silicon's physical behavior.

## Corrections carried into teaching

AI-generated code increases the need for efficient oracles, failure triage and reproducible evidence; no measured “geometric” growth of bugs is asserted. Workload layout can affect traffic, power and throttling, but the unit does not generalize that an unusual kernel layout damages SRAM. Ordinary floating-point optimization need not match a CPU reference bit for bit. Formal reasoning complements dynamic testing and depends on a precise model and scope; it is not a blanket prerequisite for every candidate. Software observability is useful testability engineering, not a replacement for scan/BIST or silicon DFT. Malformed-graph robustness and dependency/artifact supply-chain integrity need different checks.

## Released versus target-dependent experiments

The executed CPU lab evaluates fixed numerical input cases and validates explicit event lists. Its reference input range avoids squared-value overflow; it is not a production-general norm implementation. It neither runs arbitrary generated code nor explores all thread schedules. Candidate hashes identify teaching functions, not signed artifacts. Supported GPU sanitizers, real concurrent queue execution, silicon power/thermal measurement and firmware recovery require their own target qualification.

The companion accelerator course develops an executable virtual queue/DMA/firmware lifecycle model and additional invariants. Its functional results remain separate from actual silicon and kernel-driver verification.

## Firmware/runtime testing feedback incorporated — 2026-11-01

Primary documentation above rechecked on 2026-11-01, together with the [Linux DMA mapping guide](https://docs.kernel.org/core-api/dma-api-howto.html). The added slides are proposed teaching protocols derived from these sources, not reports of executed target tests or an industry survey.

| Feedback | Course treatment | Boundary |
|---|---|---|
| AI can produce more candidates than evaluators can qualify | Keep deck 37's independent evaluator; add concrete acceptance matrices in deck 38 | No unsupported numerical/geometric claim about bug growth |
| Hardware and power matter beyond output correctness | Deck 39: matched baseline/candidate runs, sensor definitions, completed-work energy and stop conditions | Board telemetry cannot diagnose local SRAM hotspots; no claim of AI layouts burning chips |
| DMA, interrupts, races and deadlocks | Deck 38: lost-IRQ/timeout ownership example; safety versus liveness, fault replay and bounded model scope | Sanitizers and finite schedule tests do not prove whole-stack correctness |
| Low precision, fusion and operation reordering | Deck 38: exact protocol versus per-operation numerical and held-out quality contracts | Neither CPU high precision nor a single strict tolerance is a universal oracle |
| Graph mutation and malformed inputs | Deck 38: valid differential tests, invalid-boundary rejection and resource-heavy valid cases | NNSmith covers valid graph generation; robustness is distinct from supply-chain provenance |
| Software testability and HIL automation | Deck 39: trace IDs, generation, progress, clock domains; one-board recovery before scaling | Software hooks are not scan/BIST; no universal vendor practice or automatic fleet flashing claimed |

### Primary-source reading guide

- **Numerics:** NVIDIA floating-point guide §§2–5 explains why association, FMA and CPU/GPU choices matter. The course acceptance matrix is a test-design recommendation; it does not prescribe an official NVIDIA tolerance or a threshold for every low-bit format.
- **Concurrency:** Linux DMA mapping guide, “Types of DMA mappings” and “Using Consistent DMA mappings,” explains ownership/synchronization and why coherent DMA memory still needs ordering barriers. The lost-interrupt schedule is an original teaching example, not a reported Linux bug. Linux fault injection supplies selected kernel failure mechanisms, not a generic injector for arbitrary accelerator interrupts.
- **Graph testing:** NNSmith §§3–5 describes operator constraints, generation and numerical validity. Pair this valid-input campaign with a separately specified parser/API rejection campaign; neither proves arbitrary firmware robustness.
- **Physical evidence:** NVIDIA SMI documentation defines device-dependent telemetry and unsupported fields. Sampling/metric semantics must be recorded. Energy estimates, counter correlation and matched-work comparisons are the proposed experimental method, not results from this course.
- **HIL:** Twister documents build-only, simulated and device execution and hardware maps. It illustrates evidence separation and orchestration; actual accelerator deployment/recovery depends on the vendor target.

The [qualification worksheet](../labs/kit/handouts/lab_ai_verification.md#d-design-a-firmwareruntime-qualification-campaign) connects the six added slides to explicit deliverables. The companion course's [verification](../../ai-accelerator-software-course/slides/05_verification.html) and [qualification](../../ai-accelerator-software-course/slides/06_qualification.html) decks retain the executable virtual-device progression; their existing content already covers fault replay, bounded reasoning, one-board HIL, telemetry and recovery.
