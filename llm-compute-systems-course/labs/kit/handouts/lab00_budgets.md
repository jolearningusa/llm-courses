# L0 — Predict resource use before profiling

**Prerequisites:** Python 3.9+, a CPU, and the workload/architecture lectures. Expected student time: 2–4 hours, pending pilot. This is an analytical lab; no GPU speed claim can be derived from it alone.

## 1. Reproduce the reference calculations

Run `python3 budgets.py --demo` from the kit directory. Confirm that the GQA cache example is 4 GiB. Explain why the 32-KV-head case is 16 GiB. Verify that the 4096-cubed GEMM has 137.44 GFLOP and 96 MiB ideal traffic. Keep GiB/MiB separate from decimal GB/TB/s.

## 2. Sweep batch size

Use M = 1, 8, 32, 128 and 4096 with K = N = 4096. Keep the assumed 100 TFLOP/s and 1 TB/s rates fixed. Tabulate work, ideal bytes, intensity and limiting term. Find the first M in your sweep that switches the limiting term. Repeat with compute or bandwidth halved and explain the direction of the change.

The calculator is provided as a reference tool. Write your own sweep/report code or call its functions interactively. Derive the crossover intensity `C/W` with matching units, and compare it to the discrete sweep.

## 3. Memory capacity

For four sequences with retained lengths 4096, 8192, 16384 and 32768, calculate logical KV using 32 layers, 8 KV heads, head dimension 128 and BF16. Add an explicitly assumed weight footprint and workspace/headroom, then assess fit under two chosen usable-memory limits. State why this does not predict actual allocator-reserved memory or MLA state.

## 4. Training state

Inspect the 1-billion-parameter, 8-rank stage-0/1/2/3 example. Reconstruct each total by hand. Compare the illustrative 12-byte optimizer/master-copy term with an 8-byte term. Explain which implementation decision could change that term and why you must inspect the framework rather than assume one universal default. Add an activation and transient-gather budget separately.

## 5. Challenge the model

Choose one prediction to falsify later with a GPU trace. Specify workload, measurement scope and the omitted effect most likely to change the result: cache reuse, nonideal bandwidth, launch latency, allocation or communication. Explain why summing all bytes or taking a global maximum may misrepresent a sequence of dependent phases.

## Submit

Use the [report template](../../report_template.md): include the sweep table, memory/sharding calculations, assumptions and three testable predictions. Label every result analytical. Run `python3 -m unittest discover -s tests -v` before submitting any changes to the calculator. Passing these tests establishes reference arithmetic, not hardware efficiency.
