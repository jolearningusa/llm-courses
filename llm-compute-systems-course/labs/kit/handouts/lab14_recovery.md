# L14 — Recover the computation, not only the weights

**Released:** actual CPU file save/load, torch 2.9.1, six deterministic synthetic steps with dropout and AdamW. Distributed durable checkpoints and real process-kill recovery remain unverified. Estimated 4–6 hours, student pilot pending.

From `labs/kit/` run `python3 recovery_lab.py > recovery-cpu.json` and the unit suite. The reference trains uninterrupted, repeats from the same seed, saves after step three, constructs fresh model/optimizer objects, restores model/optimizer/torch RNG/data cursor and completes. Exact CPU loss sequence and final parameter equality are required for this fixture. The test omits RNG restoration once and requires divergence, proving the check can detect incomplete state. Only load your own trusted local checkpoint; model/tokenizer/config identity must accompany real deployments.

Student task: add a scheduler and accumulation-position state, then interrupt at a declared safe boundary. Explain why a mid-accumulation checkpoint needs more than an optimizer step counter. The provided fixture uses fixed pre-generated CPU data, no worker queues and no CUDA RNG. List additional state for your target. Temporary-file rename protects against partial local visibility; it makes no fsync, object-store or distributed atomicity guarantee.

Reproduce the lecture interval model C/I + λI/2 with explicit seconds and assumptions. Report useful tokens over total elapsed time, including save stalls, repeated work and recovery. For a real distributed extension, test incomplete saves, completion publication, restored optimizer shards and data coverage; validate layout resharding only where the selected DCP release supports it. No speed or durability result follows from the tiny CPU example.

Submit raw JSON, the state ledger and [report](../../report_template.md). Instructor diagnostic: identical next loss is weaker evidence than the full resumed sequence plus parameters/data coverage; a new dropout RNG stream can expose a weights-only restore.
