# L5 — Online attention and matched backend comparison

**Released/validated:** CPU single-head, square self-attention in the pinned PyTorch environment; output and gradient comparisons. **Unverified extension:** fused GPU/backend performance and memory behavior. Estimated student time: 5–8 hours, pending pilot.

## A. Derive online normalization

Reproduce the lecture's scores [0, ln 2], values [2,5] calculation. Write the update for running maximum, denominator and unnormalized weighted sum. Explain why both old denominator and weighted sum must be rescaled after a maximum change. Then derive the state needed for a multi-dimensional value vector.

## B. Run and inspect the references

```sh
python3 attention_lab.py --length 33 > attention-cpu.json
python3 -m unittest discover -s tests -v
```

Read `full_attention` and `tiled_attention`. The contract is CPU float32/64, one head, matching nonempty query/key lengths and standard causal or noncausal self-attention. Decode offsets, GQA/MLA, arbitrary masks and all-masked rows are not implemented. Vary tile sizes and sequence tails, compare outputs and gradients, and verify that future values cannot affect earlier causal outputs.

## C. Distinguish algorithm and implementation

The Python reference demonstrates the online state update, not a fused FlashAttention kernel. Its autograd graph may retain per-tile intermediates; do not claim training-memory savings from its forward storage argument. The timing demo uses `no_grad` and includes Python dispatch/checksum. Dense permitted-pair arithmetic remains quadratic even though a fused implementation can avoid materializing the whole score matrix in HBM.

## D. GPU/backend extension

On supported hardware select a framework SDPA, cuDNN, FlashAttention or FlashInfer path. Record actual backend, version, dtype, mask and shape support. Validate the same contract before timing. Compare short/long lengths and uniform/ragged workloads, then inspect traffic, planning and stage balance. For training, include backward and saved/recomputed state. A causal mask or precision change must not be hidden inside a speed comparison.

Without hardware, complete A–C plus a paper evaluation critique: identify device, library baseline, mask, dtype and timing scope. Distinguish an older published result from your current unmeasured target. Submit the [report](../../report_template.md), raw CPU results and explicit extension limits.
