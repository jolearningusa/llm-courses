# Course implementation roadmap

This research/outline installment is deliberately reviewable before full slides and lab kits are built. The repository requires one intent per PR and recommends outline → modules → labs for a large course. Complete and merge each installment before starting another branch.

## Prepared in this installment

- Research synthesis, annotated papers/tutorials/repositories and review-depth limitations.
- Sixteen-week training/inference systems syllabus and L0–L15 lab specifications.
- Workload/algorithm primer, PyTorch/compiler/deployment guide, Rubin case study, NVIDIA and AMD stack maps, and AgentOS caching comparison.
- Shared experiment/report contract, hardware tiers and numerical/performance evaluation policy.

## Follow-on installments

1. **Measurement foundation:** opening lecture deck, resource calculator, workload fixtures, environment manifests and L0–L2 executable kit. Pilot CPU and one-GPU paths.
2. **Kernel lectures and labs:** CUDA execution/reductions, GEMM, attention and precision in bounded PRs. Include correctness, sanitizer and numerical checks on declared devices.
3. **Training systems:** DDP/FSDP2 and parallelism lectures/kits, MoE communication, checkpoint/recovery. Record multi-GPU evidence and provide trace-only alternatives.
4. **Inference systems:** request scheduler/KV anatomy, vLLM/llama.cpp/SGLang cache experiments, chunked prefill/disaggregation, speculation and agent waits.
5. **Deployment:** AOTInductor/ExecuTorch kit with exact producer/loader/runtime contracts and tested compatibility.
6. **Architecture integration:** Rubin hardware/software lecture and AMD portability comparison with supported paths, source dates and measured/simulated boundaries.
7. **Pilot/capstone:** calibrate workload sizes and timing, publish reproducibility/quality criteria and release the capstone package.

## Gates before calling a module ready to teach

Its assigned paper design/evaluation sections have been read, relevant upstream implementation inspected at a recorded commit, deck links/numbers checked, and lab instructions timed by a pilot. A lab must run on at least one declared configuration or be explicitly released as an analytical/trace exercise. Add tracked API dependencies to `references/tracked_sources.json` when configurations are pinned, then use the repository freshness checks.

Before launching the course, confirm access to a CUDA GPU and a small multi-GPU allocation; AMD execution requires its own supported environment. Exact model/weight/data selections and compatible environments remain pilot decisions. No Rubin purchase or allocation is required for the analytical case study.
