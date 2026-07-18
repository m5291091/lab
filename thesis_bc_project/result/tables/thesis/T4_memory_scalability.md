# T4  Memory Feasibility Boundary Validation (corrected 325557)

| Implementation | Requested Batch | Observed Outcome | Failure Class | Runtime (s) | Runner Exit | OOM Evidence | Note |
|---|---|---|---|---|---|---|---|
| GPU_Opt_Pure | 4096 | Success | None | 65.89 | 0 | none | Feasibility run (n=1) |
| GPU_Opt_Pure | 8192 | CUDA out-of-memory | CUDA OOM | N/A (CUDA OOM) | 1 | cuda_oom (host_pure.cu:144: out of memory) | Confirmed CUDA out-of-memory (host_pure.cu:144) |
| GPU_Opt | 10240 | Success | None | 238.67 | 0 | none | Feasibility run (n=1); UM oversubscription spill over NVLink-C2C |
| GPU_Opt | 12288 | Cgroup host-memory OOM kill | Cgroup host-memory OOM kill | N/A (cgroup host-memory OOM kill) | 137 | none (SIGKILL, exit 137) | Host/cgroup memory limit exceeded; not a CUDA or HBM out-of-memory |
| GPU_Opt_Pure_Chunked | 16384 | Success | None | 66.60 | 0 | none | Tested upper limit (no unlimited-capacity claim) |

> Targeted feasibility-boundary validation on the corrected 325557 graph (job 2404743, checkpoint 45352a3); each configuration n=1. This confirms feasibility ordering, not performance. Runtimes are single-run wall-clock values at different requested batches and are not a performance comparison. Failures are shown as N/A, never 0 s.
> Two failure classes are kept distinct: GPU_Opt_Pure b8192 is a confirmed CUDA (GPU-device) out-of-memory (runner exit 1, host_pure.cu:144: out of memory); GPU_Opt b12288 is a host/cgroup memory OOM kill (SIGKILL, exit 137) with CUDA-level oom_evidence=none, so it is NOT a CUDA or HBM out-of-memory.
> Observed feasible ordering within the tested range only: GPU_Opt_Pure (maximum successful requested batch 4096) < GPU_Opt (10240) < GPU_Opt_Pure_Chunked (16384). Chunked was tested to 16384; this is no unlimited-capacity claim. The input file is about 43.25 MiB; capacity pressure is the batch-dependent working set, not the input graph. Corrected 325557 only; not generalized to other graphs or GPUs.
