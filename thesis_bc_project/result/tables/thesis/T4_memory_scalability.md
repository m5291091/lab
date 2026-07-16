# T4  Memory Scalability

| Implementation | Batch Size | Effective Batch | Sub-Batch | Number of Sub-Batches | Median Runtime (s) | Status | Failure Reason | Limitation |
|---|---|---|---|---|---|---|---|---|
| GPU_Opt_Pure | 512 | Not Recorded | Not Applicable | Not Applicable | 77.74 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt_Pure | 1024 | Not Recorded | Not Applicable | Not Applicable | 67.09 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt_Pure | 2048 | Not Recorded | Not Applicable | Not Applicable | 67.64 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt_Pure | 4096 | Not Recorded | Not Applicable | Not Applicable | 68.34 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt_Pure | 8192 | Not Recorded | Not Applicable | Not Applicable | N/A (OOM) | Out of Memory | CUDA out of memory recorded in log | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt_Pure | 10240 | Not Recorded | Not Applicable | Not Applicable | N/A (OOM) | Out of Memory | CUDA out of memory recorded in log | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt_Pure | 12288 | Not Recorded | Not Applicable | Not Applicable | N/A (OOM) | Out of Memory | CUDA out of memory recorded in log | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt_Pure | 16384 | Not Recorded | Not Applicable | Not Applicable | N/A (OOM) | Out of Memory | CUDA out of memory recorded in log | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt | 512 | 512 | 512 | 1 | 76.87 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt | 1024 | 1024 | 1024 | 1 | 66.37 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt | 2048 | 2048 | 2048 | 1 | 66.83 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt | 4096 | 4096 | 4096 | 1 | 67.65 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt | 8192 | 8192 | 6596 | 2 | 109.82 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt | 10240 | 10240 | 6596 | 2 | 324.22 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt | 12288 | 12288 | 6596 | 2 | N/A (failed) | Failed | OOM_OR_FAIL (exit 137; cause not independently confirmed) | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance; n=1 (sweep stopped) |
| GPU_Opt_Pure_Chunked | 512 | 512 | 512 | 1 | 77.87 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt_Pure_Chunked | 1024 | 1024 | 1024 | 1 | 67.09 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt_Pure_Chunked | 2048 | 2048 | 2048 | 1 | 67.62 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt_Pure_Chunked | 4096 | 4096 | 4096 | 1 | 68.30 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt_Pure_Chunked | 8192 | 8192 | 6596 | 2 | 70.65 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt_Pure_Chunked | 10240 | 10240 | 6596 | 2 | 69.14 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt_Pure_Chunked | 12288 | 12288 | 6596 | 2 | 68.55 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |
| GPU_Opt_Pure_Chunked | 16384 | 16384 | 6596 | 3 | 69.32 | Success | None | Legacy feasibility only (oldtree_f05ec52_20260512); not current block-kernel performance |

> Runtime, status, and failure reason are recomputed from the matching legacy feasibility TSVs and logs. Successful medians use n=5. GPU_Opt_Pure failures are CUDA out-of-memory errors recorded in the experiment log for all 5 trials per batch. GPU_Opt b12288 is a single failed attempt (n=1; the sweep stopped) recorded as OOM_OR_FAIL with exit 137; no CUDA OOM, host OOM-kill, or scheduler OOM record exists for it, so it is not reported as confirmed Out of Memory. Failed runs are N/A, never 0 s.
> Effective Batch, Sub-Batch, and Number of Sub-Batches come from the matching legacy experiment logs when recorded. GPU_Opt_Pure does not record those fields.
> Observed feasibility in the tested range: GPU_Opt_Pure (maximum successful requested batch 4096) < GPU_Opt (10240) < GPU_Opt_Pure_Chunked (16384). This does not imply unlimited capacity.
