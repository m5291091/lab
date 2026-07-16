# T6  Experimental Environment

| Component | Specification |
|---|---|
| GPU | NVIDIA GH200 |
| Nominal HBM3 | 96 GB |
| Recorded Device Memory | 97,871 MiB (approximately 95.6 GiB or 102.6 GB) |
| Runtime-Reported Total Memory at Launch | approximately 102.0 GB (decimal GB) |
| Runtime Free Memory at Launch | approximately 101.4 GB (decimal GB; memory-budget basis, not total capacity) |
| NVIDIA Driver | 595.58.03 |
| CUDA Toolkit (nvcc) | release 13.0, V13.0.48 |
| Host C++ Compiler | g++ (GCC) 11.4.1 |
| CMake | 4.3.4 |
| Nsight Systems (nsys) | 2025.5.1.121 |
| PBS System | Miyabi-G PBS batch system |
| Group | gj17 |
| Queue | Not independently verifiable from retained job logs |
| Resource Configuration - Memory-Path Experiments | Host-memory-limited 100 GiB configuration |
| HBM3 Bandwidth (Device-to-Device) | 1818.6 GB/s (45.2% of theoretical) |
| Pinned Host-to-Device Bandwidth | 424.1 GB/s (47.1% of theoretical) |
| Pinned Device-to-Host Bandwidth | 297.6 GB/s (33.1% of theoretical) |
| NVLink-C2C Prefetch Bandwidth | 177.7 GB/s (19.7% of theoretical) |
| Main-Experiment Aggregation | Median of all recorded trials |
| Main-Experiment Warmup | None; no recorded trial was discarded |

> GPU model, nominal HBM3, recorded device memory, software, PBS system, and group from result/environment/environment.md.
> The nominal 96 GB, recorded 97,871 MiB, and runtime-reported approximately 102.0 GB refer to the same HBM3 through different units or query methods, not separate memory tiers.
> Runtime total and free memory at launch from raw_data/main_performance/proposed_variants/email-EuAll/_run/job_2357334_20260711/phase_timing.log; free memory is the launch-time available amount used as the memory-budget basis, not total capacity.
> The retained job logs do not independently verify the actual queue name; it is not an evaluation control variable.
> Bandwidth from raw_data/profiling/job_2359175_20260711/bandwidth.log.
