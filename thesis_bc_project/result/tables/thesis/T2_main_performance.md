# T2  Main Performance

| Graph | GPU_Opt Batch | GPU_Opt Median (s) | PathMerge Tuned Batch | PathMerge Median (s) | Speedup | GPU_Opt GTEPS | PathMerge GTEPS | Trials |
|---|---|---|---|---|---|---|---|---|
| email-EuAll | b512 | 30.81 | b2048 | 97.80 | 3.17 | 3.14 | 0.99 | 5 / 3 |
| roadNet-PA | b512 | 699.52 | b64 | 918.67 | 1.31 | 2.40 | 1.83 | 3 / 3 |
| roadNet-TX | b512 | 980.13 | b64 | 1482.68 | 1.51 | 2.71 | 1.79 | 3 / 3 |
| roadNet-CA | b512 | 2129.10 | b32 | 3079.72 | 1.45 | 2.55 | 1.77 | 3 / 3 |

> Speedup = median(PathMerge Tuned) / median(GPU_Opt). GPU_Opt fixed at b512.
> PathMerge tuned batch per graph (email b2048, roadNet-PA/TX b64, roadNet-CA b32).
> Trials are listed as GPU_Opt / PathMerge. Exact canonical source paths are listed in TABLE_MANIFEST.tsv.
