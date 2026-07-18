# T1  Graph Metadata

| Graph | Nodes | Edges | Average Degree | Input File [MiB] | CSR [MiB] | Used For |
|---|---|---|---|---|---|---|
| email-EuAll | 265009 | 364481 | 2.751 | 5.59 | 3.79 | Main performance (RQ1); Ablation |
| roadNet-PA | 1088092 | 1541898 | 2.834 | 28.43 | 15.91 | Main performance (RQ1); Kernel selection |
| roadNet-TX | 1379917 | 1921660 | 2.785 | 36.53 | 19.93 | Main performance (RQ1); Kernel selection |
| roadNet-CA | 1965206 | 2766607 | 2.816 | 53.83 | 28.60 | Main performance (RQ1) |
| 325557_3216152_corrected_v1 | 325557 | 3216152 | 19.758 | 43.25 | 25.78 | Ablation; Memory scalability; Correctness |
| 56438_300801 | 56438 | 300801 | 10.660 | 3.72 | 2.51 | Ablation |
| benchmark_7000 | 7000 | 41459 | 11.845 | 0.39 | 0.34 | Ablation; Correctness |
| benchmark_11023 | 11023 | 62184 | 11.283 | 0.61 | 0.52 | Ablation; Correctness |
| benchmark_85830 | 85830 | 241080 | 5.618 | 3.18 | 2.17 | Auxiliary |
| chain_200 | 200 | 199 | 1.990 | 0.00 | 0.00 | Correctness |
| random | 32212 | 101805 | 6.321 | 1.30 | 0.90 | Auxiliary |
| 325557_3216152 | 325557 | 3216152 | 19.758 | 43.25 | 25.78 | Historical (superseded by corrected_v1) |

> Nodes / Edges (undirected edge count m) / Average Degree and file sizes from result/datasets/graph_catalog.tsv. Input File [MiB] is the on-disk CSR text input file (FileSizeBytes / 1,048,576); CSR [MiB] is the in-memory CSR array ((n + 1) + 2m) x 4 bytes / 1,048,576.
> Input File [MiB] and CSR [MiB] are the STATIC graph representation on disk / in host memory; they are NOT the GPU working set. The GPU working set is the batch-dependent per-source state (Chapter 8), not the graph file size.
> The corrected 325557 graph (325557_3216152_corrected_v1, SHA256 8373244f..., checkpoint 45352a3) is used for Ablation / Memory scalability / Correctness only, NOT for the RQ1 main-performance comparison. The old malformed 325557_3216152 (SHA256 a095b2e7...) is retained only as a historical, superseded input.
