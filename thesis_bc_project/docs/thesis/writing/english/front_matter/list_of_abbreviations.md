# List of Abbreviations

This list includes only those abbreviations that are used repeatedly in the text of Chapters 1 through 11 and the Abstract, and which the reader may need to reference. Abbreviations that are not used in the text, or that appear only once and are fully explained at their first occurrence, are excluded. `Full Term` is the formal English name, and `Description` is a brief explanation.

| Abbreviation | Full Term | Description |
|:--|:--|:--|
| API | Application Programming Interface | A set of calling conventions provided by a library or execution environment. Used to refer to cuGraph and CUDA runtime specifications. |
| BC | Betweenness Centrality | A centrality metric indicating the degree to which a vertex appears on shortest paths between other vertex pairs. The target of this study. |
| BFS | Breadth-First Search | A graph traversal algorithm. Used in the Forward phase of the Brandes algorithm to compute shortest distances and path counts. |
| CPU | Central Processing Unit | Used to refer to the Grace CPU of the GH200 and host-side processing. |
| CSR | Compressed Sparse Row | A sparse graph representation using a row pointer and an adjacency array. The input and internal representation of this study. |
| CUDA | Compute Unified Device Architecture | A parallel computing platform and programming model for NVIDIA GPUs. |
| DAG | Directed Acyclic Graph | Used to refer to the shortest-path DAG rooted at a given source. |
| GPU | Graphics Processing Unit | Used to refer to the Hopper GPU of the GH200, which executes the BC computation. |
| GTEPS | Giga Traversed Edges Per Second | A throughput metric defined as $10^{9}$ traversed edges per second. |
| HBM3 | High Bandwidth Memory 3 | The on-package high-bandwidth memory of the Hopper GPU. The nominal capacity of the targeted configuration is 96 GB. |
| LPDDR5X | Low-Power Double Data Rate 5X | The low-power memory standard for the Grace CPU side. |
| NVLink-C2C | NVLink Chip-to-Chip | The coherent interconnect linking the Grace CPU and the Hopper GPU. |
| OOM | Out of Memory | Allocation failure or process termination due to insufficient memory. This study distinguishes between CUDA device-memory OOM and cgroup host-memory OOM kill. |
| PBS | Portable Batch System | The job scheduler of Miyabi-G. Used for submitting experimental jobs. |
| RQ | Research Question | A question defining the evaluation scope of this study. RQ1 through RQ4 are established. |
| RSS | Resident Set Size | The amount of physical memory occupied by a process. Mentioned as an unmeasured metric in this study. |
| SD | Standard Deviation | Mentioned as the sample standard deviation $s_T$ of the runtime (unbiased estimator, ddof=1), reported as an auxiliary value. |
| SHA256 | Secure Hash Algorithm 256-bit | A hash value used to verify the identity of the input graph and BC vector. |
| SIMT | Single Instruction, Multiple Threads | The GPU execution model where multiple threads in a warp advance the same instruction stream. |
| TSV | Tab-Separated Values | The format used for saving raw data and derived tables. |
| UM | Unified Memory | A managed allocation mechanism accessible from both the CPU and GPU via the same address space. It does not provide additional physical capacity. |

## Implementation Names

The following names are implementation and method names used in this study, not abbreviations. Therefore, they are not included in the abbreviation list above.

| Name | Description |
|:--|:--|
| GPU_Opt | The main implementation of the proposed batch-based GPU execution framework. It uses Unified Memory. |
| GPU_Opt_Pure | A memory-management variant of the same framework. It explicitly uses device-only memory. |
| GPU_Opt_Pure_Chunked | A memory-management variant of the same framework. It limits the simultaneously resident working set through source sub-batching. |
| PathMerge | A third-party implementation evaluated as a comparator. It is not the original authors' official implementation; it is an external comparator and not ground truth. |
| Pure | An abbreviated notation for GPU_Opt_Pure used in the text. |
| Chunked | An abbreviated notation for GPU_Opt_Pure_Chunked used in the text. |

GPU_Opt, GPU_Opt_Pure, and GPU_Opt_Pure_Chunked are not three independent proposals, but memory-management variants within a common GPU execution framework.

## Abbreviations Defined Only at First Use

The following abbreviations are used only once in the text, where they are fully explained with their formal names. Therefore, they are excluded from this list.

| Abbreviation | Full Term | First Occurrence |
|:--|:--|:--|
| SNAP | Stanford Network Analysis Project | Section 5.3 |
| RAPIDS Memory Manager | — | Section 5.4, Table 5.4 (The abbreviation `RMM` is not used) |

`H2D` and `D2H` are not used as abbreviations, as they were changed to plain text (host-to-device and device-to-host) in Section 7.6.
