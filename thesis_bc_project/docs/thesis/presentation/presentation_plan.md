# Presentation Plan — Gate V1.2.1

## Narrative

> Results → Agenda → BC → Brandes → GPU Execution → Evaluation → Results → Limits → Contributions → Conclusion

The main result appears on Slide 1: GPU_Opt achieved 1.31–3.17× speedup on four evaluated graphs.

Presentation timing is intentionally not fixed.
The user will adjust slide selection and speaking time after rehearsal.

## Slide map

| # | Section | Title | Narrative purpose | Figure |
|---:|---|---|---|---|
| 1 | Main | Design and Evaluation of a Batch-Based GPU Execution Framework for Betweenness Centrality on GH200 | Lead with the measured result and scope | — |
| 2 | Main | Agenda | Preview the results-first narrative | — |
| 3 | Main | Betweenness Centrality Finds Important Bridge Vertices | Explain BC with an editable five-node graph | native BC graph |
| 4 | Main | Brandes' Algorithm Reduces the Cost of Exact BC | Explain how Brandes reduces exact-BC cost | — |
| 5 | Main | I Improve How Brandes Runs on the GPU | Separate Brandes from this study's GPU contribution | — |
| 6 | Main | The Proposal Is an Integrated GPU Execution Framework | Present the integrated GPU execution framework | F04 |
| 7 | Main | Source Batching Creates a Batch-Dependent Working Set | Explain the batch-dependent working set | F05 |
| 8 | Main | Three Memory-Management Variants Share One Framework | Show three memory variants of one framework | F08 |
| 9 | Main | I Separate Performance and Capacity Studies | Separate main performance from capacity tests | — |
| 10 | Main | GPU_Opt Reduced Runtime on All Four Evaluated Graphs | Report runtime reduction on four graphs | F09 |
| 11 | Main | GPU_Opt Achieved 1.31–3.17× Speedup | Report speedup over the tuned comparator | F10 |
| 12 | Main | Hybrid BFS and Dual Streams Gave the Largest Observed Effects | Report observed component effects | F12 |
| 13 | Main | Memory Variants Expanded the Tested Batch Range | Report tested memory-capacity outcomes | F13 |
| 14 | Main | The Results Have Clear Limits | State limits without correctness as a main result | — |
| 15 | Main | Contributions | Summarize four research contributions | — |
| 16 | Main | The Framework Improved Performance and Expanded the Tested Batch Range | Close with the result, capacity range, and scope | — |
| 17 | Backup | Detailed Experimental Environment | Backup: full hardware and software environment | — |
| 18 | Backup | Graph and Batch Parameters | Backup: graph, batch, and working-set parameters | — |
| 19 | Backup | PathMerge Batch-Size Sweep | Backup: PathMerge batch sweep justifying tuning | F11 |
| 20 | Backup | Forced Block-vs-Shared Kernel Comparison | Backup: forced block-vs-shared kernel comparison | F14 |
| 21 | Backup | Phase Breakdown and Profiling Scope | Backup: phase breakdown and profiling scope | F15 |
| 22 | Backup | Detailed Correctness Evidence | Backup: required correctness validation detail | — |
| 23 | Backup | Historical Record of the Malformed Input | Backup: historical malformed-input evidence | — |

## Claim boundaries

- Brandes reduces the algorithmic cost of exact BC.
- This study improves how Brandes runs on the GPU.
- This study does not change Brandes' mathematical algorithm or asymptotic complexity.
- Source batching groups sources. It does not split the graph.
- PathMerge is a third-party external comparator, not ground truth.
- Correctness is required validation, not a main research result.
- Detailed correctness and malformed-input evidence remain in Backup Slides 22–23.
- All visible slide text is English. Speaker notes contain English and Japanese.
