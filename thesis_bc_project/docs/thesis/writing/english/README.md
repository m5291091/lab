# English Translation Guide

## Translation Provenance

- Japanese source: `thesis_bc_project/docs/thesis/writing/japanese/05_experimental_methodology.md`
- Japanese source commit: `2dbb0b58643e8d811d8543c8cee847488c2ad7fd`
- Canonical title: Design and Evaluation of a Batch-Based GPU Execution Framework for Betweenness Centrality on GH200
- Translation date: 2026-07-19
- Current Gate: Gate T2A — English Translation Foundation and Chapter 5
- Each English file must preserve the meaning, numerical values, and scope of its Japanese source.

## Language and Style

- Use American English and formal academic prose.
- Use clear active voice where appropriate and avoid unnecessary passive voice.
- Maintain a neutral single-author style. Prefer “this thesis” or “this study.”
- Do not use unexplained “we,” “our,” or “I.”
- Translate meaning rather than Japanese word order.
- Avoid overly long sentences. Use one main claim per sentence where practical, preferably keeping prose sentences near 25–35 words.
- Preserve the logical order of the source. Do not summarize, restructure, or introduce new research claims.

## Claim Discipline

Do not use the following expressions unless the source explicitly establishes the necessary scope and evidence: “always,” “all graphs,” “universally,” “state of the art,” “fastest,” “optimal,” “proves,” “guarantees,” “completely eliminates,” or “statistically significant.”

Do not describe PathMerge as the “official PathMerge implementation” or any result as “ground truth.” Do not use “unlimited” or “out-of-core,” because this thesis does not define, implement, or evaluate those claims.

Preferred evidence-sensitive wording includes:

- “on the evaluated graphs”
- “under the evaluated configuration”
- “the observed result”
- “the retained evidence”
- “the tested upper bound”
- “supports”
- “is consistent with”
- “was not independently determined”
- “should not be generalized to”
- “a third-party implementation”
- “an external comparator”

## Technical Formatting

- Enclose code identifiers in backticks or preserve their existing formatting.
- Preserve these identifiers exactly: `GPU_Opt`, `GPU_Opt_Pure`, `GPU_Opt_Pure_Chunked`, `PathMerge`, `NS_eff`, `SUB_BATCH`, `b512`, `abs_tol`, and `rel_tol`.
- Do not alter mathematical symbols or citation keys.
- Do not change the number or order of columns in Markdown tables.
- Keep HTML Source notes separate from reader-facing prose.

## Canonical English Research Questions

RQ1: On the four evaluated graphs, is the block-based GPU_Opt implementation with a fixed batch size of 512 faster than the graph-wise tuned third-party PathMerge implementation?

RQ2: To what extent do Hybrid BFS, Warp-Cooperative Accumulation, Dual-Stream Execution, and the Block Kernel contribute to the observed performance?

RQ3: On the evaluated corrected 325557 graph, how do the memory-management approaches of GPU_Opt, GPU_Opt_Pure, and GPU_Opt_Pure_Chunked affect the feasible batch size and the observed memory constraints?

RQ4: To what extent do the BC vectors produced by the proposed implementations agree with an independent reference and across different memory paths, and what numerical-representation and provenance limitations remain?
