# Chapter 9 Correctness and Numerical Behavior

This chapter answers RQ4: To what extent do the BC vectors produced by the proposed implementations agree with an independent reference and across different memory paths, and what numerical-representation and provenance limitations remain? It divides the correctness evidence into Tier A, which uses an independent CPU reference, and Tier B, which evaluates corrected-325557 cross-implementation consistency. PathMerge is the evaluated third-party implementation and an external comparator, not ground truth. `MismatchedElements=0` within the mixed tolerance does not mean bitwise identity.

## 9.1 Validation Criterion and Evidence Tiers

This study applies the following mixed absolute-relative tolerance to each BC element in the reference $r_i$ and candidate $c_i$:

$$
|r_i-c_i|\le abs\_tol+rel\_tol\max(|r_i|,|c_i|),
$$

where $\mathrm{abs\_tol}=10^{-3}$ and $\mathrm{rel\_tol}=10^{-6}$. The recorded quantities are vector length, missing indices, mismatched elements, maximum absolute and relative errors, NaN/Inf values, and vector SHA256. This study does not change the tolerances after the comparison to alter a decision.

**Table 9.1: Correctness evidence tiers used in this study.**

| Tier | Evidence | Graphs | Comparisons | Interpretation | Status |
|---|---|---|---:|---|---|
| A | Independent Sequential CPU reference vs GPU_Opt | benchmark_7000_41459, benchmark_11023_62184, chain_200 | 3 | Independent full-vector validation on small graphs | SUPPORTED |
| B | Cross-implementation consistency on corrected 325557 | UM, Pure, Chunked, and PathMerge vectors | 10 | Numerical consistency, not independent ground truth | SUPPORTED_WITH_LIMITATIONS |

T5 comprises 3 Tier A rows and 10 Tier B rows, for a total of 13 rows. All 13 rows have `MissingIndices=0`, `MismatchedElements=0`, `ToleranceResult=PASS`, and `ByteIdentical=No`.

## 9.2 Tier A: Independent CPU Reference

Tier A is a full-vector comparison that uses the Sequential CPU implementation as the independent reference and GPU_Opt b512 as the candidate. The comparison used 1 vector from each graph. Appendix A provides the run identifiers in its provenance records.

<!-- Source note (internal): PBS job 2367583.opbs. -->

**Table 9.2: Tier A full-vector comparisons against the independent Sequential CPU reference.**

| Graph | Vector Length | Missing Indices | Mismatched Elements | Max Absolute Error | Max Relative Error | Byte Identical | Result |
|---|---:|---:|---:|---:|---:|---|---|
| benchmark_7000_41459 | 7000 | 0 | 0 | 6.054e-09 | 4.563e-15 | No | PASS |
| benchmark_11023_62184 | 11023 | 0 | 0 | 2.980e-08 | 1.790e-14 | No | PASS |
| chain_200 | 200 | 0 | 0 | 0.000e+00 | 0.000e+00 | No | PASS |

<!-- Source note (internal): result/correctness/small_full_vector/correctness_summary.tsv and T5 Panel A. -->

All indices were present in each of the 3 comparisons, and no element exceeded the mixed tolerance. No NaN or Inf values were found. The Sequential and GPU_Opt vector SHA256 values differed, so the vectors were not byte-identical. The independent-reference evidence makes Tier A `SUPPORTED` only for these 3 small graphs. This study does not extrapolate the finding to the headline roadNet graphs, the corrected 325557 graph, or every UM/Chunked condition.

## 9.3 Tier B: Corrected 325557 Cross-Implementation Consistency

Tier B comprises 6 vectors for the corrected `325557_3216152_corrected_v1` graph. The configurations are GPU_Opt b1024/b9792, GPU_Opt_Pure b1024, GPU_Opt_Pure_Chunked b1024/b16384, and PathMerge b4096. Appendix A provides the measurement-series identifiers in its provenance records.

<!-- Source note (internal): PBS job 2404743; checkpoint 45352a3. -->

**Table 9.3: Tier B full-vector comparisons on the corrected 325557 graph.**

| Comparison Class | Reference | Candidate | Mismatched Elements | Max Absolute Error | Max Relative Error | Byte Identical | Result |
|---|---|---|---:|---:|---:|---|---|
| Same implementation, different batch | GPU_Opt b9792 | GPU_Opt b1024 | 0 | 3.052e-05 | 5.316e-14 | No | PASS |
| Same implementation, different batch | Chunked b16384 | Chunked b1024 | 0 | 3.052e-05 | 4.752e-14 | No | PASS |
| Same batch, different path | GPU_Opt b1024 | Pure b1024 | 0 | 1.907e-06 | 1.010e-14 | No | PASS |
| Same batch, different path | GPU_Opt b1024 | Chunked b1024 | 0 | 7.629e-06 | 1.080e-14 | No | PASS |
| Same batch, different path | Pure b1024 | Chunked b1024 | 0 | 7.629e-06 | 1.033e-14 | No | PASS |
| PathMerge cross (5 comparisons) | PathMerge b4096 | Five proposed vectors | 0 each | 1.222e-03 to 1.236e-03 | 5.089e-13 | No | PASS |

<!-- Source note (internal): result/correctness/corrected_325557/comparison_summary.tsv; result/correctness/corrected_325557/vector_summary.tsv; T5 Panel B. -->

In all 10 comparisons, the vector length was 325,557, the number of missing indices was 0, and the number of mismatched elements was 0. The 2 same-implementation, different-batch comparisons include the UM b9792 and Chunked b16384 large-batch/sub-batch conditions. The stress mismatches observed with the former malformed input did not recur with the corrected input.

The SHA256 values of the 6 vectors differed, and each comparison therefore had `ByteIdentical=No`. The results demonstrate numerical consistency within the mixed tolerance, not exact or bitwise identity. Floating-point values differed at some positions, but zero elements exceeded the tolerance. A byte difference is not itself a correctness failure under this criterion. The maximum BC index was 272816 for every implementation.

## 9.4 Role of PathMerge

PathMerge is an external comparator adapted from a third-party implementation of Galliot. The retained evidence does not confirm it as the original authors' official implementation, and this study does not treat it as an independent reference or as ground truth. The 5 Tier B pathmerge-cross comparisons that returned PASS show that both implementation groups agreed within the mixed tolerance on the corrected input. This evidence alone does not establish the general correctness of PathMerge or the proposed implementations.

## 9.5 Historical Malformed-Input Result

An integrity audit found that the former 325557 input was malformed. It stored 1-based identifiers as 0-based identifiers and contained 7 missing adjacency elements, out-of-range identifiers, and a missing vertex row. The internal provenance records retain the storage location and SHA256 of the former input.

In the historical canonical series on this input, the same-implementation, different-batch stress comparisons exceeded the formal tolerance, differences were also observed in the PathMerge cross comparisons, and the formal decision was `CORE_FAIL`. This study neither deletes these records nor relabels them as PASS. The archive retains them as historical invalid-input evidence together with the associated vectors, logs, and SHA256 records.

Tier B, based on revalidation of the corrected input reconstructed through a deterministic repair procedure, provides the current active conclusion. This thesis describes the history as the discovery of an input inconsistency followed by revalidation of the corrected input while retaining the former evidence, rather than as deletion of erroneous past results. It neither treats the former `CORE_FAIL` as a current failure nor applies the PASS decision for the corrected input retroactively to the former input.

<!-- Source note (internal): old input data/325557_3216152, SHA256 a095b2e7564e6c620bd0f5437917e0b28f4fecab289adf77633e850aa07da584; historical job 2368587 and result/correctness/memory_paths/canonical_job_2368587/; failure/; result/provenance/GRAPH_325557_INTEGRITY_AUDIT.md; repair tool tools/repair_325557_graph.py; corrected job 2404743. -->

## 9.6 Provenance Limitation of the Corrected Graph

The repair created the corrected graph by globally relabeling the former input from 1-based to 0-based identifiers and reconstructing the 7 elements of the missing final row based on symmetry. The repair is deterministic, and 2 independent generations produced byte-identical results. However, the retained evidence does not independently verify either the original generation seed or a complete upstream original; its status is therefore `ProvenanceStatus=internally_reconstructed_no_original_seed`.

This limitation makes Tier B `SUPPORTED_WITH_LIMITATIONS`. The determinism of the repair procedure does not resolve the missing upstream provenance.

The graph also retains structural limitations. The repair corrected the out-of-range vertex identifiers and the inconsistency in the number of CSR elements; it did not remove the 87,442 self-loops or the 866,924 duplicate ordered pairs with multiplicity 2 (Section 5.3). The 10 Tier B comparisons therefore demonstrate implementation-to-implementation consistency on this retained adjacency representation. They do not demonstrate agreement with an independent ground truth for simple-graph semantics after removing self-loops and parallel edges.

## 9.7 T5 Summary

**Table 9.4: T5 summary across both evidence tiers.**

| Panel | Evidence Tier | Rows | Missing Indices | Mismatched Elements | Tolerance Result | Byte Identical |
|---|---|---:|---:|---:|---|---|
| A | Independent CPU reference | 3 | 0 in all rows | 0 in all rows | PASS in all rows | No in all rows |
| B | Corrected-325557 cross-implementation consistency | 10 | 0 in all rows | 0 in all rows | PASS in all rows | No in all rows |

<!-- Canonical artifact (internal): result/tables/thesis/T5_correctness_summary.{md,tsv}. -->

## 9.8 Answer to RQ4

This section answers RQ4 separately for each evidence tier. Tier A is `SUPPORTED` because all full-vector comparisons against the independent Sequential CPU reference returned PASS on the 3 small graphs. Tier B is `SUPPORTED_WITH_LIMITATIONS` because all 10 comparisons among the 6 vectors for the corrected 325557 graph returned PASS within the mixed tolerance.

Tier B does not establish bitwise identity and is not validation against an independent reference or ground truth. PathMerge is an external comparator, and the corrected graph retains the provenance limitations that its original seed and complete upstream original remain unconfirmed. The archive retains the `CORE_FAIL` result for the former malformed input as historical evidence, but the current active conclusion excludes it.
