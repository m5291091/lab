#!/usr/bin/env python3
"""EXTERNAL_ARTIFACTS.tsv の役割を「Git外台帳」→「raw_data 移行対応表」へ変更する（ファイル名維持）。
- RawPath 列を追加（移行済み生データの raw_data/ 参照）
- CheckpointSHA 列 -> SourceSnapshotID（commit 非依存化）
- OriginalPath / SHA256 / PBSJobID / SizeBytes / Graph / Implementation 等の値は不変
"""
import csv, os, sys
TBP="/work/gj17/j17000/m5291091/lab/thesis_bc_project"
FP=os.path.join(TBP,"result/EXTERNAL_ARTIFACTS.tsv")

SS={"2367583.opbs":"small_correctness_20260712","2368587.opbs":"memory_correctness_20260712",
 "2369632.opbs":"memory_diagnostic_20260713","2368269.opbs":"memory_correctness_oom_20260712",
 "2368398.opbs":"memory_correctness_failfast_20260712"}
def cksum_map(v):
    v=v.strip()
    m={"88faffa":"phase_def_block_20260710","88faffa(2026-07-10)":"phase_def_block_20260710",
       "e32b03e9b73e9eb294685c58e488ce2a92521852":"small_correctness_20260712",
       "ac2b409c25c49c41608749afba8c7081871bfe45":"memory_correctness_20260712",
       "43d1cf5542f3234dddc93c88c5fdd72761f52271":"memory_diagnostic_20260713",
       "6282798ce9942c6297cbdf2963aa7a3c65c6b807":"memory_correctness_oom_20260712",
       "29d28c50dec5e70f8d3a9a2341904e1ee94c65f3":"memory_correctness_failfast_20260712",
       "phaseB":"phase_def_block_20260710","各job":"per_row_PBSJobID"}
    return m.get(v, v)

# OriginalPath -> RawPath（移行済み 23 ファイル）
RAW={
 "build_miyabi/result_small_correctness_20260712_181140_2367583.opbs/benchmark_7000_41459/sequential.bc.tsv":"raw_data/correctness/small_full_vector/benchmark_7000_41459/sequential/seq/job_2367583_20260712/sequential.bc.tsv",
 "build_miyabi/result_small_correctness_20260712_181140_2367583.opbs/benchmark_7000_41459/gpu_opt.bc.tsv":"raw_data/correctness/small_full_vector/benchmark_7000_41459/gpu_opt/um_b512/job_2367583_20260712/gpu_opt.bc.tsv",
 "build_miyabi/result_small_correctness_20260712_181140_2367583.opbs/benchmark_11023_62184/sequential.bc.tsv":"raw_data/correctness/small_full_vector/benchmark_11023_62184/sequential/seq/job_2367583_20260712/sequential.bc.tsv",
 "build_miyabi/result_small_correctness_20260712_181140_2367583.opbs/benchmark_11023_62184/gpu_opt.bc.tsv":"raw_data/correctness/small_full_vector/benchmark_11023_62184/gpu_opt/um_b512/job_2367583_20260712/gpu_opt.bc.tsv",
 "build_miyabi/result_small_correctness_20260712_181140_2367583.opbs/chain_200/sequential.bc.tsv":"raw_data/correctness/small_full_vector/chain_200/sequential/seq/job_2367583_20260712/sequential.bc.tsv",
 "build_miyabi/result_small_correctness_20260712_181140_2367583.opbs/chain_200/gpu_opt.bc.tsv":"raw_data/correctness/small_full_vector/chain_200/gpu_opt/um_b512/job_2367583_20260712/gpu_opt.bc.tsv",
 "build_miyabi/result_memory_correctness_20260712_220331_2368587.opbs/gpu_opt_b1024.bc.tsv":"raw_data/correctness/memory_paths/325557_3216152/gpu_opt/um_b1024/job_2368587_20260712/gpu_opt_b1024.bc.tsv",
 "build_miyabi/result_memory_correctness_20260712_220331_2368587.opbs/gpu_opt_b9792.bc.tsv":"raw_data/correctness/memory_paths/325557_3216152/gpu_opt/um_b9792/job_2368587_20260712/gpu_opt_b9792.bc.tsv",
 "build_miyabi/result_memory_correctness_20260712_220331_2368587.opbs/gpu_opt_pure_b1024.bc.tsv":"raw_data/correctness/memory_paths/325557_3216152/gpu_opt_pure/pure_b1024/job_2368587_20260712/gpu_opt_pure_b1024.bc.tsv",
 "build_miyabi/result_memory_correctness_20260712_220331_2368587.opbs/gpu_opt_pure_chunked_b1024.bc.tsv":"raw_data/correctness/memory_paths/325557_3216152/gpu_opt_pure_chunked/chunked_b1024/job_2368587_20260712/gpu_opt_pure_chunked_b1024.bc.tsv",
 "build_miyabi/result_memory_correctness_20260712_220331_2368587.opbs/gpu_opt_pure_chunked_b16384.bc.tsv":"raw_data/correctness/memory_paths/325557_3216152/gpu_opt_pure_chunked/chunked_b16384/job_2368587_20260712/gpu_opt_pure_chunked_b16384.bc.tsv",
 "build_miyabi/result_memory_correctness_20260712_220331_2368587.opbs/pathmerge_b4096.bc.tsv":"raw_data/correctness/memory_paths/325557_3216152/pathmerge_bc/pathmerge_b4096/job_2368587_20260712/pathmerge_b4096.bc.tsv",
 "build_miyabi/result_memory_diagnostic_20260713_012328_2369632.opbs/CONTROL/vector.bc.tsv":"raw_data/correctness/memory_paths/325557_3216152/gpu_opt/um_b1024_control/job_2369632_20260713/vector.bc.tsv",
 "build_miyabi/result_memory_diagnostic_20260713_012328_2369632.opbs/T-RESET/vector.bc.tsv":"raw_data/correctness/memory_paths/325557_3216152/gpu_opt/um_b1024_treset/job_2369632_20260713/vector.bc.tsv",
 "build_miyabi/result_memory_diagnostic_20260713_012328_2369632.opbs/T-NSEFF/vector.bc.tsv":"raw_data/correctness/memory_paths/325557_3216152/gpu_opt/um_b1024_tnseff/job_2369632_20260713/vector.bc.tsv",
 "build_miyabi/result_memory_correctness_20260712_211738_2368398.opbs/pathmerge_b4096.bc.tsv":"raw_data/unsuccessful/early_terminated/memory_paths/325557_3216152/pathmerge_bc/pathmerge_b4096/job_2368398_20260712/pathmerge_b4096.bc.tsv",
 "build_miyabi/result_memory_correctness_20260712_211738_2368398.opbs/gpu_opt_pure_b1024.bc.tsv":"raw_data/unsuccessful/early_terminated/memory_paths/325557_3216152/gpu_opt_pure/pure_b1024/job_2368398_20260712/gpu_opt_pure_b1024.bc.tsv",
 "build_miyabi/result_memory_correctness_20260712_204001_2368269.opbs/pathmerge_b4096.bc.tsv":"raw_data/unsuccessful/oom/memory_paths/325557_3216152/pathmerge_bc/pathmerge_b4096/job_2368269_20260712/pathmerge_b4096.bc.tsv",
 "build_miyabi/result_memory_correctness_20260712_204001_2368269.opbs/gpu_opt_b10240.bc.tsv":"raw_data/unsuccessful/oom/memory_paths/325557_3216152/gpu_opt/um_b10240/job_2368269_20260712/gpu_opt_b10240.bc.tsv",
 "thesis_bc_project/bc_mem_correct.o2368269":"raw_data/unsuccessful/oom/memory_paths/325557_3216152/pbs_stdout_job_2368269_20260712.log",
 "thesis_bc_project/bc_mem_correct.o2368398":"raw_data/unsuccessful/early_terminated/memory_paths/325557_3216152/pbs_stdout_job_2368398_20260712.log",
 "thesis_bc_project/bc_mem_correct.o2368587":"raw_data/correctness/memory_paths/325557_3216152/pbs_stdout_job_2368587_20260712.log",
 "thesis_bc_project/bc_mem_diag.o2369632":"raw_data/correctness/memory_paths/325557_3216152/pbs_stdout_job_2369632_20260713.log",
 # directory rows -> raw_data parent area (job directory-level)
 "build_miyabi/result_memory_correctness_20260712_204001_2368269.opbs/":"raw_data/unsuccessful/oom/memory_paths/325557_3216152/ (+ failure/failed/oom/memory_correctness_2368269/ 要約)",
 "build_miyabi/result_memory_correctness_20260712_211738_2368398.opbs/":"raw_data/unsuccessful/early_terminated/memory_paths/325557_3216152/ (+ failure/early_terminated/memory_correctness_2368398/ 要約)",
 "build_miyabi/result_memory_correctness_20260712_220331_2368587.opbs/":"raw_data/correctness/memory_paths/325557_3216152/ (+ result/correctness/memory_paths/canonical_job_2368587/ 要約)",
 "build_miyabi/result_memory_diagnostic_20260713_012328_2369632.opbs/":"raw_data/correctness/memory_paths/325557_3216152/ (+ result/correctness/memory_paths/diagnostic_job_2369632/ 要約)",
}
def rawpath_for(op, rtype, retention):
    op=op.strip()
    if op in RAW: return RAW[op]
    if "sqlite" in op: return "not_migrated(regeneratable_from_nsys-rep)"
    if "phaseB_verify_out" in op: return "not_migrated(excluded_large_BC_vectors)"
    if op.startswith("thesis_bc_project/*.o") or "result_0709" in op: return "not_migrated(PBS_o_files; memory jobs のみ raw_data へ)"
    if op=="build_miyabi/result_final_tables/final_speedup_tables.md": return "failure/superseded_success/final_speedup_tables_OLD.md (curated)"
    # 88faffa-era experiment dirs: curated outputs in result/, no large raw vectors
    if op.startswith("build_miyabi/result_"): return "not_migrated(curated_in_result/; 大容量 raw なし)"
    return ""

DRY="--apply" not in sys.argv
rows=list(csv.reader(open(FP),delimiter='\t'))
hdr=rows[0]
# columns
i_op=hdr.index("OriginalPath"); i_ck=hdr.index("CheckpointSHA"); i_type=hdr.index("Type"); i_ret=hdr.index("RetentionStatus")
new_hdr=hdr[:]
new_hdr[i_ck]="SourceSnapshotID"
new_hdr.insert(i_op+1,"RawPath")
out=[new_hdr]
for r in rows[1:]:
    r=r[:]
    # transform checkpoint by row: prefer PBSJobID-based mapping for memory/small jobs
    job=r[hdr.index("PBSJobID")].strip()
    ck=r[i_ck].strip()
    snap = SS.get(job) or cksum_map(ck)
    r[i_ck]=snap
    rp=rawpath_for(r[i_op], r[i_type], r[i_ret])
    r.insert(i_op+1, rp)
    out.append(r)

if not DRY:
    with open(FP,"w",newline='') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n'); w.writerows(out)
print(f"MODE={'DRY' if DRY else 'APPLY'} rows={len(out)-1}")
print("new header:", new_hdr)
print("\nRawPath assignment sample:")
for r in out[1:]:
    print(f"   snap={r[i_ck]:34} raw={r[i_op+1][:70]}")
