#!/usr/bin/env python3
"""MIGRATION_MAP.tsv の NewSHA256 を検証し、編集した metadata doc のみ再計算する。
- 生証跡(PRESERVE)コピーの SHA が変化していないことを安全確認（変化があれば異常終了）。
- raw-data 計測 SHA256 (OriginalSHA256 / 他索引) には触れない。
"""
import csv, hashlib, os, sys
TBP="/work/gj17/j17000/m5291091/lab/thesis_bc_project"
FP=os.path.join(TBP,"result/MIGRATION_MAP.tsv")

PRESERVE_PREFIX=[
 "result/correctness/memory_paths/canonical_job_2368587/",
 "result/correctness/memory_paths/diagnostic_job_2369632/",
 "failure/failed/oom/memory_correctness_2368269/",
 "failure/early_terminated/memory_correctness_2368398/",
 "result/profiling/ablation_",
 "failure/early_terminated/pathmerge_sweep_",
 "failure/superseded_success/",
 "failure/incomplete/",
 "legacy_results_miyabi/","results_miyabi/",
]
PRESERVE_EXACT=set([
 "result/correctness/small_full_vector/MANIFEST.txt","result/correctness/small_full_vector/run.log",
 "result/correctness/small_full_vector/correctness_summary.tsv"])
for g in ["benchmark_7000_41459","benchmark_11023_62184","chain_200"]:
    PRESERVE_EXACT.add(f"result/correctness/small_full_vector/{g}/comparison.md")
    PRESERVE_EXACT.add(f"result/correctness/small_full_vector/{g}/gpu_opt.stderr.log")
    PRESERVE_EXACT.add(f"result/correctness/small_full_vector/{g}/sequential.stderr.log")

def is_preserve(rel):
    return any(rel.startswith(p) for p in PRESERVE_PREFIX) or rel in PRESERVE_EXACT

def sha256(fp):
    h=hashlib.sha256()
    with open(fp,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()

APPLY="--apply" in sys.argv
rows=list(csv.reader(open(FP),delimiter='\t'))
hdr=rows[0]; iN=hdr.index("NewPath"); iS=hdr.index("NewSHA256")
changed=[]; preserve_drift=[]; recompute=[]
for r in rows[1:]:
    if len(r)<=iS: continue
    npath=r[iN].strip(); rec=r[iS].strip()
    if len(rec)!=64 or not all(c in "0123456789abcdef" for c in rec): continue
    fp=os.path.join(TBP,npath)
    if not os.path.isfile(fp): continue
    cur=sha256(fp)
    if cur==rec: continue
    # SHA differs
    if is_preserve(npath):
        preserve_drift.append((npath,rec,cur))
    else:
        recompute.append((npath,rec,cur))
        if APPLY: r[iS]=cur

print(f"NewSHA256 rows differing: preserve_drift={len(preserve_drift)} recompute(edited docs)={len(recompute)}")
if preserve_drift:
    print("!!! PRESERVE FILE SHA DRIFT (raw evidence changed - INVESTIGATE):")
    for p,a,b in preserve_drift: print(f"   {p}\n     rec={a}\n     cur={b}")
print("\nedited-doc NewSHA256 to recompute (sample):")
for p,a,b in recompute[:12]: print(f"   {p}")
print(f"   ...total {len(recompute)}")
if APPLY and not preserve_drift:
    with open(FP,"w",newline='') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n'); w.writerows(rows)
    print("\nAPPLIED NewSHA256 recompute for edited docs.")
elif APPLY and preserve_drift:
    print("\nABORTED apply due to preserve drift.")
    sys.exit(1)
