#!/usr/bin/env python3
"""索引TSVの commit 列を SourceSnapshotID へ変換する。
- MIGRATION_MAP.tsv: SourceCommit 列 -> SourceSnapshotID（データ行=実験snapshot, (none)行=archival_generated）
- coverage_matrix.tsv: CheckpointSHA 列 -> SourceSnapshotID
- failure/MANIFEST.tsv: Checkpoint 列 -> SourceSnapshotID
数値・SHA256・PBS job ID・他列は不変。
"""
import csv, os, sys
TBP="/work/gj17/j17000/m5291091/lab/thesis_bc_project"

FULL={
 "e32b03e9b73e9eb294685c58e488ce2a92521852":"small_correctness_20260712",
 "88faffa391026852a4440e5b9a063c08c29624f7":"phase_def_block_20260710",
 "ac2b409c25c49c41608749afba8c7081871bfe45":"memory_correctness_20260712",
 "43d1cf5542f3234dddc93c88c5fdd72761f52271":"memory_diagnostic_20260713",
 "6282798ce9942c6297cbdf2963aa7a3c65c6b807":"memory_correctness_oom_20260712",
 "29d28c50dec5e70f8d3a9a2341904e1ee94c65f3":"memory_correctness_failfast_20260712",
 "f05ec52ae657df40224f624e30f9cc78aaa3bd48":"oldtree_f05ec52_20260512"}
SHORT={"e32b03e9":"small_correctness_20260712","88faffa":"phase_def_block_20260710",
 "ac2b409":"memory_correctness_20260712","43d1cf5":"memory_diagnostic_20260713",
 "6282798":"memory_correctness_oom_20260712","29d28c50":"memory_correctness_failfast_20260712",
 "f05ec52":"oldtree_f05ec52_20260512"}
COMP={"88faffa(2026-07-10)":"phase_def_block_20260710","old-tree(f05ec52-era)":"oldtree_f05ec52_20260512",
 "f05ec52(旧tree測定)":"oldtree_f05ec52_20260512","f05ec52(旧tree)":"oldtree_f05ec52_20260512"}

def mapval(v):
    v=v.strip()
    if v in COMP: return COMP[v]
    if v in FULL: return FULL[v]
    if v in SHORT: return SHORT[v]
    return None  # unmapped

DRY="--apply" not in sys.argv
def transform(path, col, none_check_col=None):
    fp=os.path.join(TBP,path)
    rows=list(csv.reader(open(fp),delimiter='\t'))
    hdr=rows[0]
    ci=hdr.index(col)
    unmapped={}
    dist={}
    for r in rows[1:]:
        if len(r)<=ci: continue
        cur=r[ci]
        if none_check_col is not None and r[none_check_col].strip()=="(none)":
            r[ci]="archival_generated"
        else:
            m=mapval(cur)
            if m is None:
                # keep, record
                if cur.strip(): unmapped[cur]=unmapped.get(cur,0)+1
            else:
                r[ci]=m
        dist[r[ci]]=dist.get(r[ci],0)+1
    # rename header
    hdr[ci]="SourceSnapshotID"
    if not DRY:
        with open(fp,"w",newline='') as f:
            w=csv.writer(f,delimiter='\t',lineterminator='\n'); w.writerows(rows)
    print(f"\n## {path}  col={col}->SourceSnapshotID  {'(DRY)' if DRY else '(APPLIED)'}")
    for k,v in sorted(dist.items(),key=lambda x:-x[1]): print(f"   {v:4}  {k}")
    if unmapped:
        print("   UNMAPPED:")
        for k,v in unmapped.items(): print(f"     {v}  '{k}'")

transform("result/MIGRATION_MAP.tsv","SourceCommit",none_check_col=0)
transform("result/coverage_matrix.tsv","CheckpointSHA")
transform("failure/MANIFEST.tsv","Checkpoint")
