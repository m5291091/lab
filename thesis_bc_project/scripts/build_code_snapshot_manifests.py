#!/usr/bin/env python3
"""code_snapshots/ の各スナップショットに SOURCE_MANIFEST.tsv / SHA256SUMS / BUILD_ENV.md を生成し、
トップレベルに README.md と _legacy_audit/LEGACY_COMMIT_TO_SNAPSHOT.tsv（raw 取得・再生成には不要な旧履歴対応表）を書く。

git archive で抽出したファイル内容の SHA256 を、対応する commit の git blob 内容と突き合わせて
整合性検証する（推定ではなく実照合）。
"""
import csv, hashlib, os, subprocess

REPO = "/work/gj17/j17000/m5291091/lab"
TBP = os.path.join(REPO, "thesis_bc_project")
CS = os.path.join(TBP, "code_snapshots")

# id -> (commit_full, commit_short, date, treeroot, covers, jobids)
SNAP = {
 "small_correctness_20260712": (
   "e32b03e9b73e9eb294685c58e488ce2a92521852","e32b03e9","2026-07-12","thesis_bc_project",
   "correctness/small_full_vector（小グラフ独立参照 full-vector 正確性）","2367583.opbs"),
 "phase_def_block_20260710": (
   "88faffa391026852a4440e5b9a063c08c29624f7","88faffa","2026-07-11","thesis_bc_project",
   "main_performance/proposed_variants; tuning/pathmerge; tuning/kernel_selection; ablation; profiling; phase_breakdown; correctness/pathmerge_tuned（実験実施は 2026-07-10〜11）",
   "2356120;2357334-2357337;2355000;2355001;2359080;2359081;2359096;2359169;2360072;2360073;2361040;2361041;2362006;2354329;2354330;2354994;2354999;2359175"),
 "memory_correctness_20260712": (
   "ac2b409c25c49c41608749afba8c7081871bfe45","ac2b409","2026-07-12","thesis_bc_project",
   "correctness/memory_paths（canonical memory-path comparison matrix）","2368587.opbs"),
 "memory_diagnostic_20260713": (
   "43d1cf5542f3234dddc93c88c5fdd72761f52271","43d1cf5","2026-07-13","thesis_bc_project",
   "correctness/memory_paths（T-RESET/T-NSEFF 診断）","2369632.opbs"),
 "memory_correctness_oom_20260712": (
   "6282798ce9942c6297cbdf2963aa7a3c65c6b807","6282798","2026-07-12","thesis_bc_project",
   "unsuccessful/oom（memory-path 正確性 UM b10240 OOM）","2368269.opbs"),
 "memory_correctness_failfast_20260712": (
   "29d28c50dec5e70f8d3a9a2341904e1ee94c65f3","29d28c50","2026-07-12","thesis_bc_project",
   "unsuccessful/early_terminated（比較不一致 fail-fast）","2368398.opbs"),
 "oldtree_f05ec52_20260512": (
   "f05ec52ae657df40224f624e30f9cc78aaa3bd48","f05ec52","2026-05-12","mylab/research",
   "memory_scalability（UM オーバーサブスクリプション feasibility, 旧ツリー）; main_performance/seven_implementations（legacy, 近似・旧ツリー代表）","UMv2(not_recorded)"),
}

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def role_of(rel):
    b = os.path.basename(rel)
    if b == "CMakeLists.txt": return "cmake"
    if rel.startswith("experiments/") or b.startswith("run_benchmark.cu") or b.startswith("run_ablation.cu") or b.startswith("run_pathmerge_sweep.cu"): return "experiment_entrypoint"
    if b == "main.cpp": return "experiment_entrypoint"
    if rel.startswith("scripts/") or "/scripts/" in rel or rel.count("/")==1 and rel.startswith("scripts/"):
        if b.startswith("build_"): return "build_script"
        if b.startswith("run_"): return "run_script"
        if b.startswith("summarize") or b.startswith("statistical") or b.startswith("compare") or b.startswith("merge") or b.startswith("analyze"): return "analysis_script"
        return "script"
    if b.startswith("build_") and b.endswith(".sh"): return "build_script"
    if b.startswith("run_") and b.endswith(".sh"): return "run_script"
    if b.endswith((".sh",)): return "script"
    if b.endswith((".py",)): return "analysis_script"
    if b.endswith((".hpp",".h",".cuh")): return "header"
    if b.endswith((".cu",".cpp",".cc")): return "source"
    return "other"

def git_blob(commit, path):
    try:
        return subprocess.check_output(["git","-C",REPO,"cat-file","blob",f"{commit}:{path}"])
    except subprocess.CalledProcessError:
        return None

allmap = []
integrity_fail = []
for sid,(full,short,date,root,covers,jobs) in SNAP.items():
    dest = os.path.join(CS,sid)
    files = []
    for dp,_,fns in os.walk(dest):
        for fn in fns:
            if fn in ("SOURCE_MANIFEST.tsv","SHA256SUMS","BUILD_ENV.md"): continue
            fp = os.path.join(dp,fn)
            rel = os.path.relpath(fp,dest)
            files.append(rel)
    files.sort()
    rows=[]; sums=[]
    for rel in files:
        with open(os.path.join(dest,rel),'rb') as f: content=f.read()
        sh = sha256_bytes(content)
        # 整合性検証: git blob と内容一致か
        orig_repo_path = f"{root}/{rel}"
        blob = git_blob(full, orig_repo_path)
        if blob is None:
            integ="BLOB_NOT_FOUND"; integrity_fail.append(f"{sid}:{rel} blob missing at {orig_repo_path}")
        elif sha256_bytes(blob)==sh:
            integ="MATCH"
        else:
            integ="MISMATCH"; integrity_fail.append(f"{sid}:{rel} content != git blob")
        rows.append([rel, sid, full, orig_repo_path, str(len(content)), sh, role_of(rel), integ])
        sums.append(f"{sh}  {rel}")
    # write SOURCE_MANIFEST.tsv
    with open(os.path.join(dest,"SOURCE_MANIFEST.tsv"),"w",newline='') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n')
        w.writerow(["SnapshotRelPath","SourceSnapshotID","OriginalCommit","OriginalRepoPath","SizeBytes","SHA256","Role","GitBlobIntegrity"])
        w.writerows(rows)
    with open(os.path.join(dest,"SHA256SUMS"),"w") as f:
        f.write("\n".join(sorted(sums))+"\n")
    print(f"{sid}: {len(rows)} files, integrity MATCH={sum(1 for r in rows if r[7]=='MATCH')}/{len(rows)}")
    allmap.append((sid,full,short,date,root,covers,jobs,len(rows)))

# _legacy_audit/LEGACY_COMMIT_TO_SNAPSHOT.tsv (raw 取得・再生成には不要な旧履歴対応表)
_audit_dir = os.path.join(CS, "_legacy_audit")
os.makedirs(_audit_dir, exist_ok=True)
with open(os.path.join(_audit_dir,"LEGACY_COMMIT_TO_SNAPSHOT.tsv"),"w",newline='') as f:
    w=csv.writer(f,delimiter='\t',lineterminator='\n')
    w.writerow(["SourceSnapshotID","OriginalCommit","CommitShort","CommitDate","OriginalTreeRoot","CoversExperiments","PBSJobIDs","FileCount"])
    for sid,full,short,date,root,covers,jobs,n in allmap:
        w.writerow([sid,full,short,date,root,covers,jobs,n])

print("\nintegrity failures:", len(integrity_fail))
for x in integrity_fail[:20]: print("  -",x)
