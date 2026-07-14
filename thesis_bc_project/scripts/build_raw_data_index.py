#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""raw_data/ の索引を **raw_data/ 自身から** 自己完結で再生成する（build_miyabi 非依存）。

生成物:
  raw_data/MANIFEST.tsv            全 raw の完全マニフェスト
  raw_data/SHA256SUMS              全 raw データファイルの SHA256
  raw_data/RAW_DATA_INDEX.tsv      Git 内 raw の正式参照索引
  result/provenance/RAW_DATA_MIGRATION.tsv  旧 result/failure パス→新 raw パス対応（初回のみ session から取込; 以後は自己参照）

正式参照 = RawPath / SourceSnapshotID / SHA256 / PBSJobID。commit SHA には依存しない。
"""
import os, re, csv, hashlib, sys

TBP = "/work/gj17/j17000/m5291091/lab/thesis_bc_project"
RAW = os.path.join(TBP, "raw_data")
MIG = os.path.join(TBP, "result/provenance/RAW_DATA_MIGRATION.tsv")
SESS_MOVES = "/home/j17000/.copilot/session-state/d1db1e20-60af-40b5-8232-80d5d66e5ab7/files/migration_done.tsv"

META_NAMES = {"README.md","MANIFEST.tsv","SHA256SUMS","RAW_CLASSIFICATION.tsv","RAW_DATA_INDEX.tsv"}

def sha256(fp):
    h = hashlib.sha256()
    with open(fp,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): h.update(c)
    return h.hexdigest()

# job token -> (SourceSnapshotID, default status)
JOB = {
 '2367583':('small_correctness_20260712','success'),
 '2368587':('memory_correctness_20260712','success'),
 '2369632':('memory_diagnostic_20260713','success'),
 '2368269':('memory_correctness_oom_20260712','oom'),
 '2368398':('memory_correctness_failfast_20260712','early_terminated'),
 '2357334':('phase_def_block_20260710','success'),
 '2354994':('phase_def_block_20260710','success'),
 '2354999':('phase_def_block_20260710','success'),
 '2354329':('phase_def_block_20260710','success'),
 '2354330':('phase_def_block_20260710','success'),
 '2359175':('phase_def_block_20260710','success'),
 '2359080':('phase_def_block_20260710','early_terminated'),
 '2359096':('phase_def_block_20260710','early_terminated'),
 # Gate J1.1: 追加救出した external raw の起源 PBS ジョブ（checkpoint 88faffa = phase_def_block）
 '2355971':('phase_def_block_20260710','success'),  # phaseB kernel-selection 検証
 '2355000':('phase_def_block_20260710','success'),  # PathMerge sweep 325557
 '2355001':('phase_def_block_20260710','success'),  # PathMerge sweep roadNet-PA
 '2359081':('phase_def_block_20260710','success'),  # PathMerge sweep 325557 (b4096,8192)
 '2359169':('phase_def_block_20260710','success'),  # PathMerge sweep email-EuAll
 '2360073':('phase_def_block_20260710','success'),  # PathMerge screen roadNet-CA
 '2360072':('phase_def_block_20260710','success'),  # PathMerge screen roadNet-TX
 '2361041':('phase_def_block_20260710','success'),  # PathMerge roadNet-CA b16
 '2362006':('phase_def_block_20260710','success'),  # PathMerge confirm roadNet-CA
 '2361040':('phase_def_block_20260710','success'),  # PathMerge confirm roadNet-TX
 '2362965':('phase_def_block_20260710','success'),  # PathMerge correctness roadNet-CA
 '2360074':('phase_def_block_20260710','success'),  # PathMerge correctness email-EuAll
 'multi':  ('phase_def_block_20260710','success'),
 'notrecorded':('oldtree_f05ec52_20260512','success'),
}
# 実験ルート（長い順にマッチ）
EXP_ROOTS = [
 'correctness/small_full_vector','correctness/memory_paths',
 'main_performance/proposed_variants','main_performance/seven_implementations',
 'tuning/pathmerge','tuning/kernel_selection',
 'ablation','memory_scalability','profiling',
 'unsuccessful/oom/memory_paths','unsuccessful/early_terminated/memory_paths',
 'unsuccessful/early_terminated/pathmerge_sweep','unsuccessful/failed/profiling',
]
# 実験 -> (DerivedResultPath, UsedInThesis)
DERIVED = {
 'correctness/small_full_vector':('result/correctness/small_full_vector/','yes'),
 'correctness/memory_paths':('result/correctness/memory_paths/','yes'),
 'main_performance/proposed_variants':('result/main_performance/proposed_variants/','yes'),
 'main_performance/seven_implementations':('result/main_performance/seven_implementations/legacy_partial/','partial'),
 'tuning/pathmerge':('result/main_performance/proposed_vs_pathmerge/; result/correctness/pathmerge_tuned/','yes'),
 'tuning/kernel_selection':('result/tuning/kernel_selection/','yes'),
 'ablation':('result/ablation/','yes'),
 'memory_scalability':('result/memory_scalability/','feasibility_only'),
 'profiling':('result/profiling/','yes'),
 'unsuccessful/oom/memory_paths':('result/correctness/memory_paths/','no'),
 'unsuccessful/early_terminated/memory_paths':('result/correctness/memory_paths/canonical_job_2368587/','no'),
 'unsuccessful/early_terminated/pathmerge_sweep':('result/tuning/pathmerge/','no'),
 'unsuccessful/failed/profiling':('result/profiling/','no'),
}
FAILSUM = {
 'unsuccessful/oom/memory_paths':'failure/failed/oom/memory_correctness_2368269/',
 'unsuccessful/early_terminated/memory_paths':'failure/early_terminated/memory_correctness_2368398/',
 'unsuccessful/early_terminated/pathmerge_sweep':'failure/early_terminated/',
 'unsuccessful/failed/profiling':'failure/incomplete/',
}

# ---- J0 23 件の OriginalPath（build_miyabi/thesis_bc_project 由来）----
J0_ORIGIN = {}
def load_j0():
    ext = os.path.join(TBP,"result/EXTERNAL_ARTIFACTS.tsv")
    if not os.path.exists(ext): return
    for r in csv.DictReader(open(ext), delimiter='\t'):
        rp = r.get('RawPath','')
        if rp.startswith('raw_data/') and len(r.get('SHA256',''))==64:
            J0_ORIGIN[rp] = r['OriginalPath']

# PBS stdout ログの由来（project 直下 .o、gitignored、内容不変コピー）
PBS_ORIGIN = {
 'raw_data/correctness/small_full_vector/_job/job_2367583_20260712/pbs_stdout.log':
   'thesis_bc_project/bc_small_correct.o2367583',
}

def load_existing_manifest_origins():
    """既存 MANIFEST.tsv から RawPath->OriginalPath を読む（regen 時の自己修復; EXTERNAL 非依存）。"""
    m = {}
    fp = os.path.join(RAW,'MANIFEST.tsv')
    if not os.path.exists(fp): return m
    for r in csv.DictReader(open(fp), delimiter='\t'):
        rp = r.get('RawPath',''); op = r.get('OriginalPath','')
        if rp and op and op != 'build_miyabi(gitignored)':
            m[rp] = op
    return m

def load_migration():
    """RawPath -> OriginalResultPath (109). 優先: 既存 MIG。無ければ session。"""
    m = {}
    src = MIG if os.path.exists(MIG) else SESS_MOVES
    if not os.path.exists(src): return m
    for r in csv.DictReader(open(src), delimiter='\t'):
        # MIG: OldPath,NewRawPath,...; session: OriginalResultPath,NewRawPath,SHA256
        old = r.get('OldPath') or r.get('OriginalResultPath')
        new = r.get('NewRawPath')
        if old and new: m[new] = old
    return m

def parse(rel):
    parts = rel.split('/')
    fname = parts[-1]
    jidx = next((i for i,p in enumerate(parts) if p.startswith('job_')), None)
    if jidx is not None:
        token, date = parts[jidx][len('job_'):].rsplit('_',1)
        pre = parts[:jidx]
    else:
        mm = re.match(r'pbs_stdout_job_([0-9]+)_([0-9]+)\.log', fname)
        token, date = (mm.group(1), mm.group(2)) if mm else ('notrecorded','na')
        pre = parts[:-1]
    exp = next((e for e in EXP_ROOTS if rel.startswith(e+'/')), pre[0] if pre else 'unknown')
    rest = pre[len(exp.split('/')):]
    # seven_implementations: legacy_partial/<size>/<variant> -> graph=size, impl=variant
    if exp == 'main_performance/seven_implementations' and rest and rest[0] == 'legacy_partial':
        rest = rest[1:]
    graph = rest[0] if len(rest)>=1 else 'n/a'
    impl  = rest[1] if len(rest)>=2 else 'n/a'
    config= rest[2] if len(rest)>=3 else ('n/a' if impl!='n/a' else 'n/a')
    # 可読ラベル補正（whole-run / job-meta マーカ）
    label = {'_run':'all_impls','_job':'job_meta'}
    impl = label.get(impl, impl)
    graph = label.get(graph, graph)
    return exp, graph, impl, config, token, date

def status_of(rel, token):
    if rel.startswith('unsuccessful/oom'): return 'oom'
    if rel.startswith('unsuccessful/early_terminated'): return 'early_terminated'
    if rel.startswith('unsuccessful/failed'): return 'failed'
    return JOB.get(token,('','success'))[1]

def main():
    load_j0()
    mig = load_migration()
    existing = load_existing_manifest_origins()
    files = []
    for dp,_,fns in os.walk(RAW):
        for fn in fns:
            if fn in META_NAMES: continue
            fp = os.path.join(dp,fn)
            rel = os.path.relpath(fp, RAW)
            files.append(rel)
    files.sort()
    rows = []
    for rel in files:
        fp = os.path.join(RAW, rel)
        exp, graph, impl, config, token, date = parse(rel)
        snap = JOB.get(token, ('oldtree_f05ec52_20260512','success'))[0]
        status = status_of(rel, token)
        rawpath = 'raw_data/'+rel
        orig = (mig.get(rawpath) or J0_ORIGIN.get(rawpath) or PBS_ORIGIN.get(rawpath)
                or existing.get(rawpath) or 'build_miyabi(gitignored)')
        pbsjob = token if re.match(r'^[0-9]+$', token) else ('multi' if token=='multi' else 'not_recorded')
        derived, uit = DERIVED.get(exp, ('',''))
        failsum = FAILSUM.get(exp,'') if rel.startswith('unsuccessful/') else ''
        rows.append(dict(
            RawPath=rawpath, Experiment=exp, Graph=graph, Implementation=impl,
            Configuration=config, Status=status, RunDate=date, PBSJobID=pbsjob,
            SourceSnapshotID=snap, OriginalPath=orig, OriginalFilename=os.path.basename(rel),
            SizeBytes=str(os.path.getsize(fp)), SHA256=sha256(fp),
            UsedInThesis=uit, DerivedResultPath=derived, FailureSummaryPath=failsum, Notes=''))
    # MANIFEST.tsv
    cols=['RawPath','Experiment','Graph','Implementation','Configuration','Status','RunDate',
          'PBSJobID','SourceSnapshotID','OriginalPath','OriginalFilename','SizeBytes','SHA256',
          'UsedInThesis','DerivedResultPath','FailureSummaryPath','Notes']
    with open(os.path.join(RAW,'MANIFEST.tsv'),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,delimiter='\t'); w.writeheader()
        for r in rows: w.writerow(r)
    # SHA256SUMS
    with open(os.path.join(RAW,'SHA256SUMS'),'w') as f:
        for r in rows: f.write(f"{r['SHA256']}  {r['RawPath'][len('raw_data/'):]}\n")
    # RAW_DATA_INDEX.tsv（正式参照索引）
    icols=['RawPath','Experiment','Graph','Implementation','Configuration','Status',
           'PBSJobID','SourceSnapshotID','OriginalPath','SHA256','SizeBytes','UsedInThesis','DerivedResultPath']
    with open(os.path.join(RAW,'RAW_DATA_INDEX.tsv'),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=icols,delimiter='\t',extrasaction='ignore'); w.writeheader()
        for r in rows: w.writerow(r)
    print(f'MANIFEST rows: {len(rows)}')
    print(f'SHA256SUMS: {len(rows)}')
    print(f'RAW_DATA_INDEX rows: {len(rows)}')
    # 末尾空白(空末尾列由来の末尾タブ)を除去してクリーンな TSV にする
    for fn in ('MANIFEST.tsv','RAW_DATA_INDEX.tsv'):
        fp = os.path.join(RAW, fn)
        lines = [ln.rstrip() for ln in open(fp).read().split('\n')]
        with open(fp,'w') as f:
            f.write('\n'.join(lines))
            if lines and lines[-1] != '':
                f.write('\n')

if __name__=='__main__':
    main()
