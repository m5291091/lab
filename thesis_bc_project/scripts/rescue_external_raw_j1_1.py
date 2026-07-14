#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate J1.1: 残存 external raw を raw_data/ へ内容不変で救出する（再現可能・冪等）。

対象:
  1) PBS 起源ログ 23 件（project 直下 / result_0709 の gitignored `.o` のうち、保持実験に対応するもの）
     -> 対応する raw 実験ディレクトリへ `pbs_stdout.log` としてコピー
  2) phaseB kernel-selection BC ベクトル/err 8 件（build_miyabi/phaseB_verify_out/*）
     -> raw_data/tuning/kernel_selection/<Graph>/<block|shared>/gpu_opt/job_2355971_20260710/ へ原名コピー

コピーは内容不変（shutil.copy2）。コピー後に SHA256 一致を検証（不一致なら異常終了）。
result/provenance/RAW_DATA_MIGRATION.tsv へ external->raw 対応行を追記（SHA256 はスクリプト計算）。
索引（MANIFEST/SHA256SUMS/RAW_DATA_INDEX）は本スクリプトでは触らず、
別途 scripts/build_raw_data_index.py で自己完結再生成する。
"""
import os, csv, hashlib, shutil, sys

TBP = "/work/gj17/j17000/m5291091/lab/thesis_bc_project"
MIG = os.path.join(TBP, "result/provenance/RAW_DATA_MIGRATION.tsv")

# (external_source_rel_to_TBP, target_raw_path_rel_to_TBP, true_pbs_job, source_snapshot_id)
MIGRATIONS = [
 # --- Group A: 保持済み一次実験の起源 PBS ログ -> 実験dir/pbs_stdout.log ---
 ("bc_ablation.o2354994",   "raw_data/ablation/synthetic/job_2354994_20260710/pbs_stdout.log", "2354994", "phase_def_block_20260710"),
 ("bc_ablation.o2354999",   "raw_data/ablation/email-EuAll/job_2354999_20260710/pbs_stdout.log", "2354999", "phase_def_block_20260710"),
 ("bc_kernel_sel.o2354329", "raw_data/tuning/kernel_selection/roadNet-PA/gpu_opt_forced/job_2354329_20260710/pbs_stdout.log", "2354329", "phase_def_block_20260710"),
 ("bc_kernel_sel.o2354330", "raw_data/tuning/kernel_selection/roadNet-TX/gpu_opt_forced/job_2354330_20260710/pbs_stdout.log", "2354330", "phase_def_block_20260710"),
 ("bc_targeted.o2357334",   "raw_data/main_performance/proposed_variants/email-EuAll/_run/job_2357334_20260711/pbs_stdout.log", "2357334", "phase_def_block_20260710"),
 ("bc_targeted.o2357335",   "raw_data/main_performance/proposed_variants/roadNet-PA/_run/job_2357334_20260711/pbs_stdout.log", "2357335", "phase_def_block_20260710"),
 ("bc_targeted.o2357336",   "raw_data/main_performance/proposed_variants/roadNet-TX/_run/job_2357334_20260711/pbs_stdout.log", "2357336", "phase_def_block_20260710"),
 ("bc_targeted.o2357337",   "raw_data/main_performance/proposed_variants/roadNet-CA/_run/job_2357334_20260711/pbs_stdout.log", "2357337", "phase_def_block_20260710"),
 ("bc_profiling.o2359175",  "raw_data/profiling/job_2359175_20260711/pbs_stdout.log", "2359175", "phase_def_block_20260710"),
 ("bc_pm_sweep.o2359080",   "raw_data/unsuccessful/early_terminated/pathmerge_sweep/roadNet-PA/job_2359080_20260711/pbs_stdout.log", "2359080", "phase_def_block_20260710"),
 ("bc_pm_sweep.o2359096",   "raw_data/unsuccessful/early_terminated/pathmerge_sweep/email-EuAll/job_2359096_20260711/pbs_stdout.log", "2359096", "phase_def_block_20260710"),
 # --- Group B: PathMerge tuning sweep の起源 PBS ログ（集約 job_multi の構成要素）---
 ("bc_pm_sweep.o2355000",   "raw_data/tuning/pathmerge/325557/pathmerge_bc/job_2355000_20260710/pbs_stdout.log", "2355000", "phase_def_block_20260710"),
 ("bc_pm_sweep.o2355001",   "raw_data/tuning/pathmerge/roadNet-PA/pathmerge_bc/job_2355001_20260710/pbs_stdout.log", "2355001", "phase_def_block_20260710"),
 ("bc_pm_sweep.o2359081",   "raw_data/tuning/pathmerge/325557/pathmerge_bc/job_2359081_20260711/pbs_stdout.log", "2359081", "phase_def_block_20260710"),
 ("bc_pm_sweep.o2359169",   "raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_2359169_20260711/pbs_stdout.log", "2359169", "phase_def_block_20260710"),
 ("bc_ca_screen.o2360073",  "raw_data/tuning/pathmerge/roadNet-CA/pathmerge_bc/job_2360073_20260711/pbs_stdout.log", "2360073", "phase_def_block_20260710"),
 ("bc_tx_screen.o2360072",  "raw_data/tuning/pathmerge/roadNet-TX/pathmerge_bc/job_2360072_20260711/pbs_stdout.log", "2360072", "phase_def_block_20260710"),
 ("bc_ca_b16.o2361041",     "raw_data/tuning/pathmerge/roadNet-CA/pathmerge_bc/job_2361041_20260711/pbs_stdout.log", "2361041", "phase_def_block_20260710"),
 ("bc_ca_confirm.o2362006", "raw_data/tuning/pathmerge/roadNet-CA/pathmerge_bc/job_2362006_20260711/pbs_stdout.log", "2362006", "phase_def_block_20260710"),
 ("bc_tx_confirm.o2361040", "raw_data/tuning/pathmerge/roadNet-TX/pathmerge_bc/job_2361040_20260711/pbs_stdout.log", "2361040", "phase_def_block_20260710"),
 ("bc_ca_correct.o2362965", "raw_data/tuning/pathmerge/roadNet-CA/pathmerge_bc/job_2362965_20260711/pbs_stdout.log", "2362965", "phase_def_block_20260710"),
 ("bc_pm_correct.o2360074", "raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_2360074_20260711/pbs_stdout.log", "2360074", "phase_def_block_20260710"),
 # --- Group C: phaseB kernel-selection 検証ジョブの起源 PBS ログ ---
 ("bc_phaseB_verify.o2355971", "raw_data/tuning/kernel_selection/_phaseB_verify/job_2355971_20260710/pbs_stdout.log", "2355971", "phase_def_block_20260710"),
 # --- Section 3: phaseB kernel-selection BC ベクトル/err（原名維持）---
 ("build_miyabi/phaseB_verify_out/bc_auto_benchmark_7000_41459.txt",  "raw_data/tuning/kernel_selection/benchmark_7000_41459/block/gpu_opt/job_2355971_20260710/bc_auto_benchmark_7000_41459.txt", "2355971", "phase_def_block_20260710"),
 ("build_miyabi/phaseB_verify_out/auto_benchmark_7000_41459.err",     "raw_data/tuning/kernel_selection/benchmark_7000_41459/block/gpu_opt/job_2355971_20260710/auto_benchmark_7000_41459.err", "2355971", "phase_def_block_20260710"),
 ("build_miyabi/phaseB_verify_out/bc_shared_benchmark_7000_41459.txt","raw_data/tuning/kernel_selection/benchmark_7000_41459/shared/gpu_opt/job_2355971_20260710/bc_shared_benchmark_7000_41459.txt", "2355971", "phase_def_block_20260710"),
 ("build_miyabi/phaseB_verify_out/shared_benchmark_7000_41459.err",   "raw_data/tuning/kernel_selection/benchmark_7000_41459/shared/gpu_opt/job_2355971_20260710/shared_benchmark_7000_41459.err", "2355971", "phase_def_block_20260710"),
 ("build_miyabi/phaseB_verify_out/bc_auto_email-EuAll.txt",   "raw_data/tuning/kernel_selection/email-EuAll/block/gpu_opt/job_2355971_20260710/bc_auto_email-EuAll.txt", "2355971", "phase_def_block_20260710"),
 ("build_miyabi/phaseB_verify_out/auto_email-EuAll.err",      "raw_data/tuning/kernel_selection/email-EuAll/block/gpu_opt/job_2355971_20260710/auto_email-EuAll.err", "2355971", "phase_def_block_20260710"),
 ("build_miyabi/phaseB_verify_out/bc_shared_email-EuAll.txt", "raw_data/tuning/kernel_selection/email-EuAll/shared/gpu_opt/job_2355971_20260710/bc_shared_email-EuAll.txt", "2355971", "phase_def_block_20260710"),
 ("build_miyabi/phaseB_verify_out/shared_email-EuAll.err",    "raw_data/tuning/kernel_selection/email-EuAll/shared/gpu_opt/job_2355971_20260710/shared_email-EuAll.err", "2355971", "phase_def_block_20260710"),
]

REASON = "Gate J1.1: 残存 external raw を raw_data へ内容不変救出 (SHA256 保持)"

def sha256(fp):
    h = hashlib.sha256()
    with open(fp, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def main():
    added = []
    errors = []
    for src_rel, dst_rel, job, snap in MIGRATIONS:
        src = os.path.join(TBP, src_rel)
        dst = os.path.join(TBP, dst_rel)
        if not os.path.exists(src):
            errors.append(f"MISSING SOURCE: {src_rel}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        src_sha = sha256(src)
        if os.path.exists(dst):
            if sha256(dst) != src_sha:
                errors.append(f"DEST EXISTS w/ DIFFERENT CONTENT: {dst_rel}")
                continue
        else:
            shutil.copy2(src, dst)
            if sha256(dst) != src_sha:
                errors.append(f"COPY SHA MISMATCH: {dst_rel}")
                continue
        added.append((src_rel, dst_rel, "raw", src_sha, job, snap, REASON))

    if errors:
        print("ERRORS:")
        for e in errors:
            print("  " + e)
        sys.exit(1)

    # RAW_DATA_MIGRATION.tsv へ external->raw 対応行を追記（既存 OldPath は温存、重複追加しない）
    existing_old = set()
    rows = []
    header = ["OldPath","NewRawPath","Classification","SHA256","PBSJobID","SourceSnapshotID","Reason"]
    if os.path.exists(MIG):
        with open(MIG) as f:
            r = csv.reader(f, delimiter='\t')
            hdr = next(r)
            for row in r:
                if not row:
                    continue
                rows.append(row)
                existing_old.add(row[0])
    new_count = 0
    for src_rel, dst_rel, cls, sha, job, snap, reason in added:
        old = "thesis_bc_project/" + src_rel
        if old in existing_old:
            continue
        rows.append([old, dst_rel, cls, sha, job, snap, reason])
        new_count += 1
    with open(MIG, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)

    print(f"migrated files : {len(added)}")
    print(f"new MIGRATION rows appended : {new_count}")
    print(f"RAW_DATA_MIGRATION total rows (excl header) : {len(rows)}")

if __name__ == "__main__":
    main()
