#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate J1.1: external 42 件の全件監査表を生成し、EXTERNAL_ARTIFACTS.tsv を最終条件へスリム化する。

出力:
  result/provenance/EXTERNAL_ARTIFACTS_AUDIT.tsv  (42件, 11列; section 2)
  result/EXTERNAL_ARTIFACTS.tsv                    (最終: 再生成可能除外 sqlite 3 + directory record 1)

SHA256 / SizeBytes は物理ファイルからスクリプト計算（手計算・逆算しない）。
移行済みは TargetRawPath を rescue スクリプトの対応表と一致させる。
"""
import os, csv, hashlib, glob, sys
from importlib import util as _u

TBP = "/work/gj17/j17000/m5291091/lab/thesis_bc_project"
EXT = os.path.join(TBP, "result/EXTERNAL_ARTIFACTS.tsv")
SRC = os.path.join(TBP, "result/provenance/EXTERNAL_ARTIFACTS_J1_SOURCE.tsv")  # 凍結された J1 42行入力
AUDIT = os.path.join(TBP, "result/provenance/EXTERNAL_ARTIFACTS_AUDIT.tsv")
SQLITE_REC = os.path.join(TBP, "result/provenance/SQLITE_REGENERATION.tsv")
NSYS_VERSION = "NVIDIA Nsight Systems 2025.5.1.121-255136380782v0"
SQLITE_NOTE = ("nsys .sqlite は追跡済み .nsys-rep から再生成可能（実証済, 同サイズ生成）。"
               "cmd: nsys export --type sqlite --force-overwrite true --output <name>.sqlite "
               "raw_data/profiling/job_2359175_20260711/<name>.nsys-rep ; "
               "nsys=2025.5.1.121; 詳細=result/provenance/SQLITE_REGENERATION.tsv")

# rescue スクリプトの MIGRATIONS を読み込み、external->raw 対応を得る
_spec = _u.spec_from_file_location("rescue", os.path.join(TBP, "scripts/rescue_external_raw_j1_1.py"))
_r = _u.module_from_spec(_spec); _spec.loader.exec_module(_r)
# external basename -> target raw path
MIG_BY_SRC = {src: dst for (src, dst, job, snap) in _r.MIGRATIONS}

def sha256(fp):
    h = hashlib.sha256()
    with open(fp, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

# nsys .sqlite -> 対応 .nsys-rep（raw_data 内）
SQLITE_SRC = {
 "build_miyabi/result_profiling_20260711_092110_2359175/ablation_H1W1A0.sqlite":
   "raw_data/profiling/job_2359175_20260711/ablation_H1W1A0.nsys-rep",
 "build_miyabi/result_profiling_20260711_092110_2359175/ablation_H1W1A1.sqlite":
   "raw_data/profiling/job_2359175_20260711/ablation_H1W1A1.nsys-rep",
 "build_miyabi/result_profiling_20260711_092110_2359175/um_prefetch_gpu_opt.sqlite":
   "raw_data/profiling/job_2359175_20260711/um_prefetch_gpu_opt.nsys-rep",
}

# EXTERNAL Artifact 名 -> phaseB 物理ファイル（build_miyabi/phaseB_verify_out/<name>）
def phaseB_target(artifact):
    name = os.path.basename(artifact)
    return MIG_BY_SRC.get("build_miyabi/phaseB_verify_out/" + name)

# EXCLUDE_NOT_USED（build/smoke/superseded/old-tree, 保持 raw に対応せず論文非引用）
EXCLUDE = {
 "bc_build_smoke.o2360062": "T0.3 clean build + block smoke test（実験データなし; 再ビルドで再現可, 論文非引用）",
 "bc_targeted.o2356120": "email-EuAll gpu_opt 3試行の先行run（保持 email 性能は job 2357334 の5試行; 本runは非保持=supersededで論文非引用）",
 "result_0709/bc_ablation.o2349659": "2026-07-09 旧run ablation(4graph)（保持 ablation は 07-10 synthetic/email; データ非保持=superseded, 論文非引用）",
 "result_0709/bc_ablation.o2349660": "2026-07-09 旧run ablation(email)（07-10 job 2354999 に置換; データ非保持=superseded, 論文非引用）",
 "result_0709/bc_kernel_sel.o2349661": "2026-07-09 旧run kernel selection（保持は 07-10 roadNet-PA/TX; データ非保持=superseded, 論文非引用）",
 "result_0709/bc_pm_sweep.o2350440": "2026-07-09 旧run PathMerge sweep roadNet-PA（07-10 以降の sweep に置換; データ非保持=superseded, 論文非引用）",
 "result_0709/bc_pm_sweep.o2350446": "2026-07-09 旧run PathMerge sweep 325557（07-10 以降の sweep に置換; データ非保持=superseded, 論文非引用）",
}

def phys_path(artifact):
    """EXTERNAL Artifact 列 -> 物理パス（存在すれば絶対パス, 無ければ None）。"""
    if artifact.endswith('/'):
        return None  # directory record
    p = os.path.join(TBP, artifact)
    return p if os.path.exists(p) else None

def decide(artifact, typ):
    """(Decision, Reason, TargetRawPath, UsedInThesis, RequiredForRegen, RegeneratableFrom)"""
    base = os.path.basename(artifact)
    if typ == 'sqlite':
        src = SQLITE_SRC.get(artifact, '')
        return ("KEEP_EXTERNAL_REGENERATABLE",
                "nsys .sqlite は追跡済み .nsys-rep から `nsys export --type sqlite` で再生成可能",
                "", "no", "no", src)
    if typ == 'bc_vector_or_err':
        tgt = phaseB_target(artifact) or ''
        return ("MIGRATE_TO_RAW_DATA",
                "kernel 選択検証の BC ベクトル/err（<100MB, 論文の kernel selection 検証に必要）→ raw_data へ移行",
                tgt, "yes", "yes", "")
    if typ == 'directory':
        return ("DUPLICATE_DIRECTORY_RECORD",
                "gitignored ビルド成果物ディレクトリの補助記録（一意な external データではなく origin のみ）",
                "", "no", "no", "")
    if typ == 'pbs_stdout_log':
        # result_0709 プレフィックスを含む素の basename でも解決
        srckey = artifact if artifact in MIG_BY_SRC else base
        if srckey in MIG_BY_SRC:
            return ("MIGRATE_TO_RAW_DATA",
                    "保持実験の起源 PBS ログ → 対応 raw 実験dir へ pbs_stdout.log として移行",
                    MIG_BY_SRC[srckey], "provenance", "no", "")
        if artifact in EXCLUDE:
            return ("EXCLUDE_NOT_USED", EXCLUDE[artifact], "", "no", "no", "")
        return ("EXCLUDE_NOT_USED", "対応する保持 raw 実験なし（論文非引用）", "", "no", "no", "")
    return ("EXCLUDE_NOT_USED", "unclassified", "", "no", "no", "")

def main():
    rows = list(csv.DictReader(open(SRC), delimiter='\t'))
    audit = []
    keep_ext = []  # 最終 EXTERNAL_ARTIFACTS に残す行（元スキーマ）
    for r in rows:
        art = r['ExternalPath']; typ = r['Type']
        pp = phys_path(art)
        if art.endswith('/'):
            exists = 'directory_glob'
            size = '-'; sh = 'n/a'
        elif pp:
            exists = 'yes'; size = str(os.path.getsize(pp)); sh = sha256(pp)
        else:
            exists = 'no'; size = r.get('SizeBytes','?'); sh = r.get('SHA256','?')
        dec, reason, tgt, used, req, regen = decide(art, typ)
        audit.append(dict(
            Artifact=art, Exists=exists, Type=typ, SizeBytes=size, SHA256=sh,
            UsedInThesis=used, RequiredForRegeneration=req, RegeneratableFrom=regen,
            Decision=dec, Reason=reason, TargetRawPath=tgt))
        if dec in ("KEEP_EXTERNAL_REGENERATABLE", "DUPLICATE_DIRECTORY_RECORD"):
            keep_ext.append(r)

    # 監査表（section 2, 11列）
    acols = ["Artifact","Exists","Type","SizeBytes","SHA256","UsedInThesis",
             "RequiredForRegeneration","RegeneratableFrom","Decision","Reason","TargetRawPath"]
    with open(AUDIT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=acols, delimiter='\t', lineterminator='\n'); w.writeheader()
        for a in audit:
            # 末尾列 TargetRawPath が空だと行末にタブが残り git の trailing-whitespace 検査に落ちる。
            # 移行先を持たない行(sqlite/EXCLUDE/directory)は空タブではなく明示値 not_applicable を記録する。
            # audit 本体は変更せずコピー上で置換する(RAW_CLASSIFICATION 生成ロジックは従来通り TargetRawPath 空を参照)。
            row = dict(a)
            if not row["TargetRawPath"]:
                row["TargetRawPath"] = "not_applicable"
            w.writerow(row)

    # 最終 EXTERNAL_ARTIFACTS.tsv（元スキーマ維持、残す行のみ）。SizeBytes/SHA256 は物理から再計算。
    ecols = ["ExternalPath","Type","SizeBytes","SHA256","Reason","RetentionStatus","Note"]
    with open(EXT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ecols, delimiter='\t', lineterminator='\n'); w.writeheader()
        for r in keep_ext:
            art = r['ExternalPath']
            pp = phys_path(art)
            if pp:
                r['SizeBytes'] = str(os.path.getsize(pp)); r['SHA256'] = sha256(pp)
            if r['Type'] == 'sqlite':
                r['Reason'] = SQLITE_NOTE
                r['Note'] = "source .nsys-rep in raw_data/profiling/ (Git tracked); origin=" + art
            w.writerow({k: r.get(k,'') for k in ecols})

    # section 5: .sqlite 再生成記録
    scols = ["SqlitePath","SqliteSizeBytes","SqliteSHA256_original","NsysRepRawPath",
             "NsysRepSHA256","RegenCommand","NsysVersion","Regeneratable"]
    with open(SQLITE_REC, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=scols, delimiter='\t', lineterminator='\n'); w.writeheader()
        for sq, nr in SQLITE_SRC.items():
            pp = phys_path(sq); nrp = os.path.join(TBP, nr)
            name = os.path.basename(sq)
            w.writerow(dict(
                SqlitePath=sq,
                SqliteSizeBytes=str(os.path.getsize(pp)) if pp else "?",
                SqliteSHA256_original=sha256(pp) if pp else "?",
                NsysRepRawPath=nr,
                NsysRepSHA256=sha256(nrp) if os.path.exists(nrp) else "MISSING",
                RegenCommand=f"nsys export --type sqlite --force-overwrite true --output {name} {nr}",
                NsysVersion=NSYS_VERSION,
                Regeneratable="yes" if os.path.exists(nrp) else "no_source_missing"))

    from collections import Counter
    c = Counter(a['Decision'] for a in audit)

    # RAW_CLASSIFICATION.tsv へ J1.1 external 分類を追記（既存 239 行は保全, CurrentPath で重複防止）
    RC = os.path.join(TBP, "raw_data/RAW_CLASSIFICATION.tsv")
    DEC2CLS = {
        "MIGRATE_TO_RAW_DATA": ("raw", "rescued_external_to_raw_data"),
        "KEEP_EXTERNAL_REGENERATABLE": ("excluded_regeneratable", "keep_external_regeneratable"),
        "EXCLUDE_NOT_USED": ("excluded_not_used", "exclude_not_used"),
        "DUPLICATE_DIRECTORY_RECORD": ("directory_record", "keep_external_directory_record"),
    }
    rc_cols = ["CurrentPath","Classification","Reason","TargetPath","SHA256","UsedInThesis","DerivedConsumers","Action"]
    rc_rows = []
    seen = set()
    if os.path.exists(RC):
        with open(RC) as f:
            rr = csv.reader(f, delimiter='\t'); next(rr)
            for row in rr:
                if not row: continue
                rc_rows.append(row); seen.add(row[0])
    rc_new = 0
    for a in audit:
        cur = "thesis_bc_project/" + a['Artifact']
        if cur in seen:
            continue
        cls, act = DEC2CLS[a['Decision']]
        rc_rows.append([cur, cls, "Gate J1.1: "+a['Reason'], a['TargetRawPath'] or a['RegeneratableFrom'] or "-",
                        a['SHA256'], a['UsedInThesis'], "", act])
        rc_new += 1
    with open(RC, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n'); w.writerow(rc_cols); w.writerows(rc_rows)

    print("=== Decision breakdown (42) ===")
    for k in sorted(c): print(f"  {k}: {c[k]}")
    print(f"audit rows: {len(audit)}")
    print(f"EXTERNAL_ARTIFACTS final rows: {len(keep_ext)}")
    print(f"RAW_CLASSIFICATION new rows appended: {rc_new} (total {len(rc_rows)})")

if __name__ == "__main__":
    main()
