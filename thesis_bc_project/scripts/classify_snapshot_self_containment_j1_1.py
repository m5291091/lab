#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate J1.1: 7 code_snapshots の自己完結性を実検証して再分類する。

分類:
  SELF_CONTAINED                    - snapshot 内容のみでビルド・実行可（外部コード依存・data 不要）
  SELF_CONTAINED_WITH_CANONICAL_DATA- snapshot + canonical data のみ（外部コード依存なし）
  DEPENDENCIES_CAPTURED             - 外部コード依存を要するが _dependencies/ に固定済（実体+SHA一致を検証）
  INCOMPLETE                        - 上記いずれも満たさない（欠損あり）

「README に書いただけ」を排除するため、BUILD_ENV に記載の各 DependencyID について
 (1) _dependencies/<id>/ が実在し, (2) SHA256SUMS が全件一致し, (3) manifest digest が BUILD_ENV 記載値と一致
を検証。canonical graph pin は graph_catalog.tsv と SHA256 一致を検証。
出力: code_snapshots/SELF_CONTAINMENT.tsv
"""
import os, csv, re, hashlib, subprocess, sys

TBP = "/work/gj17/j17000/m5291091/lab/thesis_bc_project"
CS = os.path.join(TBP, "code_snapshots")
DEPS = os.path.join(CS, "_dependencies")
CATALOG = os.path.join(TBP, "result/datasets/graph_catalog.tsv")
OUT = os.path.join(CS, "SELF_CONTAINMENT.tsv")
SNAPSHOTS = ["small_correctness_20260712","phase_def_block_20260710","memory_correctness_20260712",
             "memory_diagnostic_20260713","memory_correctness_oom_20260712",
             "memory_correctness_failfast_20260712","oldtree_f05ec52_20260512"]

def sha256_file(fp):
    h = hashlib.sha256()
    with open(fp, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def verify_sha256sums(root):
    """root/SHA256SUMS を検証。全件一致で True。"""
    fp = os.path.join(root, "SHA256SUMS")
    if not os.path.exists(fp):
        return False, "no SHA256SUMS"
    n = ok = 0
    for line in open(fp):
        line = line.rstrip("\n")
        if not line:
            continue
        exp, rel = line.split("  ", 1)
        n += 1
        tp = os.path.join(root, rel)
        if os.path.exists(tp) and sha256_file(tp) == exp:
            ok += 1
    return (ok == n and n > 0), f"{ok}/{n}"

def parse_build_env(snap):
    """BUILD_ENV の J1.1 セクションから (DependencyID, DependencySHA256) と graph 名を抽出。"""
    fp = os.path.join(CS, snap, "BUILD_ENV.md")
    txt = open(fp).read()
    m = re.search(r"GATE_J1_1_DEPENDENCIES:BEGIN(.*?)GATE_J1_1_DEPENDENCIES:END", txt, re.S)
    if not m:
        return None, None
    sec = m.group(1)
    deps = re.findall(r"\|\s*`([a-z0-9_]+_\d{8})`\s*\|\s*`[^`]+`\s*\|\s*`([0-9a-f]{64})`\s*\|", sec)
    graphs = re.findall(r"\|\s*([A-Za-z0-9_\-]+)\s*\|\s*`(data/[^`]+)`\s*\|\s*`([0-9a-f]{64})`", sec)
    return deps, graphs

def load_catalog():
    return {r['graph']: r for r in csv.DictReader(open(CATALOG), delimiter='\t')}

def main():
    catalog = load_catalog()
    rows = []
    incomplete = 0
    for snap in SNAPSHOTS:
        problems = []
        # (1) snapshot 自身の内容整合
        sok, sdet = verify_sha256sums(os.path.join(CS, snap))
        if not sok:
            problems.append(f"snapshot_SHA256SUMS={sdet}")
        # (2) BUILD_ENV の依存記載を検証
        deps, graphs = parse_build_env(snap)
        if deps is None:
            problems.append("no_J1_1_section")
            deps, graphs = [], []
        dep_ok = 0
        for dep_id, dep_dig in deps:
            droot = os.path.join(DEPS, dep_id)
            if not os.path.isdir(droot):
                problems.append(f"dep_missing:{dep_id}"); continue
            dok, ddet = verify_sha256sums(droot)
            if not dok:
                problems.append(f"dep_sha_fail:{dep_id}({ddet})"); continue
            actual_dig = sha256_file(os.path.join(droot, "SHA256SUMS"))
            if actual_dig != dep_dig:
                problems.append(f"dep_digest_mismatch:{dep_id}"); continue
            dep_ok += 1
        # (3) canonical graph pin 検証（BUILD_ENV 記載 SHA == 物理 == catalog）
        graph_ok = 0
        for gname, gpath, gsha in graphs:
            phys = os.path.join(TBP, gpath)
            cat = catalog.get(gname)
            if not os.path.exists(phys):
                problems.append(f"data_missing:{gpath}"); continue
            if sha256_file(phys) != gsha:
                problems.append(f"data_sha_mismatch:{gpath}"); continue
            if not cat or cat['SHA256'] != gsha:
                problems.append(f"catalog_mismatch:{gname}"); continue
            graph_ok += 1
        # 分類
        if problems:
            cls = "INCOMPLETE"; incomplete += 1
        elif deps:
            cls = "DEPENDENCIES_CAPTURED"
        elif graphs:
            cls = "SELF_CONTAINED_WITH_CANONICAL_DATA"
        else:
            cls = "SELF_CONTAINED"
        rows.append(dict(Snapshot=snap, Classification=cls,
                         SnapshotContent=sdet, DepsVerified=f"{dep_ok}/{len(deps)}",
                         GraphsVerified=f"{graph_ok}/{len(graphs)}",
                         Problems=";".join(problems) if problems else "none"))
    cols = ["Snapshot","Classification","SnapshotContent","DepsVerified","GraphsVerified","Problems"]
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter='\t', lineterminator='\n'); w.writeheader()
        for r in rows: w.writerow(r)
    print("=== Snapshot self-containment classification ===")
    for r in rows:
        print(f"  {r['Snapshot']:38s} {r['Classification']:32s} deps={r['DepsVerified']} graphs={r['GraphsVerified']} {r['Problems']}")
    print(f"INCOMPLETE count: {incomplete}")
    if incomplete:
        print("!! INCOMPLETE present -> DO NOT proceed to commit; report required.")
        sys.exit(2)

if __name__ == "__main__":
    main()
