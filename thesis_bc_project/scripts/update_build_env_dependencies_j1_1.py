#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate J1.1: 各 code_snapshot の BUILD_ENV.md へ依存固定情報を追記する（冪等; マーカ区切り）。

追記内容:
  ## 依存固定 (Gate J1.1)
    - コード依存表: DependencyID / DependencyPath / DependencySHA256 / UsedByTarget
    - canonical graph data pin: graph_catalog.tsv と一致（path/SHA256/Nodes/Edges/directed/symmetrized/preprocessing）
DependencySHA256 = その dependency の SHA256SUMS の SHA256（manifest digest）。
"""
import os, csv, hashlib

TBP = "/work/gj17/j17000/m5291091/lab/thesis_bc_project"
CS = os.path.join(TBP, "code_snapshots")
DEPS = os.path.join(CS, "_dependencies")
CATALOG = os.path.join(TBP, "result/datasets/graph_catalog.tsv")
BEGIN = "<!-- GATE_J1_1_DEPENDENCIES:BEGIN -->"
END = "<!-- GATE_J1_1_DEPENDENCIES:END -->"

def sha256_file(fp):
    h = hashlib.sha256()
    with open(fp, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def dep_digest(dep_id):
    return sha256_file(os.path.join(DEPS, dep_id, "SHA256SUMS"))

# dependency -> UsedByTarget（thesis 用 / oldtree 用）
CUGRAPH = "cugraph_bc_subset_20260710"
BW = "bandwidth_tool_20260710"

THESIS_DEPS = [
 (CUGRAPH, "code_snapshots/_dependencies/"+CUGRAPH+"/",
  "run_benchmark（cugraph_bc baseline; Stage1 libcugraph_bc_mini.a + Stage2 link）"),
 (BW, "code_snapshots/_dependencies/"+BW+"/",
  "bandwidth_benchmark（Stage2 configure に必須; 実行は profiling のみ）"),
]
OLDTREE_DEPS = [
 (CUGRAPH, "code_snapshots/_dependencies/"+CUGRAPH+"/",
  "cuGraph baseline（vendored subset のみ; mini CMake は snapshot 内同梱; 保持 UM 実験は gpu_opt で cuGraph 非使用）"),
]

# snapshot -> (deps, [graph_catalog 名])
SNAP = {
 "small_correctness_20260712": (THESIS_DEPS, ["chain_200","benchmark_7000","benchmark_11023"]),
 "phase_def_block_20260710": (THESIS_DEPS, ["email-EuAll","roadNet-PA","roadNet-TX","roadNet-CA",
     "benchmark_7000","benchmark_11023","56438_300801","benchmark_85830","325557_3216152"]),
 "memory_correctness_20260712": (THESIS_DEPS, ["325557_3216152"]),
 "memory_diagnostic_20260713": (THESIS_DEPS, ["325557_3216152"]),
 "memory_correctness_oom_20260712": (THESIS_DEPS, ["325557_3216152"]),
 "memory_correctness_failfast_20260712": (THESIS_DEPS, ["325557_3216152"]),
 "oldtree_f05ec52_20260512": (OLDTREE_DEPS, ["325557_3216152"]),
}

def load_catalog():
    d = {}
    for r in csv.DictReader(open(CATALOG), delimiter='\t'):
        d[r['graph']] = r
    return d

def build_section(deps, graphs, catalog):
    dig = {CUGRAPH: dep_digest(CUGRAPH), BW: dep_digest(BW)}
    lines = [BEGIN, "", "## 依存固定 (Gate J1.1)", "",
        "履歴削除後も実験時と同一版を特定できるよう、外部コード依存を "
        "`code_snapshots/_dependencies/<DependencyID>/`（内容+用途名, commit SHA非依存）へ一度だけ固定し、"
        "本 snapshot から参照する。各 dependency は `SOURCE_MANIFEST.tsv` / `SHA256SUMS` / `README.md` を持つ。", "",
        "### コード依存", "",
        "| DependencyID | DependencyPath | DependencySHA256 | UsedByTarget |",
        "|:--|:--|:--|:--|"]
    for dep_id, dep_path, used in deps:
        lines.append(f"| `{dep_id}` | `{dep_path}` | `{dig[dep_id]}` | {used} |")
    lines += ["",
        "- `DependencySHA256` = 各 dependency の `SHA256SUMS` の SHA256（manifest digest）。"
        "個別ファイルの SHA256 は dependency 内 `SHA256SUMS` / `SOURCE_MANIFEST.tsv` を参照。",
        "- vendored cuGraph subset（`third_party/cugraph`, tree `eb339d4`）は **全 7 checkpoint で同一内容**"
        "（抽出 commit `88faffa` の git blob と照合済）。", "",
        "### canonical graph data（`result/datasets/graph_catalog.tsv` と一致; snapshot へは非複製）", "",
        "| Graph | canonical path | SHA256 | Nodes | Edges | UsedAsDirected | Symmetrized | Preprocessing |",
        "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for g in graphs:
        c = catalog.get(g)
        if not c:
            lines.append(f"| {g} | (catalog 欠落) | - | - | - | - | - | - |")
            continue
        lines.append(f"| {c['graph']} | `{c['path']}` | `{c['SHA256']}` | {c['n']} | {c['m']} | "
                     f"{c['UsedAsDirected']} | {c['Symmetrized']} | {c['Preprocessing']} |")
    lines += ["",
        "グラフ入力は canonical path（Git 内 `data/`）から取得し、SHA256 は上表（= `graph_catalog.tsv`）で固定。"
        "全実装は無向・非重み BC（CPU は accumulation 時 /2、pathmerge adapter は最終 /2）。", "", END, ""]
    return "\n".join(lines)

def update_build_env(snap, deps, graphs, catalog):
    fp = os.path.join(CS, snap, "BUILD_ENV.md")
    txt = open(fp).read()
    section = build_section(deps, graphs, catalog)
    if BEGIN in txt and END in txt:
        pre = txt[:txt.index(BEGIN)]
        post = txt[txt.index(END)+len(END):]
        txt = pre.rstrip('\n') + "\n\n" + section + post.lstrip('\n')
    else:
        txt = txt.rstrip('\n') + "\n\n" + section
    with open(fp, 'w') as f:
        f.write(txt)

def main():
    catalog = load_catalog()
    for snap, (deps, graphs) in SNAP.items():
        update_build_env(snap, deps, graphs, catalog)
        print(f"updated BUILD_ENV: {snap} (deps={len(deps)}, graphs={len(graphs)})")

if __name__ == "__main__":
    main()
