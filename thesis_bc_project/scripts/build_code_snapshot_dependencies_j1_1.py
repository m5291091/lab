#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate J1.1: code_snapshots の外部コード依存を _dependencies/<DependencyID>/ へ一度だけ固定する。

各 checkpoint の Git tree/blob（履歴が残る間に）から実体を抽出し、DependencyID（内容+用途名, commit SHA非依存）で保存。
全 checkpoint（e32b03e9/88faffa/ac2b409/43d1cf5/6282798/29d28c5/f05ec52）で対象 tree/blob は同一と確認済:
  - third_party/cugraph tree = eb339d4（thesis 6件 + oldtree top-level cugraph/ すべて同一）
  - cugraph_bc_mini/CMakeLists.txt blob = c286a48（thesis 6件; oldtree は別版を snapshot 内に同梱）
  - tools/bandwidth_benchmark.cu blob = c0d4945（thesis 6件; oldtree に該当ターゲットなし）

生成物（各 dependency）: 実体 + SOURCE_MANIFEST.tsv + SHA256SUMS + README.md
抽出は git archive（内容不変）。抽出後に blob SHA を検証する。
"""
import os, csv, hashlib, subprocess, sys, io, tarfile

REPO = "/work/gj17/j17000/m5291091/lab"
TBP = os.path.join(REPO, "thesis_bc_project")
DEPS = os.path.join(TBP, "code_snapshots/_dependencies")
EXTRACT_COMMIT = "88faffa391026852a4440e5b9a063c08c29624f7"  # 全 checkpoint で同一内容

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def sha256_file(fp):
    h = hashlib.sha256()
    with open(fp, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], check=True, capture_output=True).stdout

def extract_tree(commit_path, dest):
    """git archive <commit>:<path> を dest 以下へ内容不変展開。"""
    os.makedirs(dest, exist_ok=True)
    data = subprocess.run(["git", "-C", REPO, "archive", commit_path],
                          check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(dest)

def extract_blob(commit_path, dest_file):
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    data = subprocess.run(["git", "-C", REPO, "show", commit_path],
                          check=True, capture_output=True).stdout
    with open(dest_file, 'wb') as f:
        f.write(data)

# DependencyID -> 定義
DEP_CUGRAPH = "cugraph_bc_subset_20260710"
DEP_BW = "bandwidth_tool_20260710"

def build_cugraph_dep():
    root = os.path.join(DEPS, DEP_CUGRAPH)
    # third_party/cugraph（vendored BC subset, tree eb339d4）
    tp_dest = os.path.join(root, "third_party/cugraph")
    if not os.path.isdir(tp_dest) or not os.listdir(tp_dest):
        extract_tree(f"{EXTRACT_COMMIT}:thesis_bc_project/third_party/cugraph", tp_dest)
    # cugraph_bc_mini/CMakeLists.txt（thesis mini build recipe, blob c286a48）
    mini_dest = os.path.join(root, "cugraph_bc_mini/CMakeLists.txt")
    if not os.path.exists(mini_dest):
        extract_blob(f"{EXTRACT_COMMIT}:thesis_bc_project/cugraph_bc_mini/CMakeLists.txt", mini_dest)
    return root

def build_bw_dep():
    root = os.path.join(DEPS, DEP_BW)
    dest = os.path.join(root, "tools/bandwidth_benchmark.cu")
    if not os.path.exists(dest):
        extract_blob(f"{EXTRACT_COMMIT}:thesis_bc_project/tools/bandwidth_benchmark.cu", dest)
    return root

META = {"SOURCE_MANIFEST.tsv", "SHA256SUMS", "README.md"}

def write_manifests(root, dep_id, git_srcs):
    """dep 内の実体ファイルを走査し SOURCE_MANIFEST.tsv / SHA256SUMS を生成。git blob SHA を照合。"""
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            fp = os.path.join(dp, fn)
            rel = os.path.relpath(fp, root)
            # dependency 直下の meta のみ除外（ネストした同名ファイルは実体として含める）
            if rel in META:
                continue
            files.append(rel)
    files.sort()
    # git blob SHA（照合用）: rel -> thesis_bc_project 内の元パス
    rows = []
    mism = []
    for rel in files:
        fp = os.path.join(root, rel)
        sha = sha256_file(fp)
        # 元 git パス（DependencyID の由来）
        src = git_srcs(rel)
        # git blob（存在すれば）内容 SHA と照合
        gsha = ""
        if src:
            try:
                blob = subprocess.run(["git", "-C", REPO, "show", f"{EXTRACT_COMMIT}:{src}"],
                                      check=True, capture_output=True).stdout
                gsha = sha256_bytes(blob)
                if gsha != sha:
                    mism.append(rel)
            except subprocess.CalledProcessError:
                gsha = "NA"
        rows.append((rel, sha, src, gsha))
    if mism:
        print("BLOB MISMATCH:", *mism, sep="\n  ")
        sys.exit(1)
    # SOURCE_MANIFEST.tsv
    with open(os.path.join(root, "SOURCE_MANIFEST.tsv"), 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(["FileInDependency", "SHA256", "OriginalGitPath", "ExtractedFromCommit"])
        for rel, sha, src, gsha in rows:
            w.writerow([rel, sha, src or "", EXTRACT_COMMIT])
    # SHA256SUMS
    with open(os.path.join(root, "SHA256SUMS"), 'w') as f:
        for rel, sha, src, gsha in rows:
            f.write(f"{sha}  {rel}\n")
    return len(rows)

def cugraph_src(rel):
    if rel.startswith("third_party/cugraph/"):
        return "thesis_bc_project/" + rel
    if rel == "cugraph_bc_mini/CMakeLists.txt":
        return "thesis_bc_project/cugraph_bc_mini/CMakeLists.txt"
    return ""

def bw_src(rel):
    if rel == "tools/bandwidth_benchmark.cu":
        return "thesis_bc_project/tools/bandwidth_benchmark.cu"
    return ""

CUGRAPH_README = """# Dependency: cugraph_bc_subset_20260710

**用途**: `run_benchmark` の cuGraph BC ベースライン（`cugraph_bc`）をビルド・実行するための
vendored cuGraph サブセット + BC 専用ミニビルドの CMake レシピ。

## 内容

- `third_party/cugraph/` — vendored cuGraph サブセット（BC 関連のみ, upstream RAPIDS）。
  Git tree = `eb339d4ae02a0d6f6d9f75658f74fcae59079666`。**全 7 checkpoint で同一**
  （thesis 6 件の `thesis_bc_project/third_party/cugraph`、および oldtree `f05ec52` の
  top-level `cugraph/` すべて同一 tree）。`run_benchmark` は `cpp/include` を API として使用。
- `cugraph_bc_mini/CMakeLists.txt` — BC 専用ミニビルドのレシピ（rapids-cmake を v26.04.00 に固定,
  RMM/RAFT/cuCo/CCCL/spdlog/NVTX3/rapids_logger を CPM 取得）。blob = `c286a48…`
  （thesis 6 件で同一）。oldtree `f05ec52` は別版（SHA256 `67b8dee…`, git blob `f101978`）を **snapshot 内に同梱**
  （`oldtree_f05ec52_20260512/cugraph_bc_mini/CMakeLists.txt`; SOURCE_MANIFEST.tsv で SHA256 照合済）。

## 参照するスナップショット / ターゲット

| Snapshot | UsedByTarget |
|:--|:--|
| small_correctness_20260712 | run_benchmark (Stage1 libcugraph_bc_mini.a + Stage2 link) |
| phase_def_block_20260710 | run_benchmark |
| memory_correctness_20260712 | run_benchmark |
| memory_diagnostic_20260713 | run_benchmark |
| memory_correctness_oom_20260712 | run_benchmark |
| memory_correctness_failfast_20260712 | run_benchmark |
| oldtree_f05ec52_20260512 | cuGraph baseline（`../../cugraph` 参照; mini CMake は snapshot 内同梱; 保持 UM 実験は gpu_opt のみで cuGraph 非使用） |

## ビルド

Stage 1（`cugraph_bc_mini/CMakeLists.txt` + `third_party/cugraph`）で
`cugraph_bc_mini/build/libcugraph_bc_mini.a` を生成 → `run_benchmark` が IMPORTED static lib として link。
詳細は各 snapshot の `BUILD_ENV.md` と `scripts/build_cugraph_bc_mini.sh`。

## 検証

```bash
cd code_snapshots/_dependencies/cugraph_bc_subset_20260710
sha256sum -c SHA256SUMS
```
`SOURCE_MANIFEST.tsv` の各 SHA256 は commit `88faffa` の対応 git blob と一致（抽出時照合済; 全 checkpoint 同一内容）。
"""

BW_README = """# Dependency: bandwidth_tool_20260710

**用途**: `bandwidth_benchmark` ターゲット（HBM3 / NVLink-C2C 帯域計測; profiling 実験）。

## 内容

- `tools/bandwidth_benchmark.cu` — 帯域計測ツール（`CUDA::cudart` のみに依存, cuGraph 非依存）。
  blob = `c0d4945…`（thesis 6 checkpoint で同一）。

主 `CMakeLists.txt` は `add_executable(bandwidth_benchmark tools/bandwidth_benchmark.cu)` を
**無条件**に定義するため、Stage 2 の configure には本ファイルの存在が必要（実行は profiling のみ）。

## 参照するスナップショット / ターゲット

| Snapshot | UsedByTarget |
|:--|:--|
| small_correctness_20260712 | bandwidth_benchmark (Stage2 configure) |
| phase_def_block_20260710 | bandwidth_benchmark (profiling job 2359175 で実行) |
| memory_correctness_20260712 | bandwidth_benchmark (Stage2 configure) |
| memory_diagnostic_20260713 | bandwidth_benchmark (Stage2 configure) |
| memory_correctness_oom_20260712 | bandwidth_benchmark (Stage2 configure) |
| memory_correctness_failfast_20260712 | bandwidth_benchmark (Stage2 configure) |

oldtree `f05ec52` は本ターゲットを持たない（該当なし）。

## 検証

```bash
cd code_snapshots/_dependencies/bandwidth_tool_20260710
sha256sum -c SHA256SUMS
```
"""

def main():
    r1 = build_cugraph_dep()
    n1 = write_manifests(r1, DEP_CUGRAPH, cugraph_src)
    with open(os.path.join(r1, "README.md"), 'w') as f:
        f.write(CUGRAPH_README)

    r2 = build_bw_dep()
    n2 = write_manifests(r2, DEP_BW, bw_src)
    with open(os.path.join(r2, "README.md"), 'w') as f:
        f.write(BW_README)

    print(f"{DEP_CUGRAPH}: {n1} files")
    print(f"{DEP_BW}: {n2} files")

if __name__ == "__main__":
    main()
