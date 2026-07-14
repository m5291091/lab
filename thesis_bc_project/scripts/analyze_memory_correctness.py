#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""memory-path correctness 分析 (Gate G2.2) を raw BC vector から再生成する。

`result/correctness/memory_paths/analysis/` に配置した以下を、外部 raw BC
ベクトル (Git 管理外) から**再計算**して出力する。既存数値を変更・再解釈せず、
標準ライブラリのみで同一結果 (byte-identical) を再現するための再現性スクリプト。

出力:
  - run_to_run_comparison.tsv    実行間再現性 (same impl/batch, 混合許容)
  - stress_direct_comparison.tsv stress 直接比較 (GPU_Opt b9792 vs Chunked b16384)
  - six_vertex_detail.tsv        影響 8 頂点の 6 構成 BC 値・判定
  - tolerance_sensitivity.tsv    rel_tol 感度 (1e-6/2e-6/3e-6/1e-5, abs_tol=1e-3 固定)
  - Gate_G2_2_analysis.md        上記を束ねた叙述サマリ

比較規約は `scripts/compare_bc_vectors.py` と同一:
  - 共通 index の有限値のみで誤差計算、iterate は sorted(common)
  - denom = max(|a|, |b|); rel = d/denom (denom>0 のとき)
  - max_abs / max_rel は最初に最大へ到達した index を採用 (strict >)
  - 混合許容: abs_diff <= abs_tol + rel_tol * denom

Gate_G2_3_audit.md は静的コード監査 (人手叙述) であり raw ベクトルから
計算再生成できないため本スクリプトの出力対象外 (数値主張は本出力で相互検証)。

入力 raw ベクトル (既定; raw_data ルート配下, build_miyabi 非依存):
  unsuccessful/oom/memory_paths/.../job_2368269_20260712/pathmerge_b4096.bc.tsv          (pm_269)
  unsuccessful/early_terminated/memory_paths/.../job_2368398_20260712/pathmerge_b4096.bc.tsv     (pm_398)
  unsuccessful/early_terminated/memory_paths/.../job_2368398_20260712/gpu_opt_pure_b1024.bc.tsv  (pure_398)
  correctness/memory_paths/.../job_2368587_20260712/pathmerge_b4096.bc.tsv       (pm_587)
  correctness/memory_paths/.../job_2368587_20260712/gpu_opt_pure_b1024.bc.tsv    (pure_587)
  correctness/memory_paths/.../job_2368587_20260712/gpu_opt_b1024.bc.tsv         (gpu_b1024)
  correctness/memory_paths/.../job_2368587_20260712/gpu_opt_b9792.bc.tsv         (gpu_b9792)
  correctness/memory_paths/.../job_2368587_20260712/gpu_opt_pure_chunked_b1024.bc.tsv   (chunk_b1024)
  correctness/memory_paths/.../job_2368587_20260712/gpu_opt_pure_chunked_b16384.bc.tsv  (chunk_b16384)

使い方:
  python3 scripts/analyze_memory_correctness.py \
      [--raw-data-dir raw_data/correctness/memory_paths] [--graph data/325557_3216152] [--outdir OUT]
  # 旧 build_miyabi から読む場合のみ: --build-dir build_miyabi
"""
import argparse
import hashlib
import math
import os
import sys

ABS_TOL = 1e-3
REL_TOL = 1e-6

J269 = "result_memory_correctness_20260712_204001_2368269.opbs"
J398 = "result_memory_correctness_20260712_211738_2368398.opbs"
J587 = "result_memory_correctness_20260712_220331_2368587.opbs"

# 論理名 -> (jobディレクトリ, ファイル名)  ※ --build-dir 指定時のみ使用する legacy レイアウト
VEC_FILES = {
    "pm_269": (J269, "pathmerge_b4096.bc.tsv"),
    "pm_398": (J398, "pathmerge_b4096.bc.tsv"),
    "pure_398": (J398, "gpu_opt_pure_b1024.bc.tsv"),
    "pm_587": (J587, "pathmerge_b4096.bc.tsv"),
    "pure_587": (J587, "gpu_opt_pure_b1024.bc.tsv"),
    "gpu_b1024": (J587, "gpu_opt_b1024.bc.tsv"),
    "gpu_b9792": (J587, "gpu_opt_b9792.bc.tsv"),
    "chunk_b1024": (J587, "gpu_opt_pure_chunked_b1024.bc.tsv"),
    "chunk_b16384": (J587, "gpu_opt_pure_chunked_b16384.bc.tsv"),
}

# 論理名 -> raw_data ルートからの相対パス（既定; build_miyabi 非依存）
RAW_VEC = {
    "pm_269":   "unsuccessful/oom/memory_paths/325557_3216152/pathmerge_bc/pathmerge_b4096/job_2368269_20260712/pathmerge_b4096.bc.tsv",
    "pm_398":   "unsuccessful/early_terminated/memory_paths/325557_3216152/pathmerge_bc/pathmerge_b4096/job_2368398_20260712/pathmerge_b4096.bc.tsv",
    "pure_398": "unsuccessful/early_terminated/memory_paths/325557_3216152/gpu_opt_pure/pure_b1024/job_2368398_20260712/gpu_opt_pure_b1024.bc.tsv",
    "pm_587":   "correctness/memory_paths/325557_3216152/pathmerge_bc/pathmerge_b4096/job_2368587_20260712/pathmerge_b4096.bc.tsv",
    "pure_587": "correctness/memory_paths/325557_3216152/gpu_opt_pure/pure_b1024/job_2368587_20260712/gpu_opt_pure_b1024.bc.tsv",
    "gpu_b1024":"correctness/memory_paths/325557_3216152/gpu_opt/um_b1024/job_2368587_20260712/gpu_opt_b1024.bc.tsv",
    "gpu_b9792":"correctness/memory_paths/325557_3216152/gpu_opt/um_b9792/job_2368587_20260712/gpu_opt_b9792.bc.tsv",
    "chunk_b1024":"correctness/memory_paths/325557_3216152/gpu_opt_pure_chunked/chunked_b1024/job_2368587_20260712/gpu_opt_pure_chunked_b1024.bc.tsv",
    "chunk_b16384":"correctness/memory_paths/325557_3216152/gpu_opt_pure_chunked/chunked_b16384/job_2368587_20260712/gpu_opt_pure_chunked_b16384.bc.tsv",
}

# Markdown の vector 一覧・感度表の掲載順
VEC_ORDER = ["pm_269", "pm_398", "pm_587", "pure_398", "pure_587",
             "gpu_b1024", "gpu_b9792", "chunk_b1024", "chunk_b16384"]

TOL_PAIRS = [
    ("gpu_b9792 vs gpu_b1024", "gpu_b9792", "gpu_b1024"),
    ("chunk_b16384 vs chunk_b1024", "chunk_b16384", "chunk_b1024"),
    ("gpu_b9792 vs chunk_b16384", "gpu_b9792", "chunk_b16384"),
    ("Pure_b1024 398 vs 587", "pure_398", "pure_587"),
    ("PathMerge 398 vs 587", "pm_398", "pm_587"),
    ("PathMerge 269 vs 587", "pm_269", "pm_587"),
    ("PathMerge(587) vs gpu_b1024", "pm_587", "gpu_b1024"),
    ("PathMerge(587) vs pure_b1024", "pm_587", "pure_587"),
    ("PathMerge(587) vs chunk_b16384", "pm_587", "chunk_b16384"),
]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dump(path):
    """--dump-bc 出力を {idx: value} で返す (compare_bc_vectors.py と同一)。"""
    vals = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                vals[int(parts[0])] = float(parts[1])
            except ValueError:
                continue
    return vals


def max_bc(vals):
    idx = max(vals, key=lambda i: vals[i])
    return idx, vals[idx]


def compare(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL):
    common = sorted(set(a) & set(b))
    only = set(a) ^ set(b)
    nonfinite = (sum(1 for i in a if not math.isfinite(a[i]))
                 + sum(1 for i in b if not math.isfinite(b[i])))
    max_abs = 0.0
    max_abs_idx = None
    max_rel = 0.0
    max_rel_idx = None
    mismatch = 0
    bgt = 0
    blt = 0
    mset = []
    for i in common:
        va, vb = a[i], b[i]
        if not math.isfinite(va) or not math.isfinite(vb):
            continue
        d = abs(va - vb)
        if max_abs_idx is None or d > max_abs:
            max_abs, max_abs_idx = d, i
        denom = max(abs(va), abs(vb))
        rel = d / denom if denom > 0 else 0.0
        if max_rel_idx is None or rel > max_rel:
            max_rel, max_rel_idx = rel, i
        if d > abs_tol + rel_tol * denom:
            mismatch += 1
            mset.append(i)
        if vb > va:
            bgt += 1
        elif vb < va:
            blt += 1
    return dict(common=len(common), missing=len(only), nonfinite=nonfinite,
                max_abs=max_abs, max_abs_idx=max_abs_idx,
                max_rel=max_rel, max_rel_idx=max_rel_idx,
                mismatch=mismatch, bgt=bgt, blt=blt, mset=mset,
                len_a=len(a), len_b=len(b))


def mismatch_count(a, b, rel_tol, abs_tol=ABS_TOL):
    c = 0
    for i in (set(a) & set(b)):
        va, vb = a[i], b[i]
        if not math.isfinite(va) or not math.isfinite(vb):
            continue
        d = abs(va - vb)
        denom = max(abs(va), abs(vb))
        if d > abs_tol + rel_tol * denom:
            c += 1
    return c


def within(x, y):
    d = abs(x - y)
    denom = max(abs(x), abs(y))
    return d <= ABS_TOL + REL_TOL * denom


def row_run_to_run(name, sa, sb, va, vb):
    r = compare(va, vb)
    ma_i, ma_v = max_bc(va)
    mb_i, mb_v = max_bc(vb)
    verdict = "within_mixed_tol" if r["mismatch"] == 0 else "exceeds(%d)" % r["mismatch"]
    return "\t".join([
        name, sa, sb, "yes" if sa == sb else "no",
        str(r["len_a"]), str(r["len_b"]), str(r["missing"]), str(r["nonfinite"]),
        str(r["mismatch"]), "%.6e" % r["max_abs"], str(r["max_abs_idx"]),
        str(va[r["max_abs_idx"]]), str(vb[r["max_abs_idx"]]),
        "%.6e" % r["max_rel"], str(r["max_rel_idx"]),
        str(ma_i), "%.6f" % ma_v, str(mb_i), "%.6f" % mb_v,
        str(r["bgt"]), str(r["blt"]), verdict])


def row_stress(name, sa, sb, va, vb):
    r = compare(va, vb)
    ma_i, ma_v = max_bc(va)
    mb_i, mb_v = max_bc(vb)
    verdict = "within_mixed_tol" if r["mismatch"] == 0 else "exceeds(%d)" % r["mismatch"]
    return "\t".join([
        name, sa, sb, str(r["mismatch"]),
        "%.6e" % r["max_abs"], str(r["max_abs_idx"]),
        str(va[r["max_abs_idx"]]), str(vb[r["max_abs_idx"]]),
        "%.6e" % r["max_rel"], str(r["max_rel_idx"]),
        str(ma_i), "%.6f" % ma_v, str(mb_i), "%.6f" % mb_v,
        str(r["bgt"]), str(r["blt"]), verdict])


def md_line3(name, va, vb):
    r = compare(va, vb)
    verdict = "within_mixed_tol" if r["mismatch"] == 0 else "exceeds(%d)" % r["mismatch"]
    return ("  [%s] byte_identical=no mismatch=%d max_abs=%.6e@%s max_rel=%.6e@%s "
            "B>A=%d B<A=%d verdict=%s") % (
        name, r["mismatch"], r["max_abs"], r["max_abs_idx"],
        r["max_rel"], r["max_rel_idx"], r["bgt"], r["blt"], verdict)


def load_degrees(graph_path):
    with open(graph_path) as f:
        n, _m = map(int, f.readline().split())
        ptr = list(map(int, f.readline().split()))
    return lambda i: ptr[i + 1] - ptr[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-data-dir", default="raw_data/correctness/memory_paths",
                    help="memory-path raw BC ベクトルの親 (既定: raw_data/correctness/memory_paths; "
                         "raw_data ルートを親2階層から導出し unsuccessful/ 配下も解決)")
    ap.add_argument("--build-dir", default=None,
                    help="[legacy] 指定時のみ旧 build_miyabi レイアウトから読む (既定: 使用しない)")
    ap.add_argument("--graph", default="data/325557_3216152",
                    help="CSR グラフ (次数計算用)")
    ap.add_argument("--outdir", default="result/correctness/memory_paths/analysis",
                    help="出力先")
    args = ap.parse_args()

    if args.build_dir:
        # legacy: build_miyabi/<job_dir>/<file>
        paths = {k: os.path.join(args.build_dir, d, fn) for k, (d, fn) in VEC_FILES.items()}
    else:
        # 既定: raw_data ルート (= --raw-data-dir の親2階層) からの相対解決
        raw_root = os.path.normpath(os.path.join(args.raw_data_dir, "..", ".."))
        paths = {k: os.path.join(raw_root, rp) for k, rp in RAW_VEC.items()}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        sys.stderr.write("[ERROR] 入力 raw ベクトルが見つかりません:\n")
        for p in missing:
            sys.stderr.write("  %s\n" % p)
        return 2

    sha = {k: sha256_of(p) for k, p in paths.items()}
    vec = {k: load_dump(p) for k, p in paths.items()}
    graph_sha = sha256_of(args.graph)
    deg = load_degrees(args.graph)

    os.makedirs(args.outdir, exist_ok=True)

    # --- run_to_run_comparison.tsv ---
    hdr_rr = ("comparison\tsha_a\tsha_b\tbyte_identical\tlen_a\tlen_b\tmissing\t"
              "nonfinite\tmismatch\tmax_abs\tmax_abs_idx\tmax_abs_a\tmax_abs_b\t"
              "max_rel\tmax_rel_idx\tmaxbc_a_idx\tmaxbc_a_val\tmaxbc_b_idx\t"
              "maxbc_b_val\tB_gt_A\tB_lt_A\tmixed_verdict")
    with open(os.path.join(args.outdir, "run_to_run_comparison.tsv"), "w") as f:
        f.write(hdr_rr + "\n")
        f.write(row_run_to_run("Pure_b1024 398 vs 587", sha["pure_398"], sha["pure_587"], vec["pure_398"], vec["pure_587"]) + "\n")
        f.write(row_run_to_run("PathMerge 269 vs 398", sha["pm_269"], sha["pm_398"], vec["pm_269"], vec["pm_398"]) + "\n")
        f.write(row_run_to_run("PathMerge 269 vs 587", sha["pm_269"], sha["pm_587"], vec["pm_269"], vec["pm_587"]) + "\n")
        f.write(row_run_to_run("PathMerge 398 vs 587", sha["pm_398"], sha["pm_587"], vec["pm_398"], vec["pm_587"]) + "\n")

    # --- stress_direct_comparison.tsv ---
    hdr_st = ("comparison\tsha_a\tsha_b\tmismatch\tmax_abs\tmax_abs_idx\tmax_abs_a\t"
              "max_abs_b\tmax_rel\tmax_rel_idx\tmaxbc_a_idx\tmaxbc_a_val\t"
              "maxbc_b_idx\tmaxbc_b_val\tB_gt_A\tB_lt_A\tmixed_verdict")
    with open(os.path.join(args.outdir, "stress_direct_comparison.tsv"), "w") as f:
        f.write(hdr_st + "\n")
        f.write(row_stress("gpu_b9792_vs_chunk_b16384", sha["gpu_b9792"], sha["chunk_b16384"], vec["gpu_b9792"], vec["chunk_b16384"]) + "\n")

    # --- tolerance_sensitivity.tsv ---
    with open(os.path.join(args.outdir, "tolerance_sensitivity.tsv"), "w") as f:
        f.write("comparison\t1e-6(official)\t2e-6\t3e-6\t1e-5\n")
        for name, ka, kb in TOL_PAIRS:
            cs = [mismatch_count(vec[ka], vec[kb], rt) for rt in (1e-6, 2e-6, 3e-6, 1e-5)]
            f.write("\t".join([name] + [str(c) for c in cs]) + "\n")

    # --- 影響 index 集合 ---
    s1 = set(compare(vec["gpu_b9792"], vec["gpu_b1024"])["mset"])
    s2 = set(compare(vec["chunk_b16384"], vec["chunk_b1024"])["mset"])
    affected = sorted(s1 | s2)
    pm_diff = set(compare(vec["pm_587"], vec["gpu_b1024"])["mset"])

    # --- six_vertex_detail.tsv ---
    with open(os.path.join(args.outdir, "six_vertex_detail.tsv"), "w") as f:
        f.write("index\tdegree\tgpu_b9792\tgpu_b1024\tchunk_b16384\tchunk_b1024\t"
                "pure_587\tpm_587\tb9792_eq_b16384\tb1024_um_pure_chunk_within_tol\t"
                "in_pathmerge_diff\n")
        for i in affected:
            g9 = vec["gpu_b9792"][i]
            g1 = vec["gpu_b1024"][i]
            c16 = vec["chunk_b16384"][i]
            c1 = vec["chunk_b1024"][i]
            p5 = vec["pure_587"][i]
            pm = vec["pm_587"][i]
            eq = (g9 == c16)
            wtol = within(g1, p5) and within(g1, c1) and within(p5, c1)
            f.write("\t".join([str(i), str(deg(i)), str(g9), str(g1), str(c16),
                               str(c1), str(p5), str(pm), str(eq), str(wtol),
                               str(i in pm_diff)]) + "\n")

    # --- Gate_G2_2_analysis.md ---
    n = len(vec["gpu_b1024"])
    md = []
    md.append("# Gate G2.2 分析 (read-only, raw値無補正; 正式判定 abs_tol=1e-3 rel_tol=1e-6)")
    md.append("graph n=%d  graph_sha256=%s" % (n, graph_sha))
    md.append("")
    md.append("## vector SHA256 と length/NaN,Inf")
    for k in VEC_ORDER:
        nf = sum(1 for i in vec[k] if not math.isfinite(vec[k][i]))
        md.append("  %-14slen=%d nonfinite=%d sha256=%s" % (k, len(vec[k]), nf, sha[k]))
    md.append("")
    md.append("## 3. 実行間再現性 (same impl/batch)")
    md.append(md_line3("Pure_b1024 398 vs 587", vec["pure_398"], vec["pure_587"]))
    md.append(md_line3("PathMerge 269 vs 398", vec["pm_269"], vec["pm_398"]))
    md.append(md_line3("PathMerge 269 vs 587", vec["pm_269"], vec["pm_587"]))
    md.append(md_line3("PathMerge 398 vs 587", vec["pm_398"], vec["pm_587"]))
    md.append("")
    md.append("## 4. stress 直接比較 (11th, 一時のみ): GPU_Opt b9792 vs Chunked b16384")
    rs = compare(vec["gpu_b9792"], vec["chunk_b16384"])
    sv = "exceeds(%d)" % rs["mismatch"] if rs["mismatch"] else "within_mixed_tol"
    md.append("  mismatch=%d max_abs=%.6e@%s (A=%s B=%s) max_rel=%.6e@%s B>A=%d B<A=%d verdict=%s" % (
        rs["mismatch"], rs["max_abs"], rs["max_abs_idx"],
        str(vec["gpu_b9792"][rs["max_abs_idx"]]), str(vec["chunk_b16384"][rs["max_abs_idx"]]),
        rs["max_rel"], rs["max_rel_idx"], rs["bgt"], rs["blt"], sv))
    a9i, a9v = max_bc(vec["gpu_b9792"])
    c6i, c6v = max_bc(vec["chunk_b16384"])
    md.append("  MaxBC A(b9792)=idx%d,%.6f  B(b16384)=idx%d,%.6f" % (a9i, a9v, c6i, c6v))
    md.append("")
    md.append("## 5. 厳格許容超過 index 集合の比較")
    md.append("  S1 (gpu_b9792 vs gpu_b1024) : %d indices = %s" % (len(s1), sorted(s1)))
    md.append("  S2 (chunk_b16384 vs chunk_b1024): %d indices = %s" % (len(s2), sorted(s2)))
    md.append("  S1==S2 ? %s  intersection=%d union=%d" % (s1 == s2, len(s1 & s2), len(s1 | s2)))
    md.append("  PathMerge(pm_587) vs gpu_b1024 差 index 数=%d; union(S1,S2) との重なり=%d / %d" % (
        len(pm_diff), len((s1 | s2) & pm_diff), len(s1 | s2)))
    md.append("")
    md.append("## 6構成の該当 index BC値")
    for i in affected:
        g9 = vec["gpu_b9792"][i]
        g1 = vec["gpu_b1024"][i]
        c16 = vec["chunk_b16384"][i]
        c1 = vec["chunk_b1024"][i]
        p5 = vec["pure_587"][i]
        pm = vec["pm_587"][i]
        md.append("  idx %d (deg %d): gpu_b9792=%s gpu_b1024=%s chunk_b16384=%s chunk_b1024=%s pure_587=%s pm_587=%s" % (
            i, deg(i), str(g9), str(g1), str(c16), str(c1), str(p5), str(pm)))
        md.append("      b9792==b16384? %s  b1024(UM/Pure/Chunk)混合許容内? %s  PathMerge差に含む? %s" % (
            g9 == c16, within(g1, p5) and within(g1, c1) and within(p5, c1), i in pm_diff))
    md.append("")
    md.append("## 6. 許容値感度分析 (abs_tol=1e-3 固定; 正式は rel_tol=1e-6, 他は補助のみ)")
    md.append("  %-34s%13s\t%13s\t%13s\t%13s" % ("", "1e-6(official)", "2e-6", "3e-6", "1e-5"))
    for name, ka, kb in TOL_PAIRS:
        cs = [mismatch_count(vec[ka], vec[kb], rt) for rt in (1e-6, 2e-6, 3e-6, 1e-5)]
        md.append("  %-34s%13s\t%13s\t%13s\t%13s" % (name, cs[0], cs[1], cs[2], cs[3]))
    md.append("")
    with open(os.path.join(args.outdir, "Gate_G2_2_analysis.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    sys.stderr.write("[OK] regenerated 5 files under %s\n" % args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
