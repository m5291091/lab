#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最終スピードアップ表の生成 (Phase D + legacy + PathMerge 掃引のマージ)。

3 種類の実測値をマージして卒論用の最終表を生成する:

  1. 提案手法 (常時 block, Phase D 再計測)  : build_miyabi/result_benchmark_*/results.tsv
  2. PathMerge 既定 (batch 64, legacy 実測) : legacy_results_miyabi/result_paper/{medium,large}/results_no_gpu_opt.tsv
  3. PathMerge tuned (掃引の最良バッチ)     : build_miyabi/result_pathmerge_sweep_*/pathmerge_sweep_results.tsv

speedup は 2 列を併記する:
  - vs PathMerge 既定  = (PathMerge 既定 中央値) / (提案手法 中央値)
  - vs PathMerge tuned = (PathMerge 最良バッチ 中央値) / (提案手法 中央値)

集計は全て中央値 (median)。比率からの逆算値は一切用いず、実測 TSV のみを参照する。
"""
import csv
import glob
import os
import statistics as st
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _med(times):
    return st.median(times) if times else None


def load_times(tsv_path, impl, graph, time_col=5, impl_col=0, graph_col=1):
    """指定 TSV から (impl, graph) の Time_sec 列を全試行分読む。"""
    xs = []
    if not os.path.exists(tsv_path):
        return xs
    with open(tsv_path) as f:
        for r in csv.reader(f, delimiter="\t"):
            if len(r) <= max(time_col, impl_col, graph_col):
                continue
            if r[impl_col] == impl and r[graph_col] == graph:
                try:
                    xs.append(float(r[time_col]))
                except ValueError:
                    pass
    return xs


def load_sweep_batches(graph):
    """全 result_pathmerge_sweep_* から graph の {batch:[times]} を集約。

    Config 列は 'PathMerge_b<N>'。clamp された場合でもラベルは要求バッチ名のまま
    なので、そのラベル単位で集約する (実効バッチはログ側に記録)。
    """
    batches = {}
    pattern = os.path.join(PROJECT, "build_miyabi", "result_pathmerge_sweep_*",
                           "pathmerge_sweep_results.tsv")
    for tsv in sorted(glob.glob(pattern)):
        with open(tsv) as f:
            for r in csv.reader(f, delimiter="\t"):
                if len(r) < 5 or r[0] == "Config":
                    continue
                cfg, g = r[0], r[1]
                if g != graph or not cfg.startswith("PathMerge_b"):
                    continue
                try:
                    b = int(cfg.split("_b")[1])
                    t = float(r[3])
                except (ValueError, IndexError):
                    continue
                batches.setdefault(b, []).append(t)
    return batches


def best_batch(graph):
    """掃引の最良バッチを (batch, median_time) で返す。無ければ None。"""
    batches = load_sweep_batches(graph)
    if not batches:
        return None
    scored = [(b, _med(ts)) for b, ts in batches.items() if ts]
    scored = [(b, m) for b, m in scored if m is not None]
    if not scored:
        return None
    return min(scored, key=lambda x: x[1])


# ---- 提案手法 (常時 block, Phase D) の results.tsv パス ----
PHASE_D = {
    "email-EuAll": "build_miyabi/result_benchmark_20260711_005425_2357334/results.tsv",
    "roadNet-PA":  "build_miyabi/result_benchmark_20260711_005509_2357335/results.tsv",
    "roadNet-TX":  "build_miyabi/result_benchmark_20260711_005551_2357336/results.tsv",
    "roadNet-CA":  "build_miyabi/result_benchmark_20260711_012956_2357337/results.tsv",
}
# ---- legacy 実測 (PathMerge 既定 b64, および旧 shared 経路の提案手法) ----
LEGACY = {
    "email-EuAll": "legacy_results_miyabi/result_paper/medium/results_no_gpu_opt.tsv",
    "roadNet-PA":  "legacy_results_miyabi/result_paper/large/results_no_gpu_opt.tsv",
    "roadNet-TX":  "legacy_results_miyabi/result_paper/large/results_no_gpu_opt.tsv",
    "roadNet-CA":  "legacy_results_miyabi/result_paper/large/results_no_gpu_opt.tsv",
}
GRAPH_META = {
    "email-EuAll": (265009, 364481, "medium"),
    "roadNet-PA":  (1088092, 1541898, "large"),
    "roadNet-TX":  (1379917, 1921660, "large"),
    "roadNet-CA":  (1965206, 2766607, "large"),
}
PROPOSED = "GPU_Opt"  # フラッグシップ (UM)。block 再計測値。


def collect():
    rows = {}
    for g in GRAPH_META:
        prop = _med(load_times(os.path.join(PROJECT, PHASE_D[g]), PROPOSED, g))
        prop_pure = _med(load_times(os.path.join(PROJECT, PHASE_D[g]), "GPU_Opt_Pure", g))
        prop_chunk = _med(load_times(os.path.join(PROJECT, PHASE_D[g]), "GPU_Opt_Pure_Chunked", g))
        pm_default = _med(load_times(os.path.join(PROJECT, LEGACY[g]), "PathMerge_BC", g))
        old_shared = _med(load_times(os.path.join(PROJECT, LEGACY[g]), "GPU_Opt_Pure", g))
        bb = best_batch(g)
        # tuned は「掃引最良」と「既定 (b64 legacy)」の速い方 (tuning は悪化させない)
        if bb is not None:
            sweep_b, sweep_t = bb
            if pm_default is not None and pm_default <= sweep_t:
                pm_tuned, tuned_b = pm_default, 64
            else:
                pm_tuned, tuned_b = sweep_t, sweep_b
        else:
            pm_tuned, tuned_b = pm_default, 64
        rows[g] = dict(proposed=prop, proposed_pure=prop_pure, proposed_chunk=prop_chunk,
                       pm_default=pm_default, pm_tuned=pm_tuned, tuned_b=tuned_b,
                       old_shared=old_shared)
    return rows


def sp(base, prop):
    if base is None or prop is None or prop <= 0:
        return None
    return base / prop


def fx(v, unit=""):
    return "—" if v is None else (f"{v:.2f}{unit}")


def emit(rows, tier):
    graphs = [g for g in GRAPH_META if GRAPH_META[g][2] == tier]
    L = []
    label = {"medium": "medium (email-EuAll)", "large": "large (roadNet 3種)"}[tier]
    L.append(f"### {label}")
    L.append("")
    L.append("| グラフ | 提案(block) [s] | PathMerge既定 b64 [s] | PathMerge tuned [s] (batch) | vs 既定 | vs tuned |")
    L.append("|:-----|------:|------:|------:|------:|------:|")
    for g in graphs:
        d = rows[g]
        tuned_s = "—" if d["pm_tuned"] is None else f"{d['pm_tuned']:.2f} (b{d['tuned_b']})"
        L.append(
            f"| {g} | {fx(d['proposed'])} | {fx(d['pm_default'])} | {tuned_s} | "
            f"{fx(sp(d['pm_default'], d['proposed']), '×')} | {fx(sp(d['pm_tuned'], d['proposed']), '×')} |"
        )
    L.append("")
    return "\n".join(L)


def emit_reversal(rows):
    L = ["### 旧表との差分 (BFS カーネル shared→block の逆転)", "",
         "旧提案手法値は shared 経路 (legacy)、新値は常時 block (Phase D 再計測)。",
         "speedup 基準は PathMerge 既定 (b64, legacy 実測) で固定。", "",
         "| グラフ | 旧 提案(shared) [s] | 新 提案(block) [s] | 旧 speedup | 新 speedup | 逆転 |",
         "|:-----|------:|------:|------:|------:|:---:|"]
    for g in GRAPH_META:
        d = rows[g]
        old_sp = sp(d["pm_default"], d["old_shared"])
        new_sp = sp(d["pm_default"], d["proposed"])
        flip = "○" if (old_sp is not None and new_sp is not None and old_sp < 1.0 <= new_sp) else \
               ("↑" if (old_sp is not None and new_sp is not None and new_sp > old_sp) else "—")
        L.append(f"| {g} | {fx(d['old_shared'])} | {fx(d['proposed'])} | "
                 f"{fx(old_sp, '×')} | {fx(new_sp, '×')} | {flip} |")
    L.append("")
    return "\n".join(L)


def emit_sweep_detail():
    """掃引の生データ (batch 別 median) を参考表として出す。"""
    L = ["### PathMerge バッチ掃引 詳細 (batch 別 median 実行時間 [s])", ""]
    graphs = ["roadNet-PA", "roadNet-TX", "roadNet-CA", "email-EuAll", "325557_3216152"]
    for g in graphs:
        batches = load_sweep_batches(g)
        if not batches:
            continue
        items = sorted(batches.items())
        cells = " ".join(f"b{b}={_med(ts):.1f}(n{len(ts)})" for b, ts in items)
        bb = best_batch(g)
        best = f"  → 最良 b{bb[0]}={bb[1]:.1f}s" if bb else ""
        L.append(f"- **{g}**: {cells}{best}")
    L.append("")
    return "\n".join(L)


def main():
    rows = collect()
    out = ["# 最終ベンチマーク表 (提案 block × PathMerge 既定/tuned)", "",
           "集計: 全て中央値 (median)。実測 TSV のみ (比率逆算なし)。",
           "提案手法は GPU_Opt (UM, 常時 block)。", "",
           emit(rows, "medium"),
           emit(rows, "large"),
           emit_reversal(rows),
           emit_sweep_detail()]
    text = "\n".join(out)
    print(text)
    outdir = os.path.join(PROJECT, "build_miyabi", "result_final_tables")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "final_speedup_tables.md"), "w") as f:
        f.write(text + "\n")
    print(f"\n[written] {os.path.join(outdir, 'final_speedup_tables.md')}", file=sys.stderr)


if __name__ == "__main__":
    main()
