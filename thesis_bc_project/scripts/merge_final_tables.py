#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最終スピードアップ表の生成 (Phase D + legacy + PathMerge 掃引のマージ)。

**自己完結**: 入力は全て git 管理下の result/ (legacy baseline を含む) のみを
参照する (git 管理外の build_miyabi/result_* には依存しない)。新規 clone からでも
最終表を再生成できる。job ID を含む一時ディレクトリ名はハードコードしない。

3 種類の実測値をマージして卒論用の最終表を生成する:

  1. 提案手法 (常時 block, Phase D 再計測)  : raw_data/main_performance/proposed_variants/<graph>/_run/*/results.tsv
  2. PathMerge 既定 (batch 64, legacy 実測) : raw_data/main_performance/seven_implementations/legacy_partial/{medium,large}/no_gpu_opt/*/results_no_gpu_opt.tsv
  3. PathMerge tuned (掃引の最良バッチ)     : raw_data/tuning/pathmerge/<graph>/pathmerge_bc/*/*.tsv

speedup は 2 列を併記する:
  - vs PathMerge 既定  = (PathMerge 既定 中央値) / (提案手法 中央値)
  - vs PathMerge tuned = (PathMerge 最良バッチ 中央値) / (提案手法 中央値)

集計は全て中央値 (median)。比率からの逆算値は一切用いず、実測 TSV のみを参照する。
各数値の追跡可能性 (入力ファイル・実装・グラフ・試行数 n・集計方法) は末尾の
「出典・追跡可能性」節に出力する。出力先は result/tables/。
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


SWEEP_SOURCES = {}  # graph -> set(貢献した掃引 TSV の相対パス)


def load_sweep_batches(graph):
    """result/tuning/pathmerge/<graph 別>/*.tsv から graph の {batch:[times]} を集約。

    Config 列は 'PathMerge_b<N>'。clamp された場合でもラベルは要求バッチ名のまま
    なので、そのラベル単位で集約する (実効バッチはログ側に記録)。FAIL/TIMEOUT の
    マーカー行 (Time 列が非数値) は集計から除外する。貢献ファイルは SWEEP_SOURCES に記録。
    """
    batches = {}
    pattern = os.path.join(PROJECT, "raw_data", "tuning", "pathmerge", "*", "pathmerge_bc", "*", "*.tsv")
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
                    t = float(r[3])  # FAIL/TIMEOUT マーカーは ValueError で除外
                except (ValueError, IndexError):
                    continue
                batches.setdefault(b, []).append(t)
                SWEEP_SOURCES.setdefault(graph, set()).add(
                    os.path.relpath(tsv, PROJECT))
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


# ---- 提案手法 (常時 block, Phase D) の results.tsv パス (git 管理下 raw_data/) ----
PHASE_D = {
    "email-EuAll": "raw_data/main_performance/proposed_variants/email-EuAll/_run/job_2357334_20260711/results.tsv",
    "roadNet-PA":  "raw_data/main_performance/proposed_variants/roadNet-PA/_run/job_2357334_20260711/results.tsv",
    "roadNet-TX":  "raw_data/main_performance/proposed_variants/roadNet-TX/_run/job_2357334_20260711/results.tsv",
    "roadNet-CA":  "raw_data/main_performance/proposed_variants/roadNet-CA/_run/job_2357334_20260711/results.tsv",
}
# ---- legacy 実測 (PathMerge 既定 b64, および旧 shared 経路の提案手法) ----
LEGACY = {
    "email-EuAll": "raw_data/main_performance/seven_implementations/legacy_partial/medium/no_gpu_opt/job_notrecorded_legacy/results_no_gpu_opt.tsv",
    "roadNet-PA":  "raw_data/main_performance/seven_implementations/legacy_partial/large/no_gpu_opt/job_notrecorded_legacy/results_no_gpu_opt.tsv",
    "roadNet-TX":  "raw_data/main_performance/seven_implementations/legacy_partial/large/no_gpu_opt/job_notrecorded_legacy/results_no_gpu_opt.tsv",
    "roadNet-CA":  "raw_data/main_performance/seven_implementations/legacy_partial/large/no_gpu_opt/job_notrecorded_legacy/results_no_gpu_opt.tsv",
}
GRAPH_META = {
    "email-EuAll": (265009, 364481, "medium"),
    "roadNet-PA":  (1088092, 1541898, "large"),
    "roadNet-TX":  (1379917, 1921660, "large"),
    "roadNet-CA":  (1965206, 2766607, "large"),
}
PROPOSED = "GPU_Opt"  # フラッグシップ (UM)。block 再計測値。

# ---- PathMerge 掃引 TSV (グラフ別 fixed 必須入力) ----
SWEEP_TSV = {
    "email-EuAll": "raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv",
    "roadNet-PA":  "raw_data/tuning/pathmerge/roadNet-PA/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv",
    "roadNet-TX":  "raw_data/tuning/pathmerge/roadNet-TX/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv",
    "roadNet-CA":  "raw_data/tuning/pathmerge/roadNet-CA/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv",
}
# ---- tuned 採用値をグラフ別に固定 (自動選択・推定なし) ----
#   email = 掃引 b2048 / PA・TX = 掃引で b64 最適を確認し legacy b64 中央値採用 / CA = 掃引 b32
TUNED_FIXED = {
    "email-EuAll": (2048, "sweep"),
    "roadNet-PA":  (64,   "legacy"),
    "roadNet-TX":  (64,   "legacy"),
    "roadNet-CA":  (32,   "sweep"),
}


def _require(path, desc):
    """必須入力の存在を確認。欠損なら推定・代替せず非0終了 (fail-fast)。"""
    full = os.path.join(PROJECT, path)
    if not os.path.exists(full):
        sys.stderr.write(f"[FATAL] 必須入力が欠損しています ({desc}): {path}\n"
                         f"        推定値・代替値へは切り替えません。実測データを配置してください。\n")
        sys.exit(2)
    return full


def collect():
    rows = {}
    for g in GRAPH_META:
        # --- グラフ別 fixed 必須入力チェック (いずれか欠損で fail-fast, 自動切替なし) ---
        _require(PHASE_D[g],  f"提案 results.tsv {g}")
        _require(SWEEP_TSV[g], f"PathMerge 掃引 TSV {g}")
        _require(LEGACY[g],   f"legacy b64 TSV {g}")

        prop_xs = load_times(os.path.join(PROJECT, PHASE_D[g]), PROPOSED, g)
        prop = _med(prop_xs)
        prop_pure = _med(load_times(os.path.join(PROJECT, PHASE_D[g]), "GPU_Opt_Pure", g))
        prop_chunk = _med(load_times(os.path.join(PROJECT, PHASE_D[g]), "GPU_Opt_Pure_Chunked", g))
        pm_def_xs = load_times(os.path.join(PROJECT, LEGACY[g]), "PathMerge_BC", g)
        pm_default = _med(pm_def_xs)
        old_shared = _med(load_times(os.path.join(PROJECT, LEGACY[g]), "GPU_Opt_Pure", g))

        # --- tuned は TUNED_FIXED に従い固定採用 (自動選択・推定なし) ---
        tuned_b, tuned_from = TUNED_FIXED[g]
        sweep_batches = load_sweep_batches(g)
        if tuned_b not in sweep_batches or not sweep_batches[tuned_b]:
            sys.stderr.write(f"[FATAL] {g}: 掃引 TSV に固定バッチ b{tuned_b} の実測がありません。\n")
            sys.exit(2)
        sweep_n = len(sweep_batches[tuned_b])
        if tuned_from == "sweep":
            pm_tuned, tuned_src = _med(sweep_batches[tuned_b]), "掃引 (実測)"
        else:  # legacy: PA/TX は掃引で b64 最適を確認したうえで legacy b64 中央値を採用
            bb = best_batch(g)
            if bb is None or bb[0] != tuned_b:
                got = bb[0] if bb else "?"
                sys.stderr.write(f"[FATAL] {g}: 掃引最適が b{tuned_b} ではありません "
                                 f"(実測最適 b{got})。legacy b{tuned_b} 採用の前提が崩れています。\n")
                sys.exit(2)
            if pm_default is None:
                sys.stderr.write(f"[FATAL] {g}: legacy b{tuned_b} 中央値が取得できません。\n")
                sys.exit(2)
            pm_tuned, tuned_src = pm_default, "legacy 既定 (掃引で最適確認)"

        rows[g] = dict(proposed=prop, proposed_pure=prop_pure, proposed_chunk=prop_chunk,
                       pm_default=pm_default, pm_tuned=pm_tuned, tuned_b=tuned_b,
                       old_shared=old_shared,
                       proposed_n=len(prop_xs), pm_default_n=len(pm_def_xs),
                       tuned_src=tuned_src, sweep_n=sweep_n)
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


def emit_provenance(rows):
    """各数値の出典・追跡可能性 (入力ファイル・実装・グラフ・試行数 n・集計方法)。"""
    L = ["### 出典・追跡可能性 (各数値の入力ファイルと集計)", "",
         "集計方法は全て中央値 (median)。入力は git 管理下のみ。", "",
         "| グラフ | 提案(block) 出典 (impl=GPU_Opt, n) | PathMerge既定 出典 (impl=PathMerge_BC, n) | tuned 出典 (batch, n) |",
         "|:-----|:-----|:-----|:-----|"]
    for g in GRAPH_META:
        d = rows[g]
        sweep_files = sorted(SWEEP_SOURCES.get(g, []))
        sweep_str = "; ".join(sweep_files) if sweep_files else "(掃引 TSV なし)"
        L.append(
            f"| {g} | `{PHASE_D[g]}` (n={d['proposed_n']}) | "
            f"`{LEGACY[g]}` (n={d['pm_default_n']}) | "
            f"{d['tuned_src']} b{d['tuned_b']} (n={d['sweep_n']}); {sweep_str} |")
    L.append("")
    L.append("- **提案手法 (block)**: `raw_data/main_performance/proposed_variants/<graph>/_run/*/results.tsv` の "
             "`GPU_Opt` 行 Time_sec 中央値。")
    L.append("- **PathMerge 既定 (b64)**: `raw_data/main_performance/seven_implementations/legacy_partial/{medium,large}/"
             "no_gpu_opt/*/results_no_gpu_opt.tsv` の `PathMerge_BC` 行 Time_sec 中央値。")
    L.append("- **PathMerge tuned**: `raw_data/tuning/pathmerge/<graph>/pathmerge_bc/*/*.tsv` の掃引実測。"
             "採用値はグラフ別に固定 (email=掃引 b2048 / roadNet-PA・TX=掃引で b64 最適を確認し legacy b64 中央値 / "
             "roadNet-CA=掃引 b32)。**現在の主要4グラフは全て実測であり、推定値は使用していない。**")
    L.append("")
    return "\n".join(L)


def main():
    rows = collect()
    out = ["# 最終ベンチマーク表 (提案 block × PathMerge 既定/tuned)", "",
           "集計: 全て中央値 (median)。実測 TSV のみ (比率逆算なし)。",
           "入力は git 管理下の `raw_data/` (raw) と `result/` (派生) のみ "
           "(build_miyabi 非依存・新規 clone から再生成可能)。",
           "提案手法は GPU_Opt (UM, 常時 block)。", "",
           emit(rows, "medium"),
           emit(rows, "large"),
           emit_reversal(rows),
           emit_sweep_detail(),
           emit_provenance(rows)]
    text = "\n".join(out)
    print(text)
    outdir = os.path.join(PROJECT, "result", "tables")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "final_speedup_tables.md"), "w") as f:
        f.write(text + "\n")
    print(f"\n[written] {os.path.join(outdir, 'final_speedup_tables.md')}", file=sys.stderr)


if __name__ == "__main__":
    main()
