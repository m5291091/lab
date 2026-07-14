#!/usr/bin/env python3
"""
BFS カーネル 2×2 (forced shared / forced block) の集計スクリプト（選択則非依存）。

run_kernel_selection.sh が出力する kernel_selection_results.tsv
(Kernel/Graph/Trial/Time_sec/GTEPS) と kernel_selection_max_bc.tsv を読み込み、
グラフごとに shared / block を **強制実行**した直接比較を集計する。

**選択則には依存しない**。平均次数ヒューリスティクスや「正しい選択/誤選択」の判定は行わず、
forced shared と forced block の実測（中央値・標本標準偏差・試行数・速い側・速度向上・Max BC 一致）
のみを表化する。現行実装は BFS を常に block で実行し、旧実装に存在した平均次数選択則は
現在は使用していない（設計経緯のみ）。

生成物 (output_dir):
  1. kernel_selection_summary.md        — forced shared/block 比較表 + Max BC 一致
  2. kernel_selection_contributions.tsv — 機械可読テーブル

Usage:
  python3 summarize_kernel_selection.py <kernel_selection_results.tsv> [output_dir]
"""

import sys
import os
import csv
from decimal import Decimal, InvalidOperation
from statistics import median, stdev
from collections import defaultdict


def parse_args(argv):
    tsv_path = argv[1] if len(argv) > 1 else None
    output_dir = argv[2] if len(argv) > 2 else None
    return tsv_path, output_dir


def load_results(tsv_path):
    """kernel -> graph -> [time,...] を返す（選択則には使わない）。"""
    times = defaultdict(lambda: defaultdict(list))
    graph_order = []
    with open(tsv_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            if row[0].strip().lower() == "kernel":
                continue
            kernel = row[0].strip().lower()
            graph = row[1]
            # Kernel Graph Trial Time GTEPS (5列) / Kernel Graph Time GTEPS (4列)
            time_str = row[3] if len(row) >= 5 else row[2]
            if time_str in ("FAIL", "TIMEOUT", ""):
                continue
            try:
                t = float(time_str)
            except ValueError:
                continue
            if graph not in graph_order:
                graph_order.append(graph)
            times[kernel][graph].append(t)
    return times, graph_order


def fmt(v, nd=4):
    if v is None:
        return "—"
    if v < 10:
        return f"{v:.{nd}f}"
    if v < 100:
        return f"{v:.2f}"
    return f"{v:.1f}"


def load_max_bc(path):
    """kernel_selection_max_bc.tsv を graph -> {kernel: (idx, Decimal)} で返す。

    Max BC 値は元TSVの10進表現をそのまま保持するため Decimal で読む。
    同一 (kernel, graph) の全試行で (idx, val) が厳密同一であることを検証し、
    差異が存在した場合は非0終了する（任意選択しない）。
    """
    data = defaultdict(dict)
    if not os.path.isfile(path):
        return data
    trials = defaultdict(lambda: defaultdict(list))
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 4 or row[0].strip().lower() == "kernel":
                continue
            kernel = row[0].strip().lower()
            graph = row[1]
            try:
                idx = int(row[2])
                val = Decimal(row[3].strip())
            except (ValueError, InvalidOperation):
                continue
            trials[graph][kernel].append((idx, val))
    for graph, per_kernel in trials.items():
        for kernel, entries in per_kernel.items():
            uniq = set(entries)
            if len(uniq) != 1:
                print(f"ERROR: Max BC が試行間で不一致 (graph={graph}, kernel={kernel}): "
                      f"{sorted(uniq)}", file=sys.stderr)
                sys.exit(1)
            data[graph][kernel] = entries[0]
    return data


def _median(vals):
    return median(vals) if vals else None


def _sstd(vals):
    """標本標準偏差 (sample standard deviation)。n<2 は None。"""
    return stdev(vals) if vals and len(vals) >= 2 else None


def _bc_match(sh_bc, bl_bc):
    if not (sh_bc and bl_bc):
        return None
    return (sh_bc[0] == bl_bc[0]) and (sh_bc[1] == bl_bc[1])


def build_rows(times, graphs, max_bc):
    """forced shared/block の実測のみを集計（選択則非依存）。"""
    rows = []
    for g in graphs:
        sh = times.get("shared", {}).get(g, [])
        bl = times.get("block", {}).get(g, [])
        m_sh, m_bl = _median(sh), _median(bl)
        faster = speedup = None
        if m_sh is not None and m_bl is not None:
            faster = "block" if m_bl <= m_sh else "shared"
            slow, fast = max(m_sh, m_bl), min(m_sh, m_bl)
            speedup = (slow / fast) if fast > 0 else None
        mb = max_bc.get(g, {})
        sh_bc, bl_bc = mb.get("shared"), mb.get("block")
        rows.append({
            "graph": g,
            "m_shared": m_sh, "m_block": m_bl,
            "sd_shared": _sstd(sh), "sd_block": _sstd(bl),
            "n_shared": len(sh), "n_block": len(bl),
            "faster": faster, "speedup": speedup,
            "sh_bc": sh_bc, "bl_bc": bl_bc, "match": _bc_match(sh_bc, bl_bc),
        })
    return rows


def generate_md(rows):
    L = ["# BFS カーネル 2×2 比較（forced shared / forced block・選択則非依存）", ""]
    L.append("shared / block を **強制実行**した直接比較の実測（中央値・標本標準偏差・速度向上・Max BC 一致）。")
    L.append("自動選択則には依存しない。")
    L += ["", "## 実測 (中央値・標本標準偏差)", ""]
    L.append("| グラフ | shared 中央値 (s) | block 中央値 (s) | shared 標本SD | block 標本SD | n(shared) | n(block) | 速い側 | 速度向上 (遅/速) | Max BC 一致 |")
    L.append("|:-----|------:|------:|------:|------:|--:|--:|:----:|------:|:----:|")
    for r in rows:
        sd_sh = "n/a" if r["sd_shared"] is None else f"{r['sd_shared']:.4f}"
        sd_bl = "n/a" if r["sd_block"] is None else f"{r['sd_block']:.4f}"
        sp = "—" if r["speedup"] is None else f"{r['speedup']:.2f}×"
        mt = "—" if r["match"] is None else ("✓" if r["match"] else "✗")
        L.append(f"| {r['graph']} | {fmt(r['m_shared'])} | {fmt(r['m_block'])} | {sd_sh} | {sd_bl} | "
                 f"{r['n_shared']} | {r['n_block']} | {r['faster'] or '—'} | {sp} | {mt} |")

    L += ["", "## Max BC (shared / block)", ""]
    L.append("| グラフ | shared Index | shared Value | block Index | block Value | 一致 |")
    L.append("|:-----|--:|------:|--:|------:|:----:|")
    for r in rows:
        sh, bl = r["sh_bc"], r["bl_bc"]
        si = "—" if not sh else str(sh[0])
        sv = "—" if not sh else f"{sh[1]:.4f}"
        bi = "—" if not bl else str(bl[0])
        bv = "—" if not bl else f"{bl[1]:.4f}"
        mt = "—" if r["match"] is None else ("✓" if r["match"] else "✗")
        L.append(f"| {r['graph']} | {si} | {sv} | {bi} | {bv} | {mt} |")

    L += ["", "## 事実 (forced 比較の結果)", ""]
    for r in rows:
        if r["speedup"] is not None and r["faster"]:
            L.append(f"- {r['graph']}: shared≈{fmt(r['m_shared'])}s / block≈{fmt(r['m_block'])}s → "
                     f"**{r['faster']} が {r['speedup']:.2f}倍高速** "
                     f"(n(shared)={r['n_shared']}, n(block)={r['n_block']})")
    agreed = [r["graph"] for r in rows if r["match"]]
    if agreed:
        L.append(f"- shared と block の Max BC index/value は一致 ({', '.join(agreed)})。")
    L.append("- 本結果は測定した強制比較グラフに限定し、**未測定グラフへ一般化しない**。")
    L += ["", "> 注: 現行実装は BFS カーネルを常に block で実行する。"
          "旧実装には平均次数に基づく自動選択則が存在したが、現在は使用していない。"]
    return "\n".join(L)


def write_tsv(path, rows):
    hdr = ["Graph", "Shared_median_sec", "Block_median_sec",
           "Shared_sample_std", "Block_sample_std", "N_shared", "N_block",
           "Faster", "Speedup",
           "Shared_MaxBC_Index", "Shared_MaxBC_Value",
           "Block_MaxBC_Index", "Block_MaxBC_Value", "MaxBC_Match"]

    def f6(v):
        return "" if v is None else f"{v:.6f}"

    def bc_fmt(v):
        # Max BC は Decimal で保持しているので、元TSVの10進表現を非科学記法でそのまま出力する。
        # float 経由の再フォーマットを避けることで 151395302679.08 → 151395302679.079987 の
        # ような表示誤差を防ぐ。
        return "" if v is None else format(v, "f")

    with open(path, "w") as f:
        f.write("\t".join(hdr) + "\n")
        for r in rows:
            sh, bl = r["sh_bc"], r["bl_bc"]
            f.write("\t".join([
                r["graph"],
                f6(r["m_shared"]), f6(r["m_block"]),
                f6(r["sd_shared"]), f6(r["sd_block"]),
                str(r["n_shared"]), str(r["n_block"]),
                r["faster"] or "",
                "" if r["speedup"] is None else f"{r['speedup']:.6f}",
                "" if not sh else str(sh[0]), "" if not sh else bc_fmt(sh[1]),
                "" if not bl else str(bl[0]), "" if not bl else bc_fmt(bl[1]),
                "" if r["match"] is None else ("yes" if r["match"] else "no"),
            ]) + "\n")


def main():
    tsv_path, output_dir = parse_args(sys.argv)
    if tsv_path is None:
        print(f"Usage: {sys.argv[0]} <kernel_selection_results.tsv> [output_dir]",
              file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(tsv_path):
        print(f"ERROR: ファイルが見つかりません: {tsv_path}", file=sys.stderr)
        sys.exit(1)
    output_dir = output_dir or os.path.dirname(tsv_path) or "."

    times, graphs = load_results(tsv_path)
    if not graphs:
        print("ERROR: 有効な結果が 0 件です", file=sys.stderr)
        sys.exit(1)

    max_bc = load_max_bc(os.path.join(os.path.dirname(tsv_path) or ".",
                                      "kernel_selection_max_bc.tsv"))
    rows = build_rows(times, graphs, max_bc)

    md = generate_md(rows)
    md_path = os.path.join(output_dir, "kernel_selection_summary.md")
    with open(md_path, "w") as f:
        f.write(md + "\n")

    tsv_out = os.path.join(output_dir, "kernel_selection_contributions.tsv")
    write_tsv(tsv_out, rows)

    print(f"  読み込み: {len(graphs)} グラフ", file=sys.stderr)
    print(f"  → {md_path}", file=sys.stderr)
    print(f"  → {tsv_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
