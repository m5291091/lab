#!/usr/bin/env python3
"""
BFS カーネル選択 2×2 の集計スクリプト。

run_kernel_selection.sh が出力する kernel_selection_results.tsv
(Kernel/Graph/Trial/Time_sec/GTEPS) を読み込み、グラフごとに
shared-frontier 版 / block-per-source 版の中央値実行時間を比較する。

gpu_opt (host_um.cu) は選択則 S2「常時 block」を採用 (2×2 実測で全グラフ
block 優位のため)。本スクリプトは shared/block の実測比を表化し、旧則
(avg_deg<5 で shared) が誤選択で失っていた倍率 (= 遅い側/速い側の実行時間比)
を「選択機構の寄与」として可視化する。

生成物 (output_dir):
  1. kernel_selection_summary.md        — 比較表 + 選択機構の寄与 + 正確性
  2. kernel_selection_contributions.tsv — 機械可読テーブル

Usage:
  python3 summarize_kernel_selection.py <kernel_selection_results.tsv> [output_dir] [--data-dir DIR]
"""

import sys
import os
import csv
from statistics import median, pstdev
from collections import defaultdict

# ヒューリスティクスの閾値 (host_um.cu と一致させること)
AVG_DEG_THRESHOLD = 5.0


def parse_args(argv):
    tsv_path = None
    output_dir = None
    data_dir = None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--data-dir":
            i += 1
            data_dir = argv[i]
        elif tsv_path is None:
            tsv_path = a
        elif output_dir is None:
            output_dir = a
        i += 1
    return tsv_path, output_dir, data_dir


def load_results(tsv_path):
    """kernel -> graph -> [time,...], および gteps を返す。"""
    times = defaultdict(lambda: defaultdict(list))
    gteps = defaultdict(lambda: defaultdict(list))
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
            if len(row) >= 5:
                time_str, gteps_str = row[3], row[4]
            else:
                time_str, gteps_str = row[2], row[3]
            if time_str in ("FAIL", "TIMEOUT", ""):
                continue
            try:
                t = float(time_str)
                g = float(gteps_str)
            except ValueError:
                continue
            if graph not in graph_order:
                graph_order.append(graph)
            times[kernel][graph].append(t)
            gteps[kernel][graph].append(g)
    return times, gteps, graph_order


def find_data_dir(start, explicit):
    """data/ ディレクトリを explicit / 祖先探索で見つける。"""
    if explicit and os.path.isdir(explicit):
        return explicit
    cur = os.path.abspath(start)
    for _ in range(8):
        cand = os.path.join(cur, "data")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def graph_avg_deg(graph, data_dir):
    """CSR 1 行目 (n_nodes n_edges) から avg_deg = 2*E/N を算出。見つからなければ None。"""
    if not data_dir:
        return None
    candidates = [
        os.path.join(data_dir, graph),
        os.path.join(data_dir, "snap", graph),
        os.path.join(data_dir, os.path.basename(graph)),
        os.path.join(data_dir, "snap", os.path.basename(graph)),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    first = f.readline().split()
                n, e = int(first[0]), int(first[1])
                if n > 0:
                    return 2.0 * e / n
            except (ValueError, IndexError):
                return None
    return None


def med(vals):
    return median(vals) if vals else None


def fmt(v, nd=4):
    if v is None:
        return "—"
    if v < 10:
        return f"{v:.{nd}f}"
    if v < 100:
        return f"{v:.2f}"
    return f"{v:.1f}"


def load_max_bc(path):
    """kernel_selection_max_bc.tsv を graph -> {kernel: (idx, val)} で返す。"""
    data = defaultdict(dict)
    if not os.path.isfile(path):
        return data
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 4 or row[0].strip().lower() == "kernel":
                continue
            kernel = row[0].strip().lower()
            graph = row[1]
            try:
                idx, val = int(row[2]), float(row[3])
            except ValueError:
                continue
            data[graph][kernel] = (idx, val)
    return data


def build_rows(times, graphs, data_dir):
    rows = []
    for g in graphs:
        t_shared = med(times.get("shared", {}).get(g))
        t_block = med(times.get("block", {}).get(g))
        avg_deg = graph_avg_deg(g, data_dir)
        # 選択則 S2「常時 block」: 2×2 実測で全グラフ block 優位のため常に block を選ぶ。
        heuristic = "block"
        faster = None
        ratio = None
        if t_shared is not None and t_block is not None:
            faster = "shared" if t_shared <= t_block else "block"
            slow, fast = max(t_shared, t_block), min(t_shared, t_block)
            ratio = (slow / fast) if fast > 0 else None
        correct = (heuristic == faster) if (heuristic and faster) else None
        rows.append({
            "graph": g, "avg_deg": avg_deg,
            "t_shared": t_shared, "t_block": t_block,
            "heuristic": heuristic, "faster": faster,
            "ratio": ratio, "correct": correct,
        })
    return rows


def generate_md(rows, times, max_bc):
    lines = ["# BFS カーネル選択 2×2 比較", ""]
    lines.append("選択則 S2「常時 block」: 2×2 実測で全グラフ block 優位のため常に block を選択 "
                 "(旧則 avg_deg < %.1f → shared は棄却)。" % AVG_DEG_THRESHOLD)
    lines.append("")
    lines.append("| グラフ | avg_deg | shared (s) | block (s) | 速い側 | ヒューリスティクス | 正しい選択? | 選択機構の寄与 (遅/速) |")
    lines.append("|:-----|------:|------:|------:|:----:|:----:|:----:|------:|")
    for r in rows:
        correct_s = "—" if r["correct"] is None else ("✓" if r["correct"] else "✗ 誤選択")
        avgd = "—" if r["avg_deg"] is None else f"{r['avg_deg']:.2f}"
        ratio_s = "—" if r["ratio"] is None else f"{r['ratio']:.2f}×"
        lines.append(f"| {r['graph']} | {avgd} | {fmt(r['t_shared'])} | {fmt(r['t_block'])} | "
                     f"{r['faster'] or '—'} | {r['heuristic'] or '—'} | {correct_s} | {ratio_s} |")

    # まとめ
    decided = [r for r in rows if r["correct"] is not None]
    n_ok = sum(1 for r in decided if r["correct"])
    lines += ["", "## まとめ", ""]
    if decided:
        lines.append(f"- ヒューリスティクスが速い側を選べた: **{n_ok}/{len(decided)}** グラフ")
        wrong = [r["graph"] for r in decided if not r["correct"]]
        if wrong:
            lines.append(f"- 誤選択したグラフ: {', '.join(wrong)}")
        ratios = [r["ratio"] for r in decided if r["ratio"]]
        if ratios:
            lines.append(f"- 選択機構の寄与 (誤選択時に失う倍率) の範囲: "
                         f"{min(ratios):.2f}× 〜 {max(ratios):.2f}×")
    else:
        lines.append("- shared/block 両方が揃ったグラフがありません。")

    # 試行数・ばらつき
    lines += ["", "## 試行数と実行時間ばらつき (中央値 ± 標準偏差)", ""]
    lines.append("| グラフ | kernel | 中央値 (s) | 標準偏差 | n |")
    lines.append("|:-----|:----:|------:|------:|--:|")
    for r in rows:
        for kernel in ("shared", "block"):
            vals = times.get(kernel, {}).get(r["graph"])
            if not vals:
                continue
            m = median(vals)
            sd = pstdev(vals) if len(vals) > 1 else 0.0
            lines.append(f"| {r['graph']} | {kernel} | {fmt(m)} | {sd:.4f} | {len(vals)} |")

    # 正確性 (shared vs block の max BC 一致)
    if max_bc:
        lines += ["", "## 正確性 (shared vs block の Max BC 一致)", ""]
        lines.append("| グラフ | shared Max BC | block Max BC | 一致? |")
        lines.append("|:-----|------:|------:|:----:|")
        for g, kv in max_bc.items():
            vs = kv.get("shared")
            vb = kv.get("block")
            if vs and vb:
                match = "✓" if abs(vs[1] - vb[1]) < 1e-6 * max(1.0, abs(vs[1])) else "✗"
                lines.append(f"| {g} | {vs[1]:.4f} | {vb[1]:.4f} | {match} |")
            else:
                s_shared = f"{vs[1]:.4f}" if vs else "—"
                s_block = f"{vb[1]:.4f}" if vb else "—"
                lines.append(f"| {g} | {s_shared} | {s_block} | — |")
    return "\n".join(lines)


def write_tsv(path, rows):
    with open(path, "w") as f:
        f.write("Graph\tAvgDeg\tShared_sec\tBlock_sec\tFaster\tHeuristic\tCorrect\tSelectionRatio\n")
        for r in rows:
            def s(v):
                return "" if v is None else (f"{v:.6f}" if isinstance(v, float) else str(v))
            correct = "" if r["correct"] is None else ("1" if r["correct"] else "0")
            f.write(f"{r['graph']}\t{s(r['avg_deg'])}\t{s(r['t_shared'])}\t{s(r['t_block'])}\t"
                    f"{r['faster'] or ''}\t{r['heuristic'] or ''}\t{correct}\t{s(r['ratio'])}\n")


def main():
    tsv_path, output_dir, data_dir_arg = parse_args(sys.argv)
    if tsv_path is None:
        print(f"Usage: {sys.argv[0]} <kernel_selection_results.tsv> [output_dir] [--data-dir DIR]",
              file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(tsv_path):
        print(f"ERROR: ファイルが見つかりません: {tsv_path}", file=sys.stderr)
        sys.exit(1)
    output_dir = output_dir or os.path.dirname(tsv_path) or "."

    times, gteps, graphs = load_results(tsv_path)
    if not graphs:
        print("ERROR: 有効な結果が 0 件です", file=sys.stderr)
        sys.exit(1)

    data_dir = find_data_dir(os.path.dirname(tsv_path) or ".", data_dir_arg)
    if not data_dir:
        print("  警告: data/ ディレクトリが見つからず avg_deg を判定できません", file=sys.stderr)

    rows = build_rows(times, graphs, data_dir)
    max_bc = load_max_bc(os.path.join(os.path.dirname(tsv_path) or ".",
                                      "kernel_selection_max_bc.tsv"))

    md = generate_md(rows, times, max_bc)
    md_path = os.path.join(output_dir, "kernel_selection_summary.md")
    with open(md_path, "w") as f:
        f.write(md + "\n")
    print(f"  読み込み: {len(graphs)} グラフ")
    print(f"  → {md_path}")

    tsv_out = os.path.join(output_dir, "kernel_selection_contributions.tsv")
    write_tsv(tsv_out, rows)
    print(f"  → {tsv_out}")

    print()
    print(md)


if __name__ == "__main__":
    main()
