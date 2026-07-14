#!/usr/bin/env python3
"""
ベンチマーク結果の自動サマリ生成スクリプト。

run_benchmark_full.sh が出力する results.tsv + phase_timing.log を読み込み、
以下のファイルを自動生成する:

  1. summary_table.md     — Markdown テーブル (実行時間 + GTEPS)
  2. speedup_table.md     — pathmerge_bc を基準としたスピードアップ表
  3. summary_table.tsv    — タブ区切りテーブル (Excel/スプレッドシート向け)
  4. phase_table.md       — BFS/Backward フェーズ別テーブル
  5. correctness.md       — 正確性検証 (max BC 値の一致確認)

Usage:
  python3 summarize_benchmark.py results.tsv [output_dir]
"""

import sys
import csv
import os
import re
from collections import OrderedDict, defaultdict

# ============================================================
# グラフの表示順序 (ノード数昇順)
# ============================================================
GRAPH_ORDER = [
    "benchmark_7000_41459",
    "benchmark_11023_62184",
    "random",
    "56438_300801",
    "325557_3216152",
    "amazon0302",
    "email-EuAll",
    "web-NotreDame",
    "web-Stanford",
    "amazon0505",
    "web-Google",
    "roadNet-PA",
    "roadNet-TX",
    "roadNet-CA",
]

# グラフの短縮名
GRAPH_SHORT = {
    "benchmark_7000_41459": "bench_7K",
    "benchmark_11023_62184": "bench_11K",
    "random": "random_32K",
    "56438_300801": "synth_56K",
    "325557_3216152": "synth_326K",
    "amazon0302": "amazon0302",
    "email-EuAll": "email-EuAll",
    "web-NotreDame": "web-ND",
    "web-Stanford": "web-Stanford",
    "amazon0505": "amazon0505",
    "web-Google": "web-Google",
    "roadNet-PA": "roadNet-PA",
    "roadNet-TX": "roadNet-TX",
    "roadNet-CA": "roadNet-CA",
}

# グラフのノード数・辺数 (scaling 分析用)
GRAPH_META = {
    "benchmark_7000_41459": (7000, 41459),
    "benchmark_11023_62184": (11023, 62184),
    "random": (32212, 101805),
    "56438_300801": (56438, 300801),
    "325557_3216152": (325557, 3216152),
    "amazon0302": (262111, 899792),
    "email-EuAll": (265009, 418956),
    "web-NotreDame": (325729, 1469679),
    "web-Stanford": (281903, 2312497),
    "amazon0505": (410236, 2439437),
    "web-Google": (875713, 5105039),
    "roadNet-PA": (1088092, 1541898),
    "roadNet-TX": (1379917, 1921660),
    "roadNet-CA": (1965206, 2766607),
}

# 手法の表示順序
METHOD_ORDER = [
    "Sequential",
    "OMP",
    "cuGraph_BC",
    "GPU_Opt",
    "GPU_Opt_Pure",
    "PathMerge_BC",
]

# brandes_runner の出力名 → 表示名の正規化
METHOD_NORMALIZE = {
    "sequential": "Sequential",
    "Sequential": "Sequential",
    "omp": "OMP",
    "OMP": "OMP",
    "OpenMP": "OMP",
    "cugraph_bc": "cuGraph_BC",
    "cuGraph_BC": "cuGraph_BC",
    "gpu_opt": "GPU_Opt",
    "GPU_Opt": "GPU_Opt",
    "gpu_opt_pure": "GPU_Opt_Pure",
    "GPU_Opt_Pure": "GPU_Opt_Pure",
    "pathmerge_bc": "PathMerge_BC",
    "PathMerge_BC": "PathMerge_BC",
}


def load_results(tsv_path):
    """TSV ファイルを読み込み、(method, graph) → {time, gteps, nodes, edges, trials} の辞書を返す。
    複数試行がある場合は平均値を使用。"""
    raw = defaultdict(list)  # (method, graph) → [(time, gteps, nodes, edges), ...]
    with open(tsv_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            # 新形式: Impl Graph Nodes Edges Trial Time GTEPS (7列)
            # 旧形式: Impl Graph Time GTEPS (4列)
            if len(row) >= 7:
                impl, graph = row[0], row[1]
                nodes_str, edges_str = row[2], row[3]
                time_str, gteps_str = row[5], row[6]
            else:
                impl, graph, time_str, gteps_str = row[0], row[1], row[2], row[3]
                nodes_str, edges_str = "", ""

            if impl == "Implementation":
                continue
            if time_str in ("FAIL", "TIMEOUT"):
                continue
            method = METHOD_NORMALIZE.get(impl, impl)
            try:
                time_val = float(time_str)
                gteps_val = float(gteps_str)
            except ValueError:
                continue
            nodes = int(nodes_str) if nodes_str else GRAPH_META.get(graph, (0, 0))[0]
            edges = int(edges_str) if edges_str else GRAPH_META.get(graph, (0, 0))[1]
            raw[(method, graph)].append((time_val, gteps_val, nodes, edges))

    results = {}
    for (method, graph), trials in raw.items():
        n = len(trials)
        avg_time = sum(t[0] for t in trials) / n
        avg_gteps = sum(t[1] for t in trials) / n
        nodes, edges = trials[0][2], trials[0][3]
        # 標準偏差 (n>1)
        if n > 1:
            import math
            std_time = math.sqrt(sum((t[0] - avg_time) ** 2 for t in trials) / (n - 1))
        else:
            std_time = 0.0
        results[(method, graph)] = {
            "time": avg_time,
            "gteps": avg_gteps,
            "nodes": nodes,
            "edges": edges,
            "trials": n,
            "std_time": std_time,
        }
    return results


def get_graphs_in_results(results):
    """結果に含まれるグラフを GRAPH_ORDER の順序で返す。"""
    seen = set(g for (_, g) in results.keys())
    ordered = [g for g in GRAPH_ORDER if g in seen]
    # GRAPH_ORDER にないグラフも末尾に追加
    extra = sorted(seen - set(ordered))
    return ordered + extra


def get_methods_in_results(results):
    """結果に含まれる手法を METHOD_ORDER の順序で返す。"""
    seen = set(m for (m, _) in results.keys())
    ordered = [m for m in METHOD_ORDER if m in seen]
    extra = sorted(seen - set(ordered))
    return ordered + extra


def fmt_time(val):
    """実行時間を見やすくフォーマット。"""
    if val is None:
        return "—"
    if val < 0.01:
        return f"{val:.6f}"
    if val < 10:
        return f"{val:.4f}"
    if val < 100:
        return f"{val:.2f}"
    return f"{val:.1f}"


def fmt_gteps(val):
    if val is None:
        return "—"
    return f"{val:.4f}"


def fmt_speedup(val):
    if val is None:
        return "—"
    return f"{val:.2f}×"


def generate_time_table(results, graphs, methods):
    """実行時間テーブル (Markdown)。"""
    lines = []
    lines.append("# ベンチマーク結果: 実行時間 (秒)")
    lines.append("")

    # ヘッダ
    short = [GRAPH_SHORT.get(g, g) for g in graphs]
    header = "| 手法 | " + " | ".join(short) + " |"
    sep = "|:-----|" + "|".join(["------:" for _ in graphs]) + "|"
    lines.append(header)
    lines.append(sep)

    for method in methods:
        cols = []
        for graph in graphs:
            r = results.get((method, graph))
            if r:
                cols.append(fmt_time(r["time"]))
            else:
                cols.append("—")
        lines.append(f"| {method} | " + " | ".join(cols) + " |")

    return "\n".join(lines)


def generate_gteps_table(results, graphs, methods):
    """GTEPS テーブル (Markdown)。"""
    lines = []
    lines.append("")
    lines.append("# ベンチマーク結果: GTEPS (高いほど高速)")
    lines.append("")

    short = [GRAPH_SHORT.get(g, g) for g in graphs]
    header = "| 手法 | " + " | ".join(short) + " |"
    sep = "|:-----|" + "|".join(["------:" for _ in graphs]) + "|"
    lines.append(header)
    lines.append(sep)

    for method in methods:
        cols = []
        for graph in graphs:
            r = results.get((method, graph))
            if r:
                cols.append(fmt_gteps(r["gteps"]))
            else:
                cols.append("—")
        lines.append(f"| {method} | " + " | ".join(cols) + " |")

    return "\n".join(lines)


def generate_speedup_table(results, graphs, methods, baseline="PathMerge_BC"):
    """baseline に対するスピードアップ表 (Markdown)。"""
    lines = []
    lines.append("")
    lines.append(f"# スピードアップ (vs {baseline})")
    lines.append("")

    # baseline が存在するグラフのみ対象
    valid_graphs = [g for g in graphs if (baseline, g) in results]
    if not valid_graphs:
        lines.append(f"(ベースライン {baseline} の結果がありません)")
        return "\n".join(lines)

    short = [GRAPH_SHORT.get(g, g) for g in valid_graphs]
    header = "| 手法 | " + " | ".join(short) + " |"
    sep = "|:-----|" + "|".join(["------:" for _ in valid_graphs]) + "|"
    lines.append(header)
    lines.append(sep)

    for method in methods:
        cols = []
        for graph in valid_graphs:
            r = results.get((method, graph))
            b = results.get((baseline, graph))
            if r and b and r["time"] > 0:
                speedup = b["time"] / r["time"]
                cols.append(fmt_speedup(speedup))
            else:
                cols.append("—")
        lines.append(f"| {method} | " + " | ".join(cols) + " |")

    return "\n".join(lines)


def generate_per_graph_summary(results, graphs, methods):
    """グラフごとの縦型サマリ。"""
    lines = []
    lines.append("")
    lines.append("# グラフ別詳細結果")
    lines.append("")

    for graph in graphs:
        short = GRAPH_SHORT.get(graph, graph)
        lines.append(f"## {short} ({graph})")
        lines.append("")
        lines.append("| 手法 | 実行時間 (秒) | GTEPS | vs PathMerge_BC |")
        lines.append("|:-----|----------:|------:|----------------:|")

        baseline = results.get(("PathMerge_BC", graph))
        for method in methods:
            r = results.get((method, graph))
            if r:
                time_s = fmt_time(r["time"])
                gteps_s = fmt_gteps(r["gteps"])
                if baseline and r["time"] > 0:
                    sp = baseline["time"] / r["time"]
                    sp_s = fmt_speedup(sp)
                else:
                    sp_s = "—"
                lines.append(f"| {method} | {time_s} | {gteps_s} | {sp_s} |")

        lines.append("")

    return "\n".join(lines)


def generate_tsv_summary(results, graphs, methods):
    """TSV 形式のサマリ (Excel/スプレッドシート向け)。"""
    lines = []
    short = [GRAPH_SHORT.get(g, g) for g in graphs]

    # 実行時間
    lines.append("# 実行時間 (秒)")
    lines.append("Method\t" + "\t".join(short))
    for method in methods:
        cols = []
        for graph in graphs:
            r = results.get((method, graph))
            cols.append(f"{r['time']:.6f}" if r else "")
        lines.append(f"{method}\t" + "\t".join(cols))

    lines.append("")
    lines.append("# GTEPS")
    lines.append("Method\t" + "\t".join(short))
    for method in methods:
        cols = []
        for graph in graphs:
            r = results.get((method, graph))
            cols.append(f"{r['gteps']:.4f}" if r else "")
        lines.append(f"{method}\t" + "\t".join(cols))

    return "\n".join(lines)


def parse_phase_timing(log_path):
    """phase_timing.log から BFS/Backward 時間を抽出。
    Returns: {(impl_display_name, graph): {"bfs": float, "backward": float}}
    """
    phases = {}
    if not os.path.isfile(log_path):
        return phases

    current_impl = None
    current_graph = None

    with open(log_path, "r") as f:
        for line in f:
            # "Running: GPU_Opt on benchmark_7000_41459..."
            m = re.match(r"Running:\s+(\S+)\s+on\s+(\S+)\.\.\.", line)
            if m:
                current_impl = METHOD_NORMALIZE.get(m.group(1), m.group(1))
                current_graph = m.group(2).rstrip(".")
                continue

            # "[GPU Phase] BFS: 0.0215 sec, Backward: 0.0213 sec"
            # "[Phase] BFS: 1.234 sec, Backward: 5.678 sec"
            m = re.search(r"BFS:\s+([\d.]+)\s+sec.*Backward:\s+([\d.]+)\s+sec", line)
            if m and current_impl and current_graph:
                bfs_t = float(m.group(1))
                bwd_t = float(m.group(2))
                phases[(current_impl, current_graph)] = {
                    "bfs": bfs_t,
                    "backward": bwd_t,
                }

    return phases


def generate_phase_table(phases, graphs, methods):
    """BFS / Backward フェーズ別テーブル (Markdown)。"""
    lines = []
    lines.append("# フェーズ別タイミング (BFS / Backward)")
    lines.append("")

    valid_graphs = [g for g in graphs
                    if any((m, g) in phases for m in methods)]
    if not valid_graphs:
        lines.append("(フェーズ計測データがありません)")
        return "\n".join(lines)

    short = [GRAPH_SHORT.get(g, g) for g in valid_graphs]
    # 2段ヘッダ: BFS / Backward
    header1 = "| 手法 |"
    header2 = "|:-----|"
    for s in short:
        header1 += f" {s} (BFS) | {s} (Bwd) |"
        header2 += "------:|------:|"
    lines.append(header1)
    lines.append(header2)

    for method in methods:
        cols = [f" {method} "]
        for graph in valid_graphs:
            p = phases.get((method, graph))
            if p:
                cols.append(f" {p['bfs']:.4f} ")
                cols.append(f" {p['backward']:.4f} ")
            else:
                cols.append(" — ")
                cols.append(" — ")
        lines.append("|" + "|".join(cols) + "|")

    return "\n".join(lines)


def load_max_bc(maxbc_path):
    """max_bc.tsv を読み込み、(method, graph) → (index, value) を返す。"""
    data = {}
    if not os.path.isfile(maxbc_path):
        return data
    with open(maxbc_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 4 or row[0] == "Implementation":
                continue
            impl = METHOD_NORMALIZE.get(row[0], row[0])
            graph = row[1]
            try:
                idx = int(row[2])
                val = float(row[3])
            except ValueError:
                continue
            data[(impl, graph)] = (idx, val)
    return data


def generate_correctness_table(max_bc_data, graphs, methods):
    """正確性検証テーブル (Markdown)。"""
    lines = []
    lines.append("# 正確性検証 (max BC 値)")
    lines.append("")

    valid_graphs = [g for g in graphs
                    if sum(1 for m in methods if (m, g) in max_bc_data) >= 2]
    if not valid_graphs:
        lines.append("(比較可能なデータがありません)")
        return "\n".join(lines)

    for graph in valid_graphs:
        short = GRAPH_SHORT.get(graph, graph)
        lines.append(f"## {short}")
        lines.append("")
        lines.append("| 手法 | Max BC Index | Max BC Value | 差分 (%) |")
        lines.append("|:-----|:-----------|-------------:|---------:|")

        # baseline: 最初に見つかった手法
        baseline_val = None
        for method in methods:
            d = max_bc_data.get((method, graph))
            if d:
                baseline_val = d[1]
                break

        for method in methods:
            d = max_bc_data.get((method, graph))
            if d:
                idx, val = d
                if baseline_val and baseline_val > 0:
                    diff_pct = abs(val - baseline_val) / baseline_val * 100
                    diff_s = f"{diff_pct:.6f}%" if diff_pct > 0 else "0 (基準)"
                else:
                    diff_s = "—"
                lines.append(f"| {method} | {idx} | {val:.2f} | {diff_s} |")

        lines.append("")

    return "\n".join(lines)


def generate_scaling_table(results, graphs, methods):
    """スケーリングテーブル: nodes, edges, n*m, GTEPS per method。"""
    lines = []
    lines.append("# スケーリング分析 (GTEPS vs グラフサイズ)")
    lines.append("")

    method_headers = " | ".join(m for m in methods)
    lines.append(f"| グラフ | Nodes | Edges | n×m | {method_headers} |")

    sep = "|:------|------:|------:|------:|"
    sep += "".join("------:|" for _ in methods)
    lines.append(sep)

    for graph in graphs:
        r0 = None
        for m in methods:
            r0 = results.get((m, graph))
            if r0:
                break
        if not r0:
            continue
        nodes = r0.get("nodes", 0) or GRAPH_META.get(graph, (0, 0))[0]
        edges = r0.get("edges", 0) or GRAPH_META.get(graph, (0, 0))[1]
        nm = nodes * edges
        short = GRAPH_SHORT.get(graph, graph)
        row = f"| {short} | {nodes:,} | {edges:,} | {nm:.2e} |"
        for method in methods:
            r = results.get((method, graph))
            if r:
                row += f" {r['gteps']:.4f} |"
            else:
                row += " — |"
        lines.append(row)

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results.tsv> [output_dir]", file=sys.stderr)
        sys.exit(1)

    tsv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(tsv_path)
    if not output_dir:
        output_dir = "."

    if not os.path.isfile(tsv_path):
        print(f"ERROR: ファイルが見つかりません: {tsv_path}", file=sys.stderr)
        sys.exit(1)

    results = load_results(tsv_path)
    if not results:
        print("ERROR: 有効な結果が0件です", file=sys.stderr)
        sys.exit(1)

    graphs = get_graphs_in_results(results)
    methods = get_methods_in_results(results)

    # 複数試行の有無
    has_multi = any(r.get("trials", 1) > 1 for r in results.values())
    trial_note = ""
    if has_multi:
        trial_note = " (複数試行の平均値)"

    print(f"  結果読み込み: {len(results)} 件 ({len(methods)} 手法 × {len(graphs)} グラフ){trial_note}")

    # --- Phase timing ---
    phase_log_path = os.path.join(output_dir, "phase_timing.log")
    phases = parse_phase_timing(phase_log_path)

    # --- Max BC (correctness) ---
    maxbc_path = os.path.join(output_dir, "max_bc.tsv")
    max_bc_data = load_max_bc(maxbc_path)

    # --- Markdown サマリ ---
    md_parts = [
        generate_time_table(results, graphs, methods),
        generate_gteps_table(results, graphs, methods),
        generate_speedup_table(results, graphs, methods),
        generate_scaling_table(results, graphs, methods),
    ]

    if phases:
        md_parts.append(generate_phase_table(phases, graphs, methods))

    if max_bc_data:
        md_parts.append(generate_correctness_table(max_bc_data, graphs, methods))

    md_parts.append(generate_per_graph_summary(results, graphs, methods))

    md_path = os.path.join(output_dir, "summary_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_parts) + "\n")
    print(f"  → {md_path}")

    # --- スピードアップ表 (単独) ---
    sp_path = os.path.join(output_dir, "speedup_table.md")
    with open(sp_path, "w") as f:
        f.write(generate_speedup_table(results, graphs, methods) + "\n")
    print(f"  → {sp_path}")

    # --- フェーズ別テーブル (単独) ---
    if phases:
        phase_path = os.path.join(output_dir, "phase_table.md")
        with open(phase_path, "w") as f:
            f.write(generate_phase_table(phases, graphs, methods) + "\n")
        print(f"  → {phase_path}")

    # --- 正確性検証 (単独) ---
    if max_bc_data:
        corr_path = os.path.join(output_dir, "correctness.md")
        with open(corr_path, "w") as f:
            f.write(generate_correctness_table(max_bc_data, graphs, methods) + "\n")
        print(f"  → {corr_path}")

    # --- TSV サマリ ---
    tsv_out_path = os.path.join(output_dir, "summary_table.tsv")
    with open(tsv_out_path, "w") as f:
        f.write(generate_tsv_summary(results, graphs, methods) + "\n")
    print(f"  → {tsv_out_path}")

    # --- 画面にもスピードアップ表を表示 ---
    print()
    print(generate_speedup_table(results, graphs, methods))


if __name__ == "__main__":
    main()
