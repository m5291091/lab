#!/usr/bin/env python3
"""
アブレーション結果の寄与集計スクリプト。

run_ablation.sh が出力する ablation_results.tsv (Config/Graph/Trial/Time_sec/GTEPS)
を読み込み、提案手法 3 工夫 (H=ハイブリッド BFS / W=Warp 協調蓄積 / A=2 ストリーム
非同期初期化) の寄与を 2 通りの見方で集計する。

生成物 (output_dir):
  1. ablation_summary.md        — 中央値の生値表 + 寄与表 + 交互作用 + フェーズ帰属
  2. ablation_contributions.tsv — 機械可読な寄与テーブル

寄与の 2 つの見方 (どちらも「速くなる倍率」= 1.0 超で有効):
  - 単独寄与 (add-one)     : T(H0W0A0) / T(その工夫だけ ON)
  - 除外寄与 (leave-one-out): T(full から 1 つ OFF) / T(H1W1A1)
両者が乖離する場合は交互作用がある証拠 (フル要因なので主効果も併記)。

任意で ablation.log を解析し、H→BFS 時間 / W→Backward 時間 /
A→wall−(BFS+Backward) の隙間時間 へのフェーズ帰属も出力する。

Usage:
  python3 summarize_ablation.py <ablation_results.tsv> [output_dir]
"""

import sys
import os
import re
import csv
import math
from statistics import median, stdev
from collections import defaultdict

# H/W/A の 3 ビットで構成を表す。tuple(h, w, a) をキーに使う。
FACTORS = ["H", "W", "A"]
FACTOR_NAME = {
    "H": "ハイブリッド BFS",
    "W": "Warp 協調蓄積",
    "A": "2 ストリーム非同期初期化",
}
BASELINE = (0, 0, 0)
FULL = (1, 1, 1)

# 交互作用ありと判定する相対差の閾値
INTERACTION_REL_THRESHOLD = 0.10

CONFIG_RE = re.compile(r"H\s*([01])\s*_?\s*W\s*([01])\s*_?\s*A\s*([01])", re.IGNORECASE)


def parse_config(config_str):
    """'Ablation_H1_W0_A1' 等から (h, w, a) を取り出す。失敗時 None。"""
    m = CONFIG_RE.search(config_str)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def load_results(tsv_path):
    """(h,w,a) -> graph -> [time, ...] の辞書と、出現順のグラフリストを返す。"""
    times = defaultdict(lambda: defaultdict(list))
    gteps = defaultdict(lambda: defaultdict(list))
    graph_order = []
    with open(tsv_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            if row[0].strip().lower() in ("config", "implementation"):
                continue
            cfg = parse_config(row[0])
            if cfg is None:
                continue
            # Config  Graph  Trial  Time_sec  GTEPS  (5列) を想定。
            # 互換: Config Graph Time GTEPS (4列, Trial 無し) も受理。
            graph = row[1]
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
            times[cfg][graph].append(t)
            gteps[cfg][graph].append(g)
    return times, gteps, graph_order


def med_time(times, cfg, graph):
    vals = times.get(cfg, {}).get(graph)
    if not vals:
        return None
    return median(vals)


def fmt(v, nd=4):
    if v is None:
        return "—"
    if v < 10:
        return f"{v:.{nd}f}"
    if v < 100:
        return f"{v:.2f}"
    return f"{v:.1f}"


def fmt_ratio(v):
    return "—" if v is None else f"{v:.3f}×"


def single_on_config(factor):
    """その工夫だけ ON の構成 tuple を返す。"""
    return tuple(1 if f == factor else 0 for f in FACTORS)


def full_minus_config(factor):
    """full から factor だけ OFF の構成 tuple を返す。"""
    return tuple(0 if f == factor else 1 for f in FACTORS)


def safe_ratio(num, den):
    if num is None or den is None or den == 0:
        return None
    return num / den


def geomean(vals):
    vals = [v for v in vals if v is not None and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


# ============================================================
#  生値表
# ============================================================
def config_label(cfg):
    return f"H{cfg[0]}W{cfg[1]}A{cfg[2]}"


def all_configs_sorted():
    cfgs = []
    for h in (0, 1):
        for w in (0, 1):
            for a in (0, 1):
                cfgs.append((h, w, a))
    return cfgs


def generate_raw_table(times, gteps, graphs):
    lines = ["# アブレーション: 中央値 実行時間 (秒)", ""]
    header = "| 構成 | " + " | ".join(graphs) + " |"
    sep = "|:-----|" + "|".join("------:" for _ in graphs) + "|"
    lines += [header, sep]
    for cfg in all_configs_sorted():
        cols = [fmt(med_time(times, cfg, g)) for g in graphs]
        tag = config_label(cfg)
        if cfg == BASELINE:
            tag += " (baseline)"
        elif cfg == FULL:
            tag += " (full)"
        lines.append(f"| {tag} | " + " | ".join(cols) + " |")
    return "\n".join(lines)


# ============================================================
#  寄与表 (add-one / leave-one-out / 主効果 / 交互作用)
# ============================================================
def compute_contributions(times, graphs):
    """graph -> factor -> {add_one, leave_one_out, main_effect, interaction} を返す。"""
    result = {}
    for g in graphs:
        t_base = med_time(times, BASELINE, g)
        t_full = med_time(times, FULL, g)
        per_factor = {}
        for f in FACTORS:
            t_single = med_time(times, single_on_config(f), g)
            t_minus = med_time(times, full_minus_config(f), g)
            add_one = safe_ratio(t_base, t_single)          # T(000)/T(単独ON)
            leave = safe_ratio(t_minus, t_full)             # T(full-1)/T(111)
            # 主効果: 他 2 軸の全組合せで T(F=0)/T(F=1) の幾何平均
            ratios = []
            others = [x for x in FACTORS if x != f]
            for b0 in (0, 1):
                for b1 in (0, 1):
                    cfg0 = {f: 0, others[0]: b0, others[1]: b1}
                    cfg1 = {f: 1, others[0]: b0, others[1]: b1}
                    key0 = tuple(cfg0[x] for x in FACTORS)
                    key1 = tuple(cfg1[x] for x in FACTORS)
                    ratios.append(safe_ratio(med_time(times, key0, g),
                                             med_time(times, key1, g)))
            main_effect = geomean(ratios)
            interaction = None
            if add_one is not None and leave is not None and leave != 0:
                interaction = abs(add_one - leave) / leave
            per_factor[f] = {
                "add_one": add_one,
                "leave_one_out": leave,
                "main_effect": main_effect,
                "interaction": interaction,
            }
        result[g] = per_factor
    return result


def generate_contribution_tables(contrib, graphs):
    lines = []

    lines += ["", "# 単独寄与 (add-one): T(H0W0A0) / T(その工夫だけ ON)", ""]
    lines.append("| 工夫 | " + " | ".join(graphs) + " | 幾何平均 |")
    lines.append("|:-----|" + "|".join("------:" for _ in graphs) + "|------:|")
    for f in FACTORS:
        cols = [fmt_ratio(contrib[g][f]["add_one"]) for g in graphs]
        gm = geomean([contrib[g][f]["add_one"] for g in graphs])
        lines.append(f"| {f} ({FACTOR_NAME[f]}) | " + " | ".join(cols) + f" | {fmt_ratio(gm)} |")

    lines += ["", "# 除外寄与 (leave-one-out): T(full から 1 つ OFF) / T(H1W1A1)", ""]
    lines.append("| 工夫 | " + " | ".join(graphs) + " | 幾何平均 |")
    lines.append("|:-----|" + "|".join("------:" for _ in graphs) + "|------:|")
    for f in FACTORS:
        cols = [fmt_ratio(contrib[g][f]["leave_one_out"]) for g in graphs]
        gm = geomean([contrib[g][f]["leave_one_out"] for g in graphs])
        lines.append(f"| {f} ({FACTOR_NAME[f]}) | " + " | ".join(cols) + f" | {fmt_ratio(gm)} |")

    lines += ["", "# 主効果 (フル要因): 他 2 軸で平均した T(F=0)/T(F=1) の幾何平均", ""]
    lines.append("| 工夫 | " + " | ".join(graphs) + " | 幾何平均 |")
    lines.append("|:-----|" + "|".join("------:" for _ in graphs) + "|------:|")
    for f in FACTORS:
        cols = [fmt_ratio(contrib[g][f]["main_effect"]) for g in graphs]
        gm = geomean([contrib[g][f]["main_effect"] for g in graphs])
        lines.append(f"| {f} ({FACTOR_NAME[f]}) | " + " | ".join(cols) + f" | {fmt_ratio(gm)} |")

    lines += ["", "# 交互作用チェック (単独寄与 vs 除外寄与 の相対差)", ""]
    lines.append(f"閾値 {INTERACTION_REL_THRESHOLD:.0%} 超で「交互作用あり」と判定 "
                 "(単独と最終系で工夫の効き方が異なる = 他工夫との相互作用)。")
    lines.append("")
    lines.append("| 工夫 | グラフ | 単独寄与 | 除外寄与 | 相対差 | 判定 |")
    lines.append("|:-----|:-----|------:|------:|------:|:----:|")
    for f in FACTORS:
        for g in graphs:
            c = contrib[g][f]
            inter = c["interaction"]
            if inter is None:
                continue
            flag = "⚠ 交互作用" if inter > INTERACTION_REL_THRESHOLD else "—"
            lines.append(f"| {f} | {g} | {fmt_ratio(c['add_one'])} | "
                         f"{fmt_ratio(c['leave_one_out'])} | {inter:.1%} | {flag} |")
    return "\n".join(lines)


def write_contributions_tsv(path, contrib, graphs):
    with open(path, "w") as f:
        f.write("Graph\tFactor\tAddOne\tLeaveOneOut\tMainEffect\tInteractionRel\n")
        for g in graphs:
            for fac in FACTORS:
                c = contrib[g][fac]

                def s(v, pct=False):
                    if v is None:
                        return ""
                    return f"{v:.4f}"
                f.write(f"{g}\t{fac}\t{s(c['add_one'])}\t{s(c['leave_one_out'])}\t"
                        f"{s(c['main_effect'])}\t{s(c['interaction'])}\n")


# ============================================================
#  フェーズ帰属 (任意): ablation.log を解析
# ============================================================
PHASE_RE = re.compile(r"\[Ablation Phase\]\s*BFS cum=([\d.]+)\s*s,\s*Backward cum=([\d.]+)\s*s")
ABL_MARK_RE = re.compile(r"\[Ablation\s+H([01])\s+W([01])\s+A([01])\]")
GRAPH_MARK_RE = re.compile(r"===\s*graph=(\S+)\s+trial=(\d+)\s*===")


def parse_ablation_log(log_path):
    """(h,w,a) -> graph -> {'bfs':[...], 'backward':[...]} を返す。失敗時 空 dict。"""
    phases = defaultdict(lambda: defaultdict(lambda: {"bfs": [], "backward": []}))
    if not os.path.isfile(log_path):
        return {}
    cur_graph = None
    cur_cfg = None
    with open(log_path, "r") as f:
        for line in f:
            gm = GRAPH_MARK_RE.search(line)
            if gm:
                cur_graph = os.path.basename(gm.group(1))
                continue
            am = ABL_MARK_RE.search(line)
            if am:
                cur_cfg = (int(am.group(1)), int(am.group(2)), int(am.group(3)))
                continue
            pm = PHASE_RE.search(line)
            if pm and cur_cfg is not None and cur_graph is not None:
                phases[cur_cfg][cur_graph]["bfs"].append(float(pm.group(1)))
                phases[cur_cfg][cur_graph]["backward"].append(float(pm.group(2)))
    return phases


def med_phase(phases, cfg, graph, key):
    d = phases.get(cfg, {}).get(graph)
    if not d or not d[key]:
        return None
    return median(d[key])


def generate_phase_attribution(phases, times, graphs):
    """H→BFS, W→Backward, A→gap のフェーズ帰属テーブル。"""
    if not phases:
        return ""
    # graph 名は log では basename, TSV でも basename 想定だが差異に備え共通集合を使う
    log_graphs = set()
    for cfg in phases:
        log_graphs.update(phases[cfg].keys())
    graphs = [g for g in graphs if os.path.basename(g) in log_graphs or g in log_graphs] or list(log_graphs)

    def gph(cfg, g, key):
        return med_phase(phases, cfg, os.path.basename(g), key) or med_phase(phases, cfg, g, key)

    lines = ["", "# フェーズ帰属 (ablation.log 由来)", ""]
    lines.append("各構成の wall(TSV 中央値) / BFS cum / Backward cum / gap=wall−(BFS+Backward)。")
    lines.append("")
    lines.append("> **注記 (gap の解釈)**: A=1 (NS=2) では BFS cum / Backward cum が 2 ストリーム分の"
                 "合算のため、gap = wall−(BFS+Backward) が負になり得ます。これはバグではなく、2 ストリームの"
                 "カーネルが重なって実行された証拠（＝A の効果そのもの）です。負値には ‡ を付します。"
                 "論文で gap を引用する際はこの点を一文添えてください。")
    lines.append("")
    for g in graphs:
        lines.append(f"## {g}")
        lines.append("")
        lines.append("| 構成 | wall (s) | BFS cum (s) | Backward cum (s) | gap (s) |")
        lines.append("|:-----|------:|------:|------:|------:|")
        for cfg in all_configs_sorted():
            wall = med_time(times, cfg, g)
            bfs = gph(cfg, g, "bfs")
            bwd = gph(cfg, g, "backward")
            gap = None
            if wall is not None and bfs is not None and bwd is not None:
                gap = wall - bfs - bwd
            gap_str = fmt(gap)
            if gap is not None and gap < 0:
                gap_str += " ‡"  # 2 ストリーム重なり (A の効果) の証拠
            lines.append(f"| {config_label(cfg)} | {fmt(wall)} | {fmt(bfs)} | "
                         f"{fmt(bwd)} | {gap_str} |")
        lines.append("")

        # 帰属デルタ (leave-one-out ベース)
        bfs_h0 = gph(full_minus_config("H"), g, "bfs")
        bfs_h1 = gph(FULL, g, "bfs")
        bwd_w0 = gph(full_minus_config("W"), g, "backward")
        bwd_w1 = gph(FULL, g, "backward")
        wall_a0 = med_time(times, full_minus_config("A"), g)
        wall_a1 = med_time(times, FULL, g)

        def dline(label, v0, v1, unit="s"):
            if v0 is None or v1 is None:
                return f"- {label}: —"
            return f"- {label}: {fmt(v0)} → {fmt(v1)} ({unit}), Δ={fmt(v0 - v1)} {unit}"

        lines.append("**帰属 (full から各工夫を OFF→ON)**:")
        lines.append(dline("H の BFS cum 短縮 (BFS: H0→H1)", bfs_h0, bfs_h1))
        lines.append(dline("W の Backward cum 短縮 (Bwd: W0→W1)", bwd_w0, bwd_w1))
        # A は wall 短縮 (初期化/隙間の隠蔽) として現れる
        if wall_a0 is not None and wall_a1 is not None:
            lines.append(f"- A の wall 短縮 (init/gap 隠蔽): {fmt(wall_a0)} → {fmt(wall_a1)} (s), "
                         f"Δ={fmt(wall_a0 - wall_a1)} s")
        else:
            lines.append("- A の wall 短縮 (init/gap 隠蔽): —")
        lines.append("")
    return "\n".join(lines)


# ============================================================
#  試行数・ばらつきメモ
# ============================================================
def generate_trial_note(times, graphs):
    lines = ["", "# 試行数と実行時間ばらつき (中央値 ± Sample SD, ddof=1)", ""]
    lines.append("Sample SD は標本標準偏差 (ddof=1)。n<2 の場合は n/a とする。")
    lines.append("")
    lines.append("| 構成 | " + " | ".join(f"{g} Median ± Sample SD [s]" for g in graphs) + " |")
    lines.append("|:-----|" + "|".join("------:" for _ in graphs) + "|")
    for cfg in all_configs_sorted():
        cols = []
        for g in graphs:
            vals = times.get(cfg, {}).get(g)
            if not vals:
                cols.append("—")
            else:
                m = median(vals)
                sd = stdev(vals) if len(vals) >= 2 else None
                sd_str = f"{sd:.3f}" if sd is not None else "n/a"
                cols.append(f"{fmt(m)}±{sd_str} (n={len(vals)})")
        lines.append(f"| {config_label(cfg)} | " + " | ".join(cols) + " |")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ablation_results.tsv> [output_dir]", file=sys.stderr)
        sys.exit(1)
    tsv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(tsv_path) or "."
    if not os.path.isfile(tsv_path):
        print(f"ERROR: ファイルが見つかりません: {tsv_path}", file=sys.stderr)
        sys.exit(1)

    times, gteps, graphs = load_results(tsv_path)
    if not graphs:
        print("ERROR: 有効なアブレーション結果が 0 件です", file=sys.stderr)
        sys.exit(1)

    n_cfg = sum(1 for cfg in all_configs_sorted()
                if any(med_time(times, cfg, g) is not None for g in graphs))
    print(f"  読み込み: {len(graphs)} グラフ × {n_cfg} 構成")

    contrib = compute_contributions(times, graphs)

    parts = [
        generate_raw_table(times, gteps, graphs),
        generate_contribution_tables(contrib, graphs),
        generate_trial_note(times, graphs),
    ]

    # フェーズ帰属 (任意)
    log_path = os.path.join(os.path.dirname(tsv_path) or ".", "ablation.log")
    phases = parse_ablation_log(log_path)
    if phases:
        parts.append(generate_phase_attribution(phases, times, graphs))
        print(f"  フェーズ帰属: ablation.log を解析 ({len(phases)} 構成)")
    else:
        print("  フェーズ帰属: ablation.log なし/解析不可 → スキップ")

    md_path = os.path.join(output_dir, "ablation_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(parts).rstrip("\n") + "\n")
    print(f"  → {md_path}")

    tsv_out = os.path.join(output_dir, "ablation_contributions.tsv")
    write_contributions_tsv(tsv_out, contrib, graphs)
    print(f"  → {tsv_out}")

    # 画面にも寄与表を表示
    print()
    print(generate_contribution_tables(contrib, graphs))


if __name__ == "__main__":
    main()
