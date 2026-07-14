#!/usr/bin/env python3
"""
UMオーバーサブスクライブ実験の作図関数。

generate_figures_tables.py に import して使うか、単体で直接実行できる。

Usage（単体実行）:
  python3 generate_um_figures.py [tsv_path] [output_dir]

  tsv_path のデフォルト:
    <このスクリプトの場所>/../raw_data/memory_scalability/  (impl別 oversubscribe_results_*.tsv を再帰収集)
  output_dir のデフォルト: カレントディレクトリ
"""

import sys
import os
import csv
import math
from collections import defaultdict

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

# ─── スタイル定数 ─────────────────────────────────────────────────────────────
STYLE = {
    "gpu_opt_pure": dict(color="#e05252", marker="s", linestyle="-",  label="Proposal-Gen (pure)"),
    "gpu_opt":      dict(color="#3a7ebf", marker="o", linestyle="--", label="Proposal-GH200 (UM)"),
}
OOM_MARKER_KW = dict(marker="x", s=120, linewidths=2.5, zorder=5)
HBM3_LINE_KW  = dict(color="#888888", linestyle=":", linewidth=1.2, alpha=0.8)
CAPSIZE       = 4
FIG_W, FIG_H  = 7.0, 4.5


# ─── データ読み込み（両スクリプト共通）──────────────────────────────────────
def load_tsv(path: str) -> dict:
    """
    Returns:
        data[(impl, batch)] = {
            "gteps": [float, ...],   # SUCCESS 行のみ
            "time":  [float, ...],
            "n_ok":  int,
            "n_oom": int,
        }
    """
    import glob as _glob
    # path はディレクトリ(raw_data/memory_scalability, impl別 3分割TSV) または単一TSV を受け付ける
    if os.path.isdir(path):
        files = sorted(_glob.glob(os.path.join(path, "**", "oversubscribe_results_*.tsv"), recursive=True))
    else:
        files = [path]
    data = defaultdict(lambda: {"gteps": [], "time": [], "n_ok": 0, "n_oom": 0})
    for fp in files:
        with open(fp, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                impl   = row["Implementation"].strip()
                batch  = int(row["BatchSize"])
                status = row["Status"].strip()
                key    = (impl, batch)
                if status == "SUCCESS":
                    data[key]["n_ok"] += 1
                    data[key]["gteps"].append(float(row["GTEPS"]))
                    data[key]["time"].append(float(row["Time_sec"]))
                else:
                    data[key]["n_oom"] += 1
    return data


def _agg(vals):
    """(median, std) を返す。vals が空なら (None, None)。"""
    if not vals:
        return None, None
    arr = sorted(vals)
    n   = len(arr)
    med = arr[n // 2] if n % 2 else (arr[n // 2 - 1] + arr[n // 2]) / 2
    mu  = sum(vals) / n
    std = math.sqrt(sum((x - mu) ** 2 for x in vals) / n)
    return med, std


# ─── fig_um_in_capacity ───────────────────────────────────────────────────────
def fig_um_in_capacity(data: dict, out_path: str) -> None:
    """
    HBM3 在庫内（BATCH ≤ 4096）の UM / pure GTEPS 比較。

    ポイント:
    - エラーバー付き折れ線（median ± std）
    - Y 軸を tight に絞り ±0.2% 程度の差が視覚的に分かるよう設定
    - 「±0.2% 差なし」の旨を本文 annotation として入れる
    """
    if not HAS_LIBS:
        print("[WARN] matplotlib/numpy が見つかりません。fig_um_in_capacity をスキップします。")
        return

    in_cap_batches = sorted({b for (_, b) in data if b <= 4096})
    if not in_cap_batches:
        print("[WARN] 在庫内 BATCH データなし。スキップします。")
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for impl, sty in STYLE.items():
        x_vals, y_med, y_err = [], [], []
        for b in in_cap_batches:
            med, std = _agg(data.get((impl, b), {}).get("gteps", []))
            if med is None:
                continue
            x_vals.append(b)
            y_med.append(med)
            y_err.append(std)

        ax.errorbar(
            x_vals, y_med, yerr=y_err,
            capsize=CAPSIZE, linewidth=1.8, markersize=7,
            **sty
        )

    # Y 軸を在庫内データの範囲に tight に設定（差を強調しすぎないよう ±3% のパディング）
    all_gteps = [
        g for (impl, b) in data if b <= 4096
        for g in data[(impl, b)]["gteps"]
    ]
    if all_gteps:
        lo, hi = min(all_gteps), max(all_gteps)
        pad = (hi - lo) * 0.5 + 0.05  # 少し余白
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_xscale("log", base=2)
    ax.set_xticks(in_cap_batches)
    ax.set_xticklabels([str(b) for b in in_cap_batches])
    ax.set_xlabel("Batch Size (BC_BATCH_OVERRIDE)", fontsize=11)
    ax.set_ylabel("GTEPS (Giga Traversed Edges/sec)", fontsize=11)
    ax.set_title("In-Capacity Performance: UM vs Pure (synth_326K, 5 trials)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    # 「差は ±0.2%」annotation
    ax.annotate(
        "Δ < 0.2% (statistically equivalent)",
        xy=(0.97, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8)
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved: {out_path}")


# ─── fig_um_oversubscribe ─────────────────────────────────────────────────────
def fig_um_oversubscribe(data: dict, out_path: str) -> None:
    """
    全 BATCH 範囲の容量 vs 性能グラフ。

    ポイント:
    - 成功点: 実線エラーバー
    - OOM/FAIL 点: 赤/青の ✕ をプロット（Y=0 位置、専用凡例ラベル）
    - HBM3 容量境界（≈102 GB の free_mem × 0.8 ≈ 81 GB に対応する BATCH）
      を縦点線で表示。実験上は batch=8192 が最初の OOM。
    - Y 軸は 0 から始め、OOM ✕ を y=0 付近に示す
    """
    if not HAS_LIBS:
        print("[WARN] matplotlib/numpy が見つかりません。fig_um_oversubscribe をスキップします。")
        return

    all_batches = sorted({b for (_, b) in data})
    if not all_batches:
        print("[WARN] データなし。スキップします。")
        return

    fig, ax = plt.subplots(figsize=(FIG_W + 1.5, FIG_H))

    oom_handles = []

    for impl, sty in STYLE.items():
        # 成功点
        x_ok, y_med, y_err = [], [], []
        # OOM 点
        x_oom = []

        for b in all_batches:
            entry   = data.get((impl, b), {})
            gteps   = entry.get("gteps", [])
            n_oom   = entry.get("n_oom", 0)
            n_ok    = entry.get("n_ok", 0)
            if n_ok > 0:
                med, std = _agg(gteps)
                x_ok.append(b)
                y_med.append(med)
                y_err.append(std if std is not None else 0.0)
            elif n_oom > 0:
                x_oom.append(b)

        if x_ok:
            ax.errorbar(
                x_ok, y_med, yerr=y_err,
                capsize=CAPSIZE, linewidth=1.8, markersize=7,
                **sty
            )

        if x_oom:
            # OOM ✕ マーカーを Y 軸底近くに配置
            y_oom = [-0.25] * len(x_oom)   # 0 より少し下
            sc = ax.scatter(
                x_oom, y_oom,
                color=sty["color"],
                label=f"{sty['label']} OOM/FAIL",
                **OOM_MARKER_KW
            )
            oom_handles.append(sc)

    # HBM3 容量境界: batch=8192 が最初の OOM（170 GB > HBM3 102 GB × 0.8 = 81.6 GB）
    ax.axvline(x=8192, **HBM3_LINE_KW, label="HBM3 threshold (BATCH=8192, ~170 GB)")

    # Y 軸下限を -0.5 にして ✕ が見えるように
    all_gteps = [g for (_, b) in data for g in data[(_, b)]["gteps"]]
    y_top = max(all_gteps) * 1.08 if all_gteps else 20.0
    ax.set_ylim(-0.5, y_top)

    # X 軸
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_batches)
    ax.set_xticklabels([str(b) for b in all_batches], rotation=30, ha="right")

    ax.set_xlabel("Batch Size (BC_BATCH_OVERRIDE)", fontsize=11)
    ax.set_ylabel("GTEPS (Giga Traversed Edges/sec)", fontsize=11)
    ax.set_title(
        "Capacity vs Performance: UM survives HBM3 overflow\n(synth_326K, 5 trials; ✕ = OOM/FAIL)",
        fontsize=12
    )

    # 凡例: 折れ線 + ✕ を統合
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, fontsize=9, loc="lower left")

    # 「pure は OOM / UM は完走」annotation（パッチ後データ向け）
    ax.annotate(
        "Pure: OOM (cudaMalloc)\nUM:   survives via LPDDR5X (NVLink-C2C)",
        xy=(8192 * 1.05, y_top * 0.72),
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.85)
    )

    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved: {out_path}")


# ─── 単体実行エントリポイント ─────────────────────────────────────────────────
def main():
    default_tsv = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "raw_data", "memory_scalability"
    )
    tsv_path   = sys.argv[1] if len(sys.argv) > 1 else default_tsv
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    if not os.path.exists(tsv_path):
        print(f"[ERROR] TSV/dir not found: {tsv_path}", file=sys.stderr)
        sys.exit(1)

    if not HAS_LIBS:
        print("[ERROR] matplotlib と numpy が必要です: pip install matplotlib numpy")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    data = load_tsv(tsv_path)

    print("=== 作図開始 ===")
    fig_um_in_capacity(
        data,
        out_path=os.path.join(output_dir, "fig_um_in_capacity.pdf")
    )
    fig_um_oversubscribe(
        data,
        out_path=os.path.join(output_dir, "fig_um_oversubscribe.pdf")
    )
    print("完了。")


if __name__ == "__main__":
    main()
