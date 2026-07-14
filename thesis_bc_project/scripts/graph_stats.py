#!/usr/bin/env python3
# ============================================================
#  graph_stats.py — グラフ次数統計の収集
#
#  BFS カーネル選択則の再設計に向けて、各グラフの次数分布の特徴量を
#  算出する。作業仮説は「勝敗を決めるのは avg_deg ではなく次数分布の
#  歪み (ハブの有無 ≒ 最大次数)」であり、その前提確認に用いる。
#
#  入力: 本リポジトリ CSR テキスト形式 (空白区切り)
#    1) n m                        (ノード数 エッジ数)
#    2) ptr[0..n]                  (行オフセット, n+1 個)
#    3) adj[0..2m-1]               (隣接配列, 2m 個; 無向グラフ)
#
#  出力: 各グラフについて以下を算出し TSV / Markdown 表で出力する。
#    n, m, avg_deg, max_deg, p99 次数, 次数の変動係数 CV,
#    サンプル BFS 深さ (固定シードで選んだ 3 ソースの離心数の中央値)
#
#  実行 (numpy 依存):
#    uv run --with numpy python3 scripts/graph_stats.py --outdir docs
#    (もしくは numpy 導入済み環境で python3 scripts/graph_stats.py)
# ============================================================

import argparse
import os
import random
import sys

import numpy as np


# ------------------------------------------------------------
# CSR グラフの読み込み
# ------------------------------------------------------------
def load_csr(path):
    """CSR テキストを (n, m, ptr, adj) で返す。ptr/adj は np.int64 配列。"""
    with open(path, "r") as f:
        header = f.readline().split()
        n, m = int(header[0]), int(header[1])
        rest = f.read()
    # 空白 (空白/改行) 区切りの整数列をまとめて数値化
    vals = np.fromstring(rest, dtype=np.int64, sep=" ")
    if vals.size < n + 1:
        raise ValueError(
            f"{path}: ポインタ配列が不足 (期待 >= {n + 1}, 実際 {vals.size})")
    ptr = vals[: n + 1]
    # 真の隣接長は ptr[n] を信頼する。ヘッダの 2m は自己ループ除去や
    # 非対称エッジのため数個ずれる場合がある (例: 325557_3216152)。
    adj_len = int(ptr[n])
    need = (n + 1) + adj_len
    if vals.size < need:
        raise ValueError(
            f"{path}: 隣接配列が不足 (期待 {need}, 実際 {vals.size})")
    adj = vals[n + 1 : n + 1 + adj_len]
    return n, m, ptr, adj


# ------------------------------------------------------------
# レベル同期 BFS (numpy ベクトル化)。src からの離心数 (最大到達距離) を返す
# ------------------------------------------------------------
def bfs_depth(ptr, adj, src):
    n_rows = ptr.shape[0] - 1                       # 隣接行を持つ頂点数
    max_id = int(adj.max()) if adj.size else n_rows - 1
    size = max(n_rows, max_id + 1)                  # 1-indexed 等で範囲外 id があっても安全に
    dist = np.full(size, -1, dtype=np.int64)
    dist[src] = 0
    frontier = np.array([src], dtype=np.int64)
    level = 0
    while frontier.size:
        # 行を持つ (id < n_rows) フロンティアのみ展開する
        valid = frontier[frontier < n_rows]
        if valid.size == 0:
            break
        starts = ptr[valid]
        counts = ptr[valid + 1] - starts
        total = int(counts.sum())
        if total == 0:
            break
        # フロンティア各頂点の隣接スライスをベクトル化して収集
        rep_starts = np.repeat(starts, counts)
        first_idx = np.repeat(np.cumsum(counts) - counts, counts)
        offsets = np.arange(total, dtype=np.int64) - first_idx
        neigh = np.unique(adj[rep_starts + offsets])
        new = neigh[dist[neigh] < 0]
        if new.size == 0:
            break
        level += 1
        dist[new] = level
        frontier = new
    return int(dist.max())


# ------------------------------------------------------------
# 1 グラフの統計量を算出
# ------------------------------------------------------------
def compute_stats(path, n_sources=3, seed=42):
    n, m, ptr, adj = load_csr(path)
    degree = np.diff(ptr).astype(np.float64)

    avg_deg = float(degree.mean()) if n > 0 else 0.0
    max_deg = int(degree.max()) if n > 0 else 0
    p99_deg = float(np.percentile(degree, 99)) if n > 0 else 0.0
    mean = degree.mean() if n > 0 else 0.0
    cv = float(degree.std() / mean) if mean > 0 else 0.0

    # サンプル BFS 深さ: 次数 > 0 の頂点から固定シードで最大 3 ソース選択
    rng = random.Random(seed)
    candidates = np.nonzero(np.diff(ptr) > 0)[0]
    depths = []
    if candidates.size > 0:
        k = min(n_sources, candidates.size)
        idxs = rng.sample(range(candidates.size), k)
        for i in idxs:
            depths.append(bfs_depth(ptr, adj, int(candidates[i])))
    bfs_med = int(np.median(depths)) if depths else 0

    return {
        "n": n,
        "m": m,
        "avg_deg": avg_deg,
        "max_deg": max_deg,
        "p99_deg": p99_deg,
        "cv_deg": cv,
        "bfs_depth_med": bfs_med,
        "bfs_samples": depths,
    }


# 対象グラフ: (ラベル, data/ 相対パス, 分類)
DEFAULT_TARGETS = [
    ("email-EuAll",      "snap/email-EuAll",       "実データ(ハブ有)"),
    ("roadNet-PA",       "snap/roadNet-PA",        "道路網"),
    ("roadNet-TX",       "snap/roadNet-TX",        "道路網"),
    ("roadNet-CA",       "snap/roadNet-CA",        "道路網"),
    ("benchmark_85830",  "benchmark_85830.data",   "合成ベンチ"),
    ("benchmark_7000",   "benchmark_7000_41459",   "合成ベンチ"),
    ("benchmark_11023",  "benchmark_11023_62184",  "合成ベンチ"),
    ("56438_300801",     "56438_300801",           "合成ベンチ"),
    ("325557_3216152",   "325557_3216152",         "合成ベンチ"),
    # 合成グラフ (存在すれば参考として追加)
    ("chain_200",        "chain_200",              "合成(鎖状)"),
    ("random",           "random",                 "合成(ランダム)"),
]


def main():
    parser = argparse.ArgumentParser(description="グラフ次数統計の収集")
    parser.add_argument(
        "--data-dir", default=None,
        help="data ディレクトリ (既定: スクリプトから見た ../data)")
    parser.add_argument(
        "--outdir", default=".",
        help="TSV/Markdown 出力先 (既定: カレント)")
    parser.add_argument(
        "--sources", type=int, default=3, help="BFS サンプルソース数")
    parser.add_argument(
        "--seed", type=int, default=42, help="ソース選択の乱数シード")
    parser.add_argument(
        "extra", nargs="*",
        help="追加グラフパス (絶対 or data/ 相対)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = args.data_dir or os.path.join(project_dir, "data")

    targets = list(DEFAULT_TARGETS)
    for e in args.extra:
        targets.append((os.path.basename(e), e, "追加"))

    rows = []
    for label, rel, category in targets:
        path = rel if os.path.isabs(rel) else os.path.join(data_dir, rel)
        if not os.path.isfile(path):
            print(f"[SKIP] {label}: グラフなし ({path})", file=sys.stderr)
            continue
        print(f"[STAT] {label} ...", file=sys.stderr, flush=True)
        try:
            st = compute_stats(path, n_sources=args.sources, seed=args.seed)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {label}: {exc}", file=sys.stderr)
            continue
        st.update({"graph": label, "category": category})
        rows.append(st)
        print(
            f"        n={st['n']} m={st['m']} avg={st['avg_deg']:.2f} "
            f"max={st['max_deg']} p99={st['p99_deg']:.0f} "
            f"CV={st['cv_deg']:.2f} bfs_med={st['bfs_depth_med']} "
            f"(samples={st['bfs_samples']})",
            file=sys.stderr,
        )

    if not rows:
        print("有効なグラフがありません", file=sys.stderr)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    tsv_path = os.path.join(args.outdir, "graph_stats.tsv")
    md_path = os.path.join(args.outdir, "graph_stats.md")

    cols = ["graph", "category", "n", "m", "avg_deg", "max_deg",
            "p99_deg", "cv_deg", "bfs_depth_med"]
    with open(tsv_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join([
                r["graph"], r["category"], str(r["n"]), str(r["m"]),
                f"{r['avg_deg']:.3f}", str(r["max_deg"]),
                f"{r['p99_deg']:.1f}", f"{r['cv_deg']:.3f}",
                str(r["bfs_depth_med"]),
            ]) + "\n")

    # Markdown 表
    md_lines = [
        "# グラフ次数統計 (graph_stats.py)",
        "",
        f"- BFS サンプルソース数: {args.sources} (シード {args.seed}, 次数>0 から選択)",
        "- CV = 次数の標準偏差 / 平均 (母標準偏差)",
        "",
        "| グラフ | 分類 | n | m | avg_deg | max_deg | p99_deg | CV | BFS深さ(中央) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['graph']} | {r['category']} | {r['n']:,} | {r['m']:,} | "
            f"{r['avg_deg']:.2f} | {r['max_deg']:,} | {r['p99_deg']:.0f} | "
            f"{r['cv_deg']:.2f} | {r['bfs_depth_med']} |")
    md_text = "\n".join(md_lines) + "\n"
    with open(md_path, "w") as f:
        f.write(md_text)

    print("\n" + md_text)
    print(f"[OK] TSV : {tsv_path}", file=sys.stderr)
    print(f"[OK] MD  : {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
