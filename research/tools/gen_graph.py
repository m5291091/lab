#!/usr/bin/env python3
"""
合成グラフ生成ツール → CSR 形式直接出力

対応モデル:
  ba  (Barabási-Albert スケールフリーグラフ)
      実世界ソーシャル/Webグラフを模倣。ハブが存在し、BC 値が偏る。
      パラメータ: --m (各新規ノードが接続する既存ノード数, デフォルト 5)

  er  (Erdős-Rényi ランダムグラフ G(n,p))
      均一ランダム接続。BC 値が均一に分布。
      パラメータ: --p (辺出現確率, デフォルト 1e-5)
                  辺数の概算 = n*(n-1)/2 * p

  grid (2D グリッドグラフ)
      道路ネットワーク類似。BFS 深さが深く、BC フェーズ時間比較に有効。
      パラメータ: --nodes は縦横 sqrt(n) × sqrt(n) に自動調整

使用方法:
    python3 tools/gen_graph.py --model ba --nodes 1000000 --seed 42 -o data/syn_ba_1M
    python3 tools/gen_graph.py --model er --nodes 500000  --p 2e-5 --seed 0  -o data/syn_er_500k
    python3 tools/gen_graph.py --model grid --nodes 1000000 --seed 0 -o data/syn_grid_1M

出力 CSR 形式:
    行 1: n_nodes n_edges
    行 2: ptr[0..n_nodes]  (n_nodes+1 個の row pointer)
    行 3: adj[0..2*n_edges-1]  (無向辺を両方向で格納)
"""
import argparse
import math
import os
import random
import sys


# ------------------------------------------------------------------ #
#  グラフ生成関数
# ------------------------------------------------------------------ #

def gen_ba(n: int, m: int, rng: random.Random):
    """Barabási-Albert 優先接続モデル (O(n*m))"""
    assert m >= 1 and n > m
    # 初期クリークを m+1 ノードで構築
    adj = [set() for _ in range(n)]
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i].add(j)
            adj[j].add(i)

    # 次数に比例した確率で接続 (stubs リスト方式)
    stubs = []
    for i in range(m + 1):
        stubs.extend([i] * (m if i > 0 else m + 1))

    for new_node in range(m + 1, n):
        targets = set()
        while len(targets) < m:
            t = rng.choice(stubs)
            if t != new_node:
                targets.add(t)
        for t in targets:
            adj[new_node].add(t)
            adj[t].add(new_node)
        stubs.extend([new_node] * m)
        for t in targets:
            stubs.append(t)

    return adj


def gen_er(n: int, p: float, rng: random.Random):
    """Erdős-Rényi G(n,p) ランダムグラフ (スキップ法で効率化)"""
    adj = [set() for _ in range(n)]
    # Batagelj & Brandes のスキップ法: O(n + m)
    v = 1
    w = -1
    log1mp = math.log(1.0 - p)
    while v < n:
        r = rng.random()
        if r < 1e-300:
            r = 1e-300
        w += 1 + int(math.log(r) / log1mp)
        while w >= v and v < n:
            w -= v
            v += 1
        if v < n:
            adj[v].add(w)
            adj[w].add(v)
    return adj


def gen_grid(n: int, rng: random.Random):
    """2D グリッドグラフ (√n × √n、端をトリミング)"""
    side = int(math.isqrt(n))
    actual_n = side * side
    adj = [set() for _ in range(actual_n)]
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if c + 1 < side:
                adj[i].add(i + 1)
                adj[i + 1].add(i)
            if r + 1 < side:
                adj[i].add(i + side)
                adj[i + side].add(i)
    return adj, actual_n


# ------------------------------------------------------------------ #
#  CSR 書き出し
# ------------------------------------------------------------------ #

def write_csr(adj, n_nodes, output_path):
    """隣接リスト → CSR 形式で書き出し"""
    adj_sorted = [sorted(adj[i]) for i in range(n_nodes)]
    total_adj = sum(len(a) for a in adj_sorted)
    n_edges = total_adj // 2

    print(f"  グラフ統計:", flush=True)
    print(f"    ノード数  : {n_nodes:,}", flush=True)
    print(f"    辺数      : {n_edges:,}", flush=True)
    if n_nodes > 0:
        degrees = [len(a) for a in adj_sorted]
        print(f"    平均次数  : {total_adj / n_nodes:.2f}", flush=True)
        print(f"    最大次数  : {max(degrees):,}", flush=True)
        isolated = sum(1 for d in degrees if d == 0)
        print(f"    孤立ノード: {isolated:,}", flush=True)

    print(f"  書き出し中: {output_path}", flush=True)

    WRITE_BUF = 1_000_000
    with open(output_path, 'w') as f:
        f.write(f"{n_nodes} {n_edges}\n")

        ptr = [0] * (n_nodes + 1)
        for i, neighbors in enumerate(adj_sorted):
            ptr[i + 1] = ptr[i] + len(neighbors)
        f.write(' '.join(str(p) for p in ptr) + '\n')

        buf = []
        for neighbors in adj_sorted:
            buf.extend(neighbors)
            if len(buf) >= WRITE_BUF:
                f.write(' '.join(str(x) for x in buf) + ' ')
                buf = []
        if buf:
            f.write(' '.join(str(x) for x in buf))
        f.write('\n')

    sz_mb = os.path.getsize(output_path) / 1e6
    print(f"  完了: {output_path}  ({sz_mb:.1f} MB)", flush=True)


# ------------------------------------------------------------------ #
#  main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description='合成グラフ生成 → CSR 形式出力',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--model', choices=['ba', 'er', 'grid'], default='ba',
                        help='グラフモデル (デフォルト: ba)')
    parser.add_argument('--nodes', type=int, default=100_000,
                        help='ノード数 (デフォルト: 100000)')
    parser.add_argument('--m', type=int, default=5,
                        help='BA モデルの接続数 m (デフォルト: 5)')
    parser.add_argument('--p', type=float, default=1e-5,
                        help='ER モデルの辺出現確率 p (デフォルト: 1e-5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='乱数シード (デフォルト: 42)')
    parser.add_argument('-o', '--output', required=True,
                        help='出力 CSR ファイルパス')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    n = args.nodes

    print(f"  モデル : {args.model}", flush=True)
    print(f"  ノード数: {n:,}", flush=True)
    print(f"  シード : {args.seed}", flush=True)

    if args.model == 'ba':
        print(f"  m      : {args.m}", flush=True)
        adj = gen_ba(n, args.m, rng)
        write_csr(adj, n, args.output)

    elif args.model == 'er':
        print(f"  p      : {args.p:.2e}  (期待辺数 ≈ {int(n*(n-1)/2*args.p):,})", flush=True)
        adj = gen_er(n, args.p, rng)
        write_csr(adj, n, args.output)

    elif args.model == 'grid':
        adj, actual_n = gen_grid(n, rng)
        if actual_n != n:
            print(f"  注意: grid はノード数を {actual_n:,} に調整 ({int(math.isqrt(n))}×{int(math.isqrt(n))})", flush=True)
        write_csr(adj, actual_n, args.output)


if __name__ == '__main__':
    main()
