#!/usr/bin/env python3
"""
SNAP エッジリスト形式 → BC 実装用 CSR 形式 変換スクリプト (改良版)

SNAP 形式:
  # コメント行 (# で始まる行はスキップ)
  from_node  to_node
  ...

出力 CSR 形式 (無向グラフ):
  n_nodes  n_edges
  ptr[0] ptr[1] ... ptr[n_nodes]    (n_nodes+1 個)
  adj[0] adj[1] ... adj[2*n_edges-1]  (2*n_edges 個; 各辺を両方向に格納)

対応入力形式:
  - テキストファイル (.txt)
  - gzip 圧縮ファイル (.gz) ← 直接読み込み可能、gunzip 不要

使用方法:
    python3 snap_to_csr.py input.txt output_csr [--directed]
    python3 snap_to_csr.py input.txt.gz output_csr [--directed]
    --directed: 有向グラフとして変換 (辺を一方向のみ格納)
"""
import sys
import os
import gzip
import argparse
from collections import defaultdict

PROGRESS_INTERVAL = 1_000_000  # 100万辺ごとに進捗表示


def open_input(path):
    """テキストまたは gzip ファイルを自動判定して開く"""
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8', errors='replace')
    return open(path, 'r', encoding='utf-8', errors='replace')


def main():
    parser = argparse.ArgumentParser(description='SNAP エッジリスト → CSR 変換 (改良版)')
    parser.add_argument('input',  help='SNAP 形式入力ファイル (.txt または .gz)')
    parser.add_argument('output', help='CSR 出力ファイル名')
    parser.add_argument('--directed', action='store_true',
                        help='有向グラフとして変換 (デフォルト: 無向)')
    args = parser.parse_args()

    print(f"  入力: {args.input}", flush=True)

    # エッジリスト読み込み
    edges = []
    node_set = set()
    line_count = 0
    skip_count = 0

    with open_input(args.input) as src:
        for raw in src:
            line_count += 1
            raw = raw.strip()
            if not raw or raw.startswith('#'):
                skip_count += 1
                continue
            parts = raw.split()
            if len(parts) < 2:
                skip_count += 1
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                skip_count += 1
                continue
            if u == v:
                continue  # 自己ループを除去
            edges.append((u, v))
            node_set.add(u)
            node_set.add(v)

            if len(edges) % PROGRESS_INTERVAL == 0:
                print(f"    ... {len(edges):,} 辺読み込み済み", flush=True)

    print(f"  読み込み完了: {len(edges):,} 辺, {len(node_set):,} ノード (リマップ前)", flush=True)

    # ノード番号を 0-indexed に連番リマップ
    node_list = sorted(node_set)
    node_map = {n: i for i, n in enumerate(node_list)}
    n_nodes = len(node_list)

    # 隣接リストを構築 (無向の場合は両方向)
    adj = defaultdict(set)
    for u_raw, v_raw in edges:
        u = node_map[u_raw]
        v = node_map[v_raw]
        adj[u].add(v)
        if not args.directed:
            adj[v].add(u)

    # 各ノードの隣接リストをソート
    adj_sorted = [sorted(adj[i]) for i in range(n_nodes)]

    # 辺数カウント
    total_adj = sum(len(a) for a in adj_sorted)
    if args.directed:
        n_edges = total_adj
    else:
        n_edges = total_adj // 2

    # グラフ統計計算
    degrees = [len(a) for a in adj_sorted]
    max_deg = max(degrees) if degrees else 0
    avg_deg = total_adj / n_nodes if n_nodes > 0 else 0
    isolated = sum(1 for d in degrees if d == 0)

    print(f"  変換後: {n_nodes:,} ノード, {n_edges:,} 辺 ({'有向' if args.directed else '無向'})", flush=True)
    print(f"  統計  : 平均次数={avg_deg:.2f}, 最大次数={max_deg:,}, 孤立ノード数={isolated:,}", flush=True)
    print(f"  出力  : {args.output}", flush=True)

    # CSR 形式で書き出し
    with open(args.output, 'w') as f:
        f.write(f"{n_nodes} {n_edges}\n")

        # ptr 配列 (CSR row pointers)
        ptr = [0] * (n_nodes + 1)
        for i, neighbors in enumerate(adj_sorted):
            ptr[i + 1] = ptr[i] + len(neighbors)
        f.write(' '.join(str(p) for p in ptr) + '\n')

        # adj 配列 (CSR column indices) — 1M エントリごとにバッファ書き出し
        WRITE_BUF = 1_000_000
        buf = []
        written = 0
        for neighbors in adj_sorted:
            buf.extend(neighbors)
            if len(buf) >= WRITE_BUF:
                f.write(' '.join(str(x) for x in buf))
                buf = []
                written += WRITE_BUF
                if written % (10 * WRITE_BUF) == 0:
                    print(f"    ... {written:,} インデックス書き出し済み", flush=True)
                f.write(' ')
        if buf:
            f.write(' '.join(str(x) for x in buf))
        f.write('\n')

    file_size_mb = os.path.getsize(args.output) / 1e6
    print(f"  完了  : {args.output}  ({file_size_mb:.1f} MB)", flush=True)


if __name__ == '__main__':
    main()
