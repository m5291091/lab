#!/usr/bin/env python3
"""
CSR グラフ整合性検証器 (Gate W7.3A)

data/ の 3 行 CSR テキスト形式グラフを検査し、PASS / FAIL を判定する。
不正入力を GPU ランナーへ渡す前に停止させるためのゲートとして使う。

検査項目:
    header (n, m) / row pointer 数 (n+1) / ptr[0]==0 / ptr 単調非減少 /
    ptr[n]==2m / adjacency 要素数==2m / 全頂点 ID が 0<=v<n / 次数統計 /
    有向-無向対称性 / self-loop 数 / duplicate edge 数と多重度 /
    孤立頂点数 / 連結成分数 / SHA256 / file size / validation status

使用方法:
    python3 tools/validate_graph_csr.py data/325557_3216152_corrected_v1
    python3 tools/validate_graph_csr.py data/foo --json report.json
    python3 tools/validate_graph_csr.py data/foo --quiet      # 判定行のみ

終了コード:
    0 = PASS   1 = FAIL (整合性違反)   2 = 使用法 / 入出力エラー

注意:
    本検証器は入力を一切書き換えない。読み取り専用である。
"""
import argparse
import hashlib
import json
import os
import re
import sys
import warnings

import numpy as np

# 非負整数・負整数・空白のみを許容 (それ以外は FAIL として報告)
TOKEN_RE = re.compile(r"^[\s\-0-9]*$")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_ints(line):
    """空白区切り整数列を int64 配列へ。数値以外を含む場合は None を返す。"""
    if not TOKEN_RE.match(line):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return np.fromstring(line, dtype=np.int64, sep=" ")


def count_components(n, u, v):
    """union-find で連結成分数と最大成分サイズを求める。

    scipy.sparse.csgraph は環境の scipy が numpy 2.x と非互換なため使わない。
    u, v は自己ループを除いた無向辺の端点 (重複可)。
    """
    parent = list(range(n))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:      # path compression
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(u.tolist(), v.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    _, sizes = np.unique(roots, return_counts=True)
    return int(len(sizes)), int(sizes.max())


class Report:
    def __init__(self, path):
        self.path = path
        self.checks = []      # (name, ok, detail)
        self.stats = {}
        self.fatal = False    # 以降の検査を続行できない致命的違反

    def check(self, name, ok, detail=""):
        self.checks.append((name, bool(ok), detail))
        return bool(ok)

    def stat(self, key, value):
        self.stats[key] = value

    @property
    def failures(self):
        return [(n, d) for (n, ok, d) in self.checks if not ok]

    @property
    def status(self):
        return "PASS" if not self.failures else "FAIL"


def validate(path, report):
    r = report
    r.stat("path", path)
    r.stat("file_size_bytes", os.path.getsize(path))
    r.stat("sha256", sha256_of(path))

    with open(path, "r") as f:
        raw = f.read()
    lines = raw.split("\n")
    # 末尾の空行は許容する
    while lines and lines[-1].strip() == "":
        lines.pop()

    if not r.check("line_count==3", len(lines) == 3, f"actual={len(lines)}"):
        r.fatal = True
        return

    # ---- header ----
    head = lines[0].split()
    if not r.check("header has 2 ints", len(head) == 2, f"actual={head[:5]}"):
        r.fatal = True
        return
    try:
        n, m = int(head[0]), int(head[1])
    except ValueError:
        r.check("header parses as int", False, f"header={lines[0][:80]!r}")
        r.fatal = True
        return
    r.stat("n", n)
    r.stat("m", m)
    r.stat("expected_adj_len_2m", 2 * m)
    r.check("n > 0", n > 0, f"n={n}")
    r.check("m >= 0", m >= 0, f"m={m}")
    if n <= 0:
        r.fatal = True
        return

    # ---- row pointer ----
    ptr = parse_ints(lines[1])
    if ptr is None:
        r.check("ptr line is numeric", False, "non-numeric token in row-pointer line")
        r.fatal = True
        return
    r.stat("ptr_len", int(len(ptr)))
    if not r.check("len(ptr)==n+1", len(ptr) == n + 1, f"actual={len(ptr)} expected={n + 1}"):
        r.fatal = True
        return
    r.check("ptr[0]==0", ptr[0] == 0, f"actual={int(ptr[0])}")
    diffs = np.diff(ptr)
    bad = np.flatnonzero(diffs < 0)
    r.check("ptr non-decreasing", len(bad) == 0,
            "" if len(bad) == 0 else
            f"{len(bad)} decreasing steps; first at i={int(bad[0])} "
            f"(ptr[{int(bad[0])}]={int(ptr[bad[0]])} > ptr[{int(bad[0]) + 1}]={int(ptr[bad[0] + 1])})")
    r.stat("ptr_n", int(ptr[-1]))
    r.check("ptr[n]==2m", int(ptr[-1]) == 2 * m,
            f"ptr[n]={int(ptr[-1])} expected 2m={2 * m} (diff={2 * m - int(ptr[-1])})")

    # ---- adjacency ----
    adj = parse_ints(lines[2])
    if adj is None:
        r.check("adj line is numeric", False, "non-numeric token in adjacency line")
        r.fatal = True
        return
    r.stat("adj_len", int(len(adj)))
    adj_ok = r.check("len(adj)==2m", len(adj) == 2 * m,
                     f"actual={len(adj)} expected={2 * m} (shortage={2 * m - len(adj)})")

    # ---- 頂点 ID 範囲 ----
    if len(adj) == 0:
        r.stat("adj_min", None)
        r.stat("adj_max", None)
        oor_idx = np.array([], dtype=np.int64)
    else:
        r.stat("adj_min", int(adj.min()))
        r.stat("adj_max", int(adj.max()))
        oor_idx = np.flatnonzero((adj < 0) | (adj >= n))
    r.stat("out_of_range_count", int(len(oor_idx)))
    if len(oor_idx):
        head_list = [(int(i), int(adj[i])) for i in oor_idx[:8]]
        vals = sorted({int(v) for v in adj[oor_idx]})
        r.stat("out_of_range_values", vals[:16])
        r.check("all 0<=v<n", False,
                f"{len(oor_idx)} entries out of range [0,{n - 1}]; "
                f"first (adj_index, value): {head_list}; distinct values={vals[:16]}")
    else:
        r.stat("out_of_range_values", [])
        r.check("all 0<=v<n", True)

    range_ok = len(oor_idx) == 0

    # ---- 次数 ----
    deg = np.diff(ptr)
    r.stat("degree_sum", int(deg.sum()))
    r.check("sum(degree)==len(adj)", int(deg.sum()) == len(adj),
            f"sum(deg)={int(deg.sum())} len(adj)={len(adj)}")
    r.stat("degree_min", int(deg.min()) if len(deg) else None)
    r.stat("degree_max", int(deg.max()) if len(deg) else None)
    r.stat("degree_mean", round(float(deg.mean()), 6) if len(deg) else None)
    isolated = int(np.count_nonzero(deg == 0))
    r.stat("isolated_vertices", isolated)
    iso_idx = np.flatnonzero(deg == 0)
    r.stat("isolated_vertex_sample", [int(i) for i in iso_idx[:8]])

    # 構造が壊れている場合、以降の多重集合解析は意味を持たないので打ち切る
    if not (adj_ok and range_ok and int(deg.sum()) == len(adj)):
        r.stat("symmetry", "not_computed")
        r.stat("self_loop_edges", "not_computed")
        r.stat("duplicate_edges", "not_computed")
        r.stat("connected_components", "not_computed")
        return

    # ---- 対称性 / self-loop / duplicate ----
    src = np.repeat(np.arange(n, dtype=np.int64), deg)
    dst = adj
    key_fwd = src * np.int64(n) + dst
    key_rev = dst * np.int64(n) + src
    f_sorted = np.sort(key_fwd)
    r_sorted = np.sort(key_rev)
    symmetric = bool(f_sorted.shape == r_sorted.shape and np.array_equal(f_sorted, r_sorted))
    r.stat("symmetry", "symmetric" if symmetric else "asymmetric")
    if symmetric:
        r.check("undirected symmetry", True)
    else:
        # 非対称な有向ペアを列挙して報告
        uf, cf = np.unique(key_fwd, return_counts=True)
        ur, cr = np.unique(key_rev, return_counts=True)
        allk = np.union1d(uf, ur)
        cnt_f = np.zeros(len(allk), dtype=np.int64)
        cnt_r = np.zeros(len(allk), dtype=np.int64)
        cnt_f[np.searchsorted(allk, uf)] = cf
        cnt_r[np.searchsorted(allk, ur)] = cr
        mism = np.flatnonzero(cnt_f != cnt_r)
        r.stat("asymmetric_ordered_pairs", int(len(mism)))
        ex = []
        for k in mism[:8]:
            key = int(allk[k])
            ex.append(((key // n, key % n), int(cnt_f[k]), int(cnt_r[k])))
        r.check("undirected symmetry", False,
                f"{len(mism)} ordered pairs with count(u,v)!=count(v,u); "
                f"first (u,v),fwd,rev: {ex}")

    self_entries = int(np.count_nonzero(src == dst))
    r.stat("self_loop_adj_entries", self_entries)
    r.stat("self_loop_edges", self_entries // 2)
    r.check("self-loop entries even (stored twice)", self_entries % 2 == 0,
            "" if self_entries % 2 == 0 else
            f"self-loop adjacency entries={self_entries} は奇数 (両方向格納の規約に反する)")

    uniq, counts = np.unique(key_fwd, return_counts=True)
    dup_pairs = int(np.count_nonzero(counts > 1))
    r.stat("distinct_ordered_pairs", int(len(uniq)))
    r.stat("duplicate_ordered_pairs", dup_pairs)
    r.stat("extra_entries_from_duplicates", int((counts - 1).sum()))
    mult_hist = {int(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))}
    r.stat("multiplicity_histogram", mult_hist)
    r.stat("max_multiplicity", int(counts.max()) if len(counts) else 0)

    # ---- 連結成分 ----
    # 自己ループは連結性に寄与しないので除外し、無向辺を一意化してから union-find
    sel = src < dst
    ukey = np.unique(src[sel] * np.int64(n) + dst[sel])
    ncomp, largest = count_components(n, ukey // n, ukey % n)
    r.stat("connected_components", ncomp)
    r.stat("largest_component_size", largest)


def main():
    ap = argparse.ArgumentParser(description="CSR グラフ整合性検証器 (Gate W7.3A)")
    ap.add_argument("graph", help="CSR テキストグラフのパス")
    ap.add_argument("--json", help="検査結果を JSON で書き出す")
    ap.add_argument("--quiet", action="store_true", help="判定行のみ出力")
    args = ap.parse_args()

    if not os.path.isfile(args.graph):
        print(f"[ERROR] not a file: {args.graph}", file=sys.stderr)
        return 2

    rep = Report(args.graph)
    try:
        validate(args.graph, rep)
    except OSError as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"=== validate_graph_csr: {args.graph} ===")
        for k, v in rep.stats.items():
            print(f"  {k:32s}: {v}")
        print("  --- checks ---")
        for name, ok, detail in rep.checks:
            mark = "OK  " if ok else "FAIL"
            print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))

    print(f"VALIDATION {rep.status}: {args.graph} "
          f"({len(rep.failures)} failed check(s))")

    if args.json:
        payload = dict(rep.stats)
        payload["validation_status"] = rep.status
        payload["checks"] = [{"name": n, "ok": ok, "detail": d} for (n, ok, d) in rep.checks]
        payload["failed_checks"] = [n for (n, _) in rep.failures]
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)
            f.write("\n")
        if not args.quiet:
            print(f"  -> JSON: {args.json}")

    return 0 if rep.status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
