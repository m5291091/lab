#!/usr/bin/env python3
"""
data/325557_3216152 の決定的修復 (Gate W7.3A)

旧ファイル `data/325557_3216152` は、1-based 頂点番号の CSR を 0-based として
格納した不正入力である。結果として:

  * 隣接要素が 2m に 7 個不足する (ptr[n]=6432297, 2m=6432304)
  * 値 325557 (= n) が 7 個含まれ、0-based では範囲外
  * 頂点 0 が孤立 (1-based の欠番)
  * 最終頂点 (旧 1-based の 325557) の行が存在しない
    -- row pointer は ptr[0..n] しかなく、旧頂点 n の行は ptr[n+1] を要するため

本スクリプトは旧ファイルを **読み取り専用** で解釈し、グラフ全体の
relabelling (k -> k-1) として修復した新ファイルを別名で生成する。
旧ファイルは削除・上書き・修正しない。

修復方針 (Gate W7.3A §3):
  1. 旧 CSR を 1-based 頂点による無向 multigraph として解釈する
  2. 内部の対称性・多重度から欠落した最終頂点行の 7 要素を再構成する
  3. source と destination の両方を k -> k-1 で 0-based へ変換する
  4. self-loop と duplicate edge の多重度を保持する
  5. dedup・self-loop 除去・辺の追加削除などの正規化はしない
  6. n=325557, m=3216152 を維持する
  7. 同じ入力から常に byte-identical な出力を生成する

欠落 7 要素の一意性:
  旧頂点 n の次数 = 2m - ptr[n] = 7 であり、他行に出現する値 n の個数も 7。
  両者が一致するため、旧頂点 n の self-loop は (7-7)/2 = 0 本と確定する
  (self-loop は両方向 2 要素として格納される規約)。したがって旧頂点 n の
  隣接多重集合は「他行で n を持つ行の所有者」の多重集合として一意に定まる。
  行内の並び順は、逆辺の出現位置順と昇順ソートが一致することを検証している
  (どちらの規約でも同一の並びになるため、順序の選択は結果を左右しない)。

使用方法:
    python3 tools/repair_325557_graph.py                       # 既定の入出力
    python3 tools/repair_325557_graph.py --output /tmp/g       # 出力先を指定
    python3 tools/repair_325557_graph.py --dry-run             # 検査のみ (書き出さない)

終了コード:
    0 = 成功   1 = 前提条件の不一致 / 検証失敗 (出力しない)   2 = 使用法・入出力エラー
"""
import argparse
import hashlib
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)

DEFAULT_INPUT = os.path.join(PROJECT, "data", "325557_3216152")
DEFAULT_OUTPUT = os.path.join(PROJECT, "data", "325557_3216152_corrected_v1")

# Gate W7.2 / W7.3A で確認した旧ファイルの同一性
EXPECTED_LEGACY_SHA256 = "a095b2e7564e6c620bd0f5437917e0b28f4fecab289adf77633e850aa07da584"
EXPECTED_N = 325557
EXPECTED_M = 3216152
EXPECTED_SHORTAGE = 7

WRITE_CHUNK = 1_000_000


class PreconditionError(Exception):
    """旧ファイルが Gate W7.2 で確認した異常条件と一致しない。"""


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(cond, msg):
    if not cond:
        raise PreconditionError(msg)


def read_legacy(path):
    with open(path, "r") as f:
        raw = f.read()
    lines = raw.split("\n")
    while lines and lines[-1].strip() == "":
        lines.pop()
    require(len(lines) == 3, f"3 行 CSR ではない (lines={len(lines)})")
    head = lines[0].split()
    require(len(head) == 2, f"ヘッダが 2 整数ではない: {head[:4]}")
    n, m = int(head[0]), int(head[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ptr = np.fromstring(lines[1], dtype=np.int64, sep=" ")
        adj = np.fromstring(lines[2], dtype=np.int64, sep=" ")
    return n, m, ptr, adj


def check_preconditions(path, n, m, ptr, adj, sha):
    """Gate W7.2 で確認した異常条件と一致しなければ PreconditionError。"""
    require(sha == EXPECTED_LEGACY_SHA256,
            f"旧ファイルの SHA256 が想定と異なる\n  expected={EXPECTED_LEGACY_SHA256}\n  actual  ={sha}")
    require(n == EXPECTED_N and m == EXPECTED_M,
            f"ヘッダが想定と異なる: n={n} m={m} (expected n={EXPECTED_N} m={EXPECTED_M})")
    require(len(ptr) == n + 1, f"len(ptr)={len(ptr)} != n+1={n + 1}")
    require(int(ptr[0]) == 0, f"ptr[0]={int(ptr[0])} != 0")
    require(bool(np.all(np.diff(ptr) >= 0)), "ptr が単調非減少でない")
    require(int(ptr[-1]) == len(adj), f"ptr[n]={int(ptr[-1])} != len(adj)={len(adj)}")

    shortage = 2 * m - len(adj)
    require(shortage == EXPECTED_SHORTAGE,
            f"隣接要素の不足数が想定と異なる: shortage={shortage} (expected {EXPECTED_SHORTAGE})")
    require(int(ptr[1]) == 0, f"頂点 0 が孤立していない (ptr[1]={int(ptr[1])})")
    require(int(adj.min()) == 1, f"adj の最小値が 1 でない: {int(adj.min())} (1-based ではない)")
    require(int(adj.max()) == n, f"adj の最大値が n={n} でない: {int(adj.max())}")
    require(int(np.count_nonzero(adj == 0)) == 0, "adj に値 0 が存在する (1-based 前提と矛盾)")

    oor = int(np.count_nonzero(adj >= n))
    require(oor == EXPECTED_SHORTAGE,
            f"範囲外 ID (>= n) の個数が想定と異なる: {oor} (expected {EXPECTED_SHORTAGE})")

    deg = np.diff(ptr)
    require(int(np.count_nonzero(deg == 0)) == 1 and int(np.flatnonzero(deg == 0)[0]) == 0,
            "孤立行が頂点 0 のみではない")

    missing_deg = 2 * m - int(ptr[-1])
    occ = int(np.count_nonzero(adj == n))
    require(missing_deg == occ,
            f"欠落行の次数 {missing_deg} と値 {n} の出現数 {occ} が一致しない "
            f"(self-loop の有無を一意に決定できない)")
    return missing_deg, occ


def reconstruct_missing_row(n, ptr, adj):
    """旧頂点 n (1-based) の隣接行を対称性から再構成する。"""
    pos = np.flatnonzero(adj == n)                       # 他行での n の出現位置
    owners = np.searchsorted(ptr, pos, side="right") - 1  # その要素を持つ行 = n の隣接頂点
    by_position = owners.copy()                           # 逆辺の出現位置順
    ascending = np.sort(owners)                           # 昇順ソート
    require(np.array_equal(by_position, ascending),
            "再構成行の並びが規約により異なる (逆辺出現位置順 != 昇順)。"
            "順序が一意に定まらないため停止する")
    require(int(np.count_nonzero(owners == n)) == 0,
            f"再構成行に自己参照が現れた (旧頂点 {n} の self-loop は検出できない)")
    return ascending


def verify_repaired(n, m, new_ptr, new_adj, old_ptr, old_adj):
    """修復後グラフの不変条件を検証する。"""
    require(len(new_ptr) == n + 1, f"new len(ptr)={len(new_ptr)} != n+1")
    require(int(new_ptr[0]) == 0, "new ptr[0] != 0")
    require(bool(np.all(np.diff(new_ptr) >= 0)), "new ptr が単調非減少でない")
    require(int(new_ptr[-1]) == 2 * m, f"new ptr[n]={int(new_ptr[-1])} != 2m={2 * m}")
    require(len(new_adj) == 2 * m, f"new len(adj)={len(new_adj)} != 2m={2 * m}")
    require(int(new_adj.min()) >= 0 and int(new_adj.max()) < n,
            f"new adj に範囲外 ID: [{int(new_adj.min())}, {int(new_adj.max())}] (valid [0,{n - 1}])")

    deg = np.diff(new_ptr)
    require(int(deg.sum()) == len(new_adj), "new sum(degree) != len(adj)")

    # 対称性 (多重度込み)
    src = np.repeat(np.arange(n, dtype=np.int64), deg)
    key_f = np.sort(src * np.int64(n) + new_adj)
    key_r = np.sort(new_adj * np.int64(n) + src)
    require(np.array_equal(key_f, key_r), "修復後グラフが対称でない")

    # self-loop / duplicate の多重度が保存されているか (旧: 頂点 n の行を除く)
    old_deg = np.diff(old_ptr)
    old_src = np.repeat(np.arange(len(old_deg), dtype=np.int64), old_deg)
    old_self = int(np.count_nonzero(old_src == old_adj))
    new_self = int(np.count_nonzero(src == new_adj))
    require(old_self == new_self,
            f"self-loop 要素数が変化した: old={old_self} new={new_self}")

    _, old_counts = np.unique(old_src * np.int64(n + 1) + old_adj, return_counts=True)
    _, new_counts = np.unique(src * np.int64(n) + new_adj, return_counts=True)
    old_hist = dict(zip(*[a.tolist() for a in np.unique(old_counts, return_counts=True)]))
    new_hist = dict(zip(*[a.tolist() for a in np.unique(new_counts, return_counts=True)]))
    # 旧は欠落行 (7 要素) を欠くため、その分だけ多重度 1 のペア数が増える想定
    return {
        "self_loop_entries": new_self,
        "old_multiplicity_hist": old_hist,
        "new_multiplicity_hist": new_hist,
    }


def write_csr(path, n, m, ptr, adj):
    """3 行 CSR テキストを書き出す (単一スペース区切り・LF・末尾改行 1 個)。"""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"{n} {m}\n")
        for i in range(0, len(ptr), WRITE_CHUNK):
            f.write(" ".join(map(str, ptr[i:i + WRITE_CHUNK].tolist())))
            f.write(" " if i + WRITE_CHUNK < len(ptr) else "\n")
        for i in range(0, len(adj), WRITE_CHUNK):
            f.write(" ".join(map(str, adj[i:i + WRITE_CHUNK].tolist())))
            f.write(" " if i + WRITE_CHUNK < len(adj) else "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="325557_3216152 の決定的修復 (Gate W7.3A)")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="旧 (malformed) CSR グラフ")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="修復版の出力先 (別名必須)")
    ap.add_argument("--dry-run", action="store_true", help="検査のみ行い書き出さない")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] input not found: {args.input}", file=sys.stderr)
        return 2
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        print("[ERROR] 旧ファイルを上書きしようとしている (別名が必要)", file=sys.stderr)
        return 2

    print(f"[1/5] 旧ファイル読込: {args.input}")
    sha = sha256_of(args.input)
    print(f"      sha256={sha}")
    print(f"      size={os.path.getsize(args.input)} bytes")

    try:
        n, m, ptr, adj = read_legacy(args.input)
        print(f"[2/5] 前提条件検査 (Gate W7.2 の異常条件と一致するか)")
        missing_deg, occ = check_preconditions(args.input, n, m, ptr, adj, sha)
        print(f"      n={n} m={m} 2m={2 * m} len(ptr)={len(ptr)} len(adj)={len(adj)}")
        print(f"      欠落: 最終頂点 (1-based {n}) の行, 次数={missing_deg}, "
              f"値 {n} の出現数={occ} -> self-loop={((missing_deg - occ) // 2)} 本")

        print(f"[3/5] 欠落行の再構成 (対称性)")
        missing_row = reconstruct_missing_row(n, ptr, adj)
        print(f"      旧頂点 {n} の隣接 (1-based, 多重度込み): {missing_row.tolist()}")
        print(f"      -> 0-based:                            {(missing_row - 1).tolist()}")

        print(f"[4/5] relabelling (k -> k-1; source 行と destination の両方)")
        # new row i (0-based) = old vertex i+1 の行
        #   i = 0..n-2 -> old_ptr[i+1] .. old_ptr[i+2]  (旧行 1..n-1; 旧行 0 は空)
        #   i = n-1    -> 再構成した旧頂点 n の行
        new_ptr = np.empty(n + 1, dtype=np.int64)
        new_ptr[:n] = ptr[1:]          # ptr[1..n] : 旧行 1..n-1 の開始位置 + 旧頂点 n の開始位置
        new_ptr[n] = 2 * m
        new_adj = np.concatenate([adj, missing_row]) - 1

        stats = verify_repaired(n, m, new_ptr, new_adj, ptr, adj)
        print(f"      self-loop 要素数: {stats['self_loop_entries']} (= {stats['self_loop_entries'] // 2} 本, 保存)")
        print(f"      多重度ヒスト old={stats['old_multiplicity_hist']} new={stats['new_multiplicity_hist']}")
        print(f"      検証: 対称性 OK / 範囲 [0,{n - 1}] OK / len(adj)=2m={2 * m} OK / n,m 維持 OK")
    except PreconditionError as e:
        print(f"[ERROR] 前提条件の不一致のため修復を中止した (出力なし):\n  {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("[5/5] --dry-run: 書き出しをスキップした")
        return 0

    print(f"[5/5] 書き出し: {args.output}")
    write_csr(args.output, n, m, new_ptr, new_adj)
    out_sha = sha256_of(args.output)
    print(f"      sha256={out_sha}")
    print(f"      size={os.path.getsize(args.output)} bytes")
    print("OK: 修復版を生成した (旧ファイルは未変更)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
