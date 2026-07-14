#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""実験結果 TSV の自動異常検査 (FAIL / OOM / TIMEOUT / 欠損試行)。

対象:
  - PathMerge 掃引 TSV : Config<TAB>Graph<TAB>Trial<TAB>Time_sec<TAB>GTEPS
  - ベンチ TSV         : Impl<TAB>Graph<TAB>Nodes<TAB>Edges<TAB>Trial<TAB>Time_sec<TAB>GTEPS

検査:
  1. Time 列が FAIL / TIMEOUT / OOM 等のマーカー = 異常。
  2. (Config|Impl, Graph) 毎の有効試行数が --expect 未満 = 欠損試行。
  3. 併せてログ (--logs) の out-of-memory / bad_alloc / cudaError を grep。

異常が 1 件でもあれば非 0 終了 (STOP & REPORT を促す)。

使い方:
  check_results_integrity.py TSV [TSV ...] [--expect N] [--logs LOG ...]
"""
import argparse
import collections
import csv
import os
import re
import sys

MARKERS = {"FAIL", "TIMEOUT", "OOM", "ERROR", "NAN"}
LOG_PATTERNS = re.compile(
    r"out of memory|bad_alloc|cudaError|cudaErrorMemoryAllocation|"
    r"std::bad_alloc|Segmentation fault|core dumped",
    re.IGNORECASE)


def col_time_and_keys(row):
    """行から (impl/config, graph, trial, time_str) を列数に応じて抽出。"""
    n = len(row)
    if n >= 7:                    # ベンチ TSV (Impl Graph Nodes Edges Trial Time GTEPS)
        return row[0], row[1], row[4], row[5]
    if n >= 5:                    # 掃引 TSV (Config Graph Trial Time GTEPS)
        return row[0], row[1], row[2], row[3]
    return None, None, None, None


def scan_tsv(path):
    anomalies = []
    counts = collections.defaultdict(int)
    if not os.path.exists(path):
        return [f"[MISSING FILE] {path}"], counts
    with open(path) as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row or row[0] in ("Config", "Implementation", "Impl"):
                continue
            impl, graph, trial, tstr = col_time_and_keys(row)
            if impl is None:
                continue
            up = (tstr or "").strip().upper()
            if up in MARKERS or not tstr:
                anomalies.append(
                    f"[{up or 'EMPTY'}] {os.path.basename(path)}: {impl} {graph} trial={trial}")
                continue
            try:
                float(tstr)
                counts[(impl, graph)] += 1
            except ValueError:
                anomalies.append(
                    f"[NON-NUMERIC] {os.path.basename(path)}: {impl} {graph} trial={trial} time='{tstr}'")
    return anomalies, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", nargs="+")
    ap.add_argument("--expect", type=int, default=0,
                    help="(Config|Impl,Graph) 毎に要求する最小有効試行数 (0=チェックしない)")
    ap.add_argument("--logs", nargs="*", default=[])
    args = ap.parse_args()

    all_anom = []
    total_counts = collections.defaultdict(int)
    for path in args.tsv:
        anom, counts = scan_tsv(path)
        all_anom.extend(anom)
        for k, v in counts.items():
            total_counts[k] += v

    # 欠損試行チェック
    if args.expect > 0:
        for (impl, graph), c in sorted(total_counts.items()):
            if c < args.expect:
                all_anom.append(
                    f"[MISSING TRIALS] {impl} {graph}: 有効試行 {c} < 要求 {args.expect}")

    # ログ grep
    for lg in args.logs:
        if not os.path.exists(lg):
            all_anom.append(f"[MISSING LOG] {lg}")
            continue
        with open(lg, errors="ignore") as f:
            for i, line in enumerate(f, 1):
                if LOG_PATTERNS.search(line):
                    all_anom.append(f"[LOG:{os.path.basename(lg)}:{i}] {line.strip()[:120]}")

    print("=== 実験結果 整合性検査 ===")
    print(f"対象 TSV: {len(args.tsv)}  有効 (impl,graph) 群: {len(total_counts)}")
    for (impl, graph), c in sorted(total_counts.items()):
        print(f"  - {impl} @ {graph}: 有効試行 n={c}")
    if all_anom:
        print(f"\n❌ 異常 {len(all_anom)} 件:")
        for a in all_anom:
            print(f"  {a}")
        print("\n=> STOP & REPORT: 原因と job ID を確認し、古い binary で継続しないこと。")
        return 1
    print("\n✅ 異常なし (FAIL/OOM/TIMEOUT/欠損なし)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
