#!/usr/bin/env python3
"""
BC 正確性検証スクリプト: 参照実装 (sequential) と各実装の全 BC 値を比較する
使用方法:
    python3 compare_bc.py <reference.txt> <result.txt> [--rel-tol 1e-6]
出力:
    - 一致した場合: PASS メッセージ
    - 不一致の場合: 最大誤差・不一致頂点数を報告して非ゼロ終了
"""
import sys
import argparse
import math

def load_bc(filename):
    """BC ダンプファイルを読み込む。先頭の '#' 行はスキップ"""
    values = []
    header = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                # 例: # impl=Sequential graph=benchmark_7000_41459 nodes=7000
                for token in line[1:].strip().split():
                    k, _, v = token.partition('=')
                    header[k] = v
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                values.append(float(parts[1]))
    return header, values

def main():
    parser = argparse.ArgumentParser(description='BC 正確性検証')
    parser.add_argument('reference', help='参照 BC ファイル (sequential --dump-bc の出力)')
    parser.add_argument('result',    help='検証対象 BC ファイル')
    parser.add_argument('--rel-tol', type=float, default=1e-6,
                        help='許容相対誤差 (デフォルト: 1e-6)')
    args = parser.parse_args()

    ref_hdr,  ref_vals  = load_bc(args.reference)
    res_hdr,  res_vals  = load_bc(args.result)

    print(f"  Reference : impl={ref_hdr.get('impl','?')} graph={ref_hdr.get('graph','?')} nodes={len(ref_vals)}")
    print(f"  Target    : impl={res_hdr.get('impl','?')} graph={res_hdr.get('graph','?')} nodes={len(res_vals)}")
    print(f"  Tolerance : rel_tol={args.rel_tol:.0e}")

    if len(ref_vals) != len(res_vals):
        print(f"FAIL: node count mismatch {len(ref_vals)} vs {len(res_vals)}", file=sys.stderr)
        sys.exit(1)

    n = len(ref_vals)
    max_rel_err = 0.0
    max_abs_err = 0.0
    fail_count  = 0
    fail_examples = []

    for i in range(n):
        r = ref_vals[i]
        v = res_vals[i]
        abs_err = abs(r - v)
        denom   = max(abs(r), 1e-15)  # 0 に対する除算を防ぐ
        rel_err = abs_err / denom

        if rel_err > args.rel_tol:
            fail_count += 1
            if len(fail_examples) < 5:
                fail_examples.append((i, r, v, rel_err))
        if rel_err > max_rel_err:
            max_rel_err = rel_err
            max_abs_err = abs_err

    print(f"  Nodes checked : {n}")
    print(f"  Max rel error : {max_rel_err:.3e}")
    print(f"  Max abs error : {max_abs_err:.3e}")
    print(f"  Mismatches    : {fail_count} / {n}")

    if fail_count > 0:
        print(f"\n  First {len(fail_examples)} mismatches:")
        for (idx, ref, got, rerr) in fail_examples:
            print(f"    node {idx:7d}: ref={ref:.10e}  got={got:.10e}  rel_err={rerr:.3e}")
        print(f"\nFAIL: {fail_count} nodes exceed rel_tol={args.rel_tol:.0e}", file=sys.stderr)
        sys.exit(1)

    print("PASS: All BC values match within tolerance.")
    sys.exit(0)

if __name__ == '__main__':
    main()
