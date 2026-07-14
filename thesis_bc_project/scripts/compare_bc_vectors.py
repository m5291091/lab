#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2 つの BC ダンプ (--dump-bc 出力) の数値一致を検証する。

--dump-bc 形式:
    # impl=... graph=... nodes=N
    <node_idx>\t<bc_value>          (bc_value は %.15e)

比較項目:
  - ベクトル長 (各ファイル)
  - 欠損 index 数 (片方にしか無い index)
  - 最大絶対誤差 / 最大相対誤差 (共通 index 上) とその index
  - Max BC の index/value (各ファイル)
  - 許容誤差内か (rel_tol / abs_tol)

使い方:
    compare_bc_vectors.py FILE_A FILE_B [--label-a LA] [--label-b LB]
                          [--rel-tol 1e-6] [--abs-tol 1e-3] [--out summary.md]

巨大な BC ベクトル自体は保存しない。サマリ (markdown) のみ出力する。
"""
import argparse
import hashlib
import math
import sys


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dump(path):
    """dump ファイルを {idx: value} と header 文字列で返す。"""
    vals = {}
    header = ""
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                header = line.lstrip("# ").strip()
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
                v = float(parts[1])
            except ValueError:
                continue
            vals[idx] = v
    return vals, header


def max_bc(vals):
    if not vals:
        return (None, None)
    idx = max(vals, key=lambda i: vals[i])
    return idx, vals[idx]


def display_value(value):
    """比較対象の値を補正せず、人が識別できる形で表示する。"""
    if math.isnan(value):
        return "NaN"
    if value == math.inf:
        return "+Inf"
    if value == -math.inf:
        return "-Inf"
    return repr(value)


def nonfinite_detail(items):
    if not items:
        return "なし"
    return "; ".join(
        f"index {idx}, value {display_value(value)}" for idx, value in items
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--rel-tol", type=float, default=1e-6)
    ap.add_argument("--abs-tol", type=float, default=1e-3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="サマリに追記する 'キー=値' メタデータ (checkpoint SHA / job ID / batch 等)")
    args = ap.parse_args()

    la = args.label_a or args.file_a
    lb = args.label_b or args.file_b

    # 一時ベクトル削除前にハッシュを保存できるよう SHA256 を先に計算する
    sha_a = sha256_of(args.file_a)
    sha_b = sha256_of(args.file_b)

    a, ha = load_dump(args.file_a)
    b, hb = load_dump(args.file_b)

    ka, kb = set(a), set(b)
    common = ka & kb
    only_a = ka - kb
    only_b = kb - ka
    nonfinite_a = sorted((i, a[i]) for i in ka if not math.isfinite(a[i]))
    nonfinite_b = sorted((i, b[i]) for i in kb if not math.isfinite(b[i]))
    nonfinite_count = len(nonfinite_a) + len(nonfinite_b)

    # 共通 index 上の有限値だけで誤差を計算する。非有限値は別途無条件 FAIL にする。
    max_abs = 0.0
    max_abs_idx = None
    max_rel = 0.0
    max_rel_idx = None
    mixed_mismatch = 0  # 混合許容 (numpy allclose 式) を満たさない要素数
    for i in sorted(common):
        va, vb = a[i], b[i]
        if not math.isfinite(va) or not math.isfinite(vb):
            continue
        d = abs(va - vb)
        if max_abs_idx is None or d > max_abs:
            max_abs, max_abs_idx = d, i
        denom = max(abs(va), abs(vb))
        rel = d / denom if denom > 0 else 0.0
        if max_rel_idx is None or rel > max_rel:
            max_rel, max_rel_idx = rel, i
        # 混合許容: abs_diff <= abs_tol + rel_tol * max(|a|,|b|)
        if d > args.abs_tol + args.rel_tol * denom:
            mixed_mismatch += 1

    mba_i, mba_v = max_bc(a)
    mbb_i, mbb_v = max_bc(b)

    structural_ok = (len(only_a) == 0 and len(only_b) == 0 and len(a) == len(b))
    abs_only_ok = (max_abs <= args.abs_tol)
    # 総合判定: 構造一致かつ混合許容で不一致 0 なら PASS。
    # 絶対許容単独では超過 (abs_only_ok=False) でも混合許容を満たせば PASS(absolute-only warning)。
    if nonfinite_count != 0:
        overall = ("FAIL (非有限値あり: "
                   f"A={len(nonfinite_a)} 件, B={len(nonfinite_b)} 件)")
    elif not structural_ok:
        overall = "FAIL (長さ不一致 または 欠損 index あり)"
    elif mixed_mismatch == 0:
        overall = "PASS" if abs_only_ok else "PASS (absolute-only warning)"
    else:
        overall = f"FAIL (混合許容で不一致 {mixed_mismatch} 件)"
    within = overall.startswith("PASS")

    lines = []
    lines.append(f"# BC ベクトル数値比較: {la} vs {lb}")
    lines.append("")
    lines.append(f"- 入力A: `{args.file_a}`  (header: {ha})")
    lines.append(f"  - SHA256: `{sha_a}`")
    lines.append(f"- 入力B: `{args.file_b}`  (header: {hb})")
    lines.append(f"  - SHA256: `{sha_b}`")
    if args.extra:
        lines.append("")
        for kv in args.extra:
            if "=" in kv:
                k, v = kv.split("=", 1)
                lines.append(f"- {k.strip()}: {v.strip()}")
            else:
                lines.append(f"- {kv}")
    lines.append("")
    lines.append("| 項目 | 値 |")
    lines.append("|:--|:--|")
    lines.append(f"| ベクトル長 A | {len(a)} |")
    lines.append(f"| ベクトル長 B | {len(b)} |")
    lines.append(f"| 共通 index 数 | {len(common)} |")
    lines.append(f"| 欠損 index 数 (A のみ) | {len(only_a)} |")
    lines.append(f"| 欠損 index 数 (B のみ) | {len(only_b)} |")
    lines.append(f"| 非有限値数 A | {len(nonfinite_a)} |")
    lines.append(f"| 非有限値詳細 A (vector A: {la}) | {nonfinite_detail(nonfinite_a)} |")
    lines.append(f"| 非有限値数 B | {len(nonfinite_b)} |")
    lines.append(f"| 非有限値詳細 B (vector B: {lb}) | {nonfinite_detail(nonfinite_b)} |")
    lines.append(f"| 最大絶対誤差 | {max_abs:.6e} (index {max_abs_idx}) |")
    if max_abs_idx is not None:
        lines.append("| 最大絶対誤差 index の値 | "
                     f"A={display_value(a[max_abs_idx])}, "
                     f"B={display_value(b[max_abs_idx])} |")
    else:
        lines.append("| 最大絶対誤差 index の値 | A=not_available, B=not_available |")
    lines.append(f"| 最大相対誤差 | {max_rel:.6e} (index {max_rel_idx}) |")
    if max_rel_idx is not None:
        lines.append("| 最大相対誤差 index の値 | "
                     f"A={display_value(a[max_rel_idx])}, "
                     f"B={display_value(b[max_rel_idx])} |")
    else:
        lines.append("| 最大相対誤差 index の値 | A=not_available, B=not_available |")
    lines.append(f"| Max BC A | index {mba_i}, value {mba_v:.6f} |" if mba_i is not None
                 else "| Max BC A | — |")
    lines.append(f"| Max BC B | index {mbb_i}, value {mbb_v:.6f} |" if mbb_i is not None
                 else "| Max BC B | — |")
    lines.append(f"| 許容値 | rel_tol={args.rel_tol:g}, abs_tol={args.abs_tol:g} |")
    lines.append(f"| 絶対許容のみ (abs_diff ≤ abs_tol) | "
                 f"{'OK' if abs_only_ok else 'WARN (超過; 巨大 magnitude で不適切な場合あり)'} |")
    lines.append(f"| 混合許容 abs_diff ≤ abs_tol+rel_tol·max(\\|a\\|,\\|b\\|) 不一致要素数 | "
                 f"{mixed_mismatch} |")
    lines.append(f"| **総合判定** | **{overall}** |")
    lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\n[written] {args.out}", file=sys.stderr)
    return 0 if within else 3


if __name__ == "__main__":
    sys.exit(main())
