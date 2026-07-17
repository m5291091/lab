#!/usr/bin/env python3
"""Compare two BC dumps without changing their recorded values.

The historical CLI is retained.  ``--expected-length`` adds an explicit index
domain; malformed, empty, duplicate, missing, out-of-range, and non-finite
vectors fail before numerical tolerance can yield PASS.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from validate_bc_vector import scan_vector


def display_value(value):
    if math.isnan(value):
        return "NaN"
    if value == math.inf:
        return "+Inf"
    if value == -math.inf:
        return "-Inf"
    return repr(value)


def max_bc(values):
    finite = {index: value for index, value in values.items() if math.isfinite(value)}
    if not finite:
        return None, None
    index = max(finite, key=lambda item: finite[item])
    return index, finite[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    parser.add_argument("--rel-tol", type=float, default=1e-6)
    parser.add_argument("--abs-tol", type=float, default=1e-3)
    parser.add_argument("--expected-length", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="summary metadata as key=value (historical option)",
    )
    args = parser.parse_args()

    if args.rel_tol < 0 or args.abs_tol < 0:
        parser.error("tolerances must be non-negative")

    label_a = args.label_a or args.file_a
    label_b = args.label_b or args.file_b
    try:
        scan_a = scan_vector(args.file_a, args.expected_length)
        scan_b = scan_vector(args.file_b, args.expected_length)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    a, b = scan_a.values, scan_b.values
    keys_a, keys_b = set(a), set(b)
    common = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a

    max_abs = 0.0
    max_abs_index = None
    max_rel = 0.0
    max_rel_index = None
    mixed_mismatch = 0
    for index in sorted(common):
        value_a, value_b = a[index], b[index]
        if not math.isfinite(value_a) or not math.isfinite(value_b):
            continue
        difference = abs(value_a - value_b)
        if max_abs_index is None or difference > max_abs:
            max_abs, max_abs_index = difference, index
        denominator = max(abs(value_a), abs(value_b))
        relative = difference / denominator if denominator > 0 else 0.0
        if max_rel_index is None or relative > max_rel:
            max_rel, max_rel_index = relative, index
        # Historical mixed tolerance; intentionally unchanged.
        if difference > args.abs_tol + args.rel_tol * denominator:
            mixed_mismatch += 1

    structural_ok = (
        scan_a.valid
        and scan_b.valid
        and keys_a == keys_b
        and len(a) == len(b)
        and len(a) > 0
    )
    abs_only_ok = max_abs <= args.abs_tol
    if not scan_a.valid or not scan_b.valid:
        overall = "FAIL (vector integrity violation)"
    elif not structural_ok:
        overall = "FAIL (length/index mismatch)"
    elif mixed_mismatch == 0:
        overall = "PASS" if abs_only_ok else "PASS (absolute-only warning)"
    else:
        overall = f"FAIL (混合許容で不一致 {mixed_mismatch} 件)"
    passed = overall.startswith("PASS")

    max_a_index, max_a_value = max_bc(a)
    max_b_index, max_b_value = max_bc(b)
    lines = [
        f"# BC ベクトル数値比較: {label_a} vs {label_b}",
        "",
        f"- 入力A: `{args.file_a}`  (header: {scan_a.header})",
        f"  - SHA256: `{scan_a.sha256}`",
        f"- 入力B: `{args.file_b}`  (header: {scan_b.header})",
        f"  - SHA256: `{scan_b.sha256}`",
    ]
    if args.extra:
        lines.append("")
        for item in args.extra:
            key, separator, value = item.partition("=")
            lines.append(f"- {key.strip()}: {value.strip()}" if separator else f"- {item}")
    lines += [
        "",
        "| 項目 | 値 |",
        "|:--|:--|",
        f"| 期待ベクトル長 | {args.expected_length if args.expected_length is not None else 'not_recorded'} |",
        f"| ベクトル長 A | {len(a)} |",
        f"| ベクトル長 B | {len(b)} |",
        f"| データ行数 A / B | {scan_a.data_rows} / {scan_b.data_rows} |",
        f"| 共通 index 数 | {len(common)} |",
        f"| 欠損 index 数 (A のみ) | {len(only_a)} |",
        f"| 欠損 index 数 (B のみ) | {len(only_b)} |",
        f"| expected domain 欠損 A / B | {scan_a.missing_indices} / {scan_b.missing_indices} |",
        f"| duplicate index A / B | {scan_a.duplicate_indices} / {scan_b.duplicate_indices} |",
        f"| out-of-range index A / B | {scan_a.out_of_range_indices} / {scan_b.out_of_range_indices} |",
        f"| parse error A / B | {scan_a.parse_errors} / {scan_b.parse_errors} |",
        f"| 値列欠損 A / B | {scan_a.value_column_missing} / {scan_b.value_column_missing} |",
        f"| NaN A / B | {scan_a.nan_values} / {scan_b.nan_values} |",
        f"| +Inf A / B | {scan_a.positive_inf_values} / {scan_b.positive_inf_values} |",
        f"| -Inf A / B | {scan_a.negative_inf_values} / {scan_b.negative_inf_values} |",
        f"| 最大絶対誤差 | {max_abs:.6e} (index {max_abs_index}) |",
        f"| 最大相対誤差 | {max_rel:.6e} (index {max_rel_index}) |",
    ]
    if max_abs_index is not None:
        lines.append(
            "| 最大絶対誤差 index の値 | "
            f"A={display_value(a[max_abs_index])}, B={display_value(b[max_abs_index])} |"
        )
    else:
        lines.append("| 最大絶対誤差 index の値 | A=not_available, B=not_available |")
    if max_rel_index is not None:
        lines.append(
            "| 最大相対誤差 index の値 | "
            f"A={display_value(a[max_rel_index])}, B={display_value(b[max_rel_index])} |"
        )
    else:
        lines.append("| 最大相対誤差 index の値 | A=not_available, B=not_available |")
    lines += [
        (
            f"| Max BC A | index {max_a_index}, value {max_a_value:.6f} |"
            if max_a_index is not None
            else "| Max BC A | — |"
        ),
        (
            f"| Max BC B | index {max_b_index}, value {max_b_value:.6f} |"
            if max_b_index is not None
            else "| Max BC B | — |"
        ),
        f"| 許容値 | rel_tol={args.rel_tol:g}, abs_tol={args.abs_tol:g} |",
        f"| 絶対許容のみ (abs_diff ≤ abs_tol) | {'OK' if abs_only_ok else 'WARN (超過; 巨大 magnitude で不適切な場合あり)'} |",
        "| 混合許容 abs_diff ≤ abs_tol+rel_tol·max(\\|a\\|,\\|b\\|) 不一致要素数 | "
        f"{mixed_mismatch} |",
        f"| **総合判定** | **{overall}** |",
        "",
    ]
    text = "\n".join(lines)
    print(text)

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as stream:
                stream.write(text + "\n")
        except OSError as error:
            print(f"ERROR: cannot write {args.out}: {error}", file=sys.stderr)
            return 2
        print(f"\n[written] {args.out}", file=sys.stderr)

    payload = {
        "status": "PASS" if passed else "FAIL",
        "overall": overall,
        "expected_length": args.expected_length,
        "length_a": len(a),
        "length_b": len(b),
        "missing_a": len(only_a),
        "missing_b": len(only_b),
        "mismatched_elements": mixed_mismatch,
        "max_abs_error": max_abs,
        "max_abs_index": max_abs_index,
        "max_rel_error": max_rel,
        "max_rel_index": max_rel_index,
        "sha256_a": scan_a.sha256,
        "sha256_b": scan_b.sha256,
        "vector_a": scan_a.summary(),
        "vector_b": scan_b.summary(),
    }
    if args.json:
        try:
            with open(args.json, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
        except OSError as error:
            print(f"ERROR: cannot write {args.json}: {error}", file=sys.stderr)
            return 2
    return 0 if passed else 3


if __name__ == "__main__":
    sys.exit(main())
