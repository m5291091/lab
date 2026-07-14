#!/usr/bin/env python3
"""
UM / HBM3 オーバーサブスクライブ実験の集計スクリプト。

入力:
  1. oversubscribe_results.tsv
  2. um_experiment.log

出力:
  - BATCH サイズごとの gpu_opt / gpu_opt_pure の
    GTEPS median / std / Time median / status テーブル
  - gpu_opt の [GPU Phase] ログから算出した
    prefetch_ms / kernel_ms 比率テーブル

Usage:
  python3 summarize_oversubscribe.py [tsv_path] [log_path]

デフォルト:
  tsv_path = <script_dir>/../raw_data/memory_scalability/ (impl別 oversubscribe_results_*.tsv を再帰収集)
  log_path = <script_dir>/../raw_data/memory_scalability/ (impl別 um_experiment_*.log を再帰収集)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict

try:
    import numpy as np

    def _median(vals: list[float]) -> float:
        return float(np.median(vals))

    def _std(vals: list[float]) -> float:
        return float(np.std(vals, ddof=0))

except ImportError:

    def _median(vals: list[float]) -> float:
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

    def _std(vals: list[float]) -> float:
        n = len(vals)
        if n == 0:
            return 0.0
        mu = sum(vals) / n
        return math.sqrt(sum((x - mu) ** 2 for x in vals) / n)


IMPL_DISPLAY = {
    "gpu_opt": "gpu_opt (UM streaming)",
    "gpu_opt_pure": "gpu_opt_pure (single-alloc)",
    "gpu_opt_pure_chunked": "gpu_opt_pure_chunked (manual chunking)",
}
IMPL_ORDER = ["gpu_opt_pure", "gpu_opt_pure_chunked", "gpu_opt"]
SUCCESS = "SUCCESS"

RUNNING_RE = re.compile(r"^Running:\s+([A-Za-z0-9_]+)\s+on\s+(.+?)\.\.\.$")
RUN_HEADER_RE = re.compile(r"^===\s*([A-Za-z0-9_]+)\s+batch=(\d+)\s+trial=(\d+)\s+rc=(\d+)\s*===$")
MEM_BATCH_RE = re.compile(r"\bBATCH=(\d+)\b")
MEM_BATCH_PER_STREAM_RE = re.compile(r"\bbatch_per_stream=(\d+)\b")
GPU_PHASE_RE = re.compile(
    r"\[GPU Phase\]\s*"
    r"BFS wall=([0-9.]+)\s*s.*?"
    r"Backward wall=([0-9.]+)\s*s"
    r"(?:.*?Prefetch cum=([0-9.]+)\s*s)?"
)


def normalize_impl_name(name: str) -> str:
    lowered = name.strip().lower()
    mapping = {
        "gpu_opt": "gpu_opt",
        "gpu_opt_pure": "gpu_opt_pure",
        "gpu_opt_pure_chunked": "gpu_opt_pure_chunked",
        "gpu_opt_(um)": "gpu_opt",
        "gpu_opt_pure_(manual)": "gpu_opt_pure",
        "gpu_opt_(manual)": "gpu_opt_pure",
        "gpu_opt_(pure)": "gpu_opt_pure",
        "gpu_opt_pure_(pure)": "gpu_opt_pure",
        "gpu_opt_pure_chunked_(manual)": "gpu_opt_pure_chunked",
        "gpuopt": "gpu_opt",
        "gpuoptpure": "gpu_opt_pure",
        "gpuoptpurechunked": "gpu_opt_pure_chunked",
        "gpu_opt_um": "gpu_opt",
        "gpu_opt_pure_um": "gpu_opt_pure",
        "gpu_opt_pure_manual": "gpu_opt_pure",
        "gpu_opt_pure_gpu": "gpu_opt_pure",
        "gpu_opt_gpu": "gpu_opt",
        "gpu_opt_pure_on": "gpu_opt_pure",
        "gpu_opt_on": "gpu_opt",
        "gpu_opt_pureon": "gpu_opt_pure",
        "gpu_opton": "gpu_opt",
        "gpu_opt_pure...": "gpu_opt_pure",
        "gpu_opt...": "gpu_opt",
        "gpu_opt_pure_chunked...": "gpu_opt_pure_chunked",
        "gpu_opt_pure:": "gpu_opt_pure",
        "gpu_opt:": "gpu_opt",
        "gpu_opt_pure_chunked:": "gpu_opt_pure_chunked",
        "gpu_opt_pure)": "gpu_opt_pure",
        "gpu_opt)": "gpu_opt",
        "gpu_opt_pure_chunked)": "gpu_opt_pure_chunked",
        "gpu_opt_pure]": "gpu_opt_pure",
        "gpu_opt]": "gpu_opt",
        "gpu_opt_pure_chunked]": "gpu_opt_pure_chunked",
        "gpu_opt_pure,": "gpu_opt_pure",
        "gpu_opt,": "gpu_opt",
        "gpu_opt_pure_chunked,": "gpu_opt_pure_chunked",
        "gpu_opt_pure;": "gpu_opt_pure",
        "gpu_opt;": "gpu_opt",
        "gpu_opt_pure_chunked;": "gpu_opt_pure_chunked",
        "gpu_opt_pure.": "gpu_opt_pure",
        "gpu_opt.": "gpu_opt",
        "gpu_opt_pure_chunked.": "gpu_opt_pure_chunked",
        "gpu_opt_pure_": "gpu_opt_pure",
        "gpu_opt_": "gpu_opt",
        "gpu_opt_pure_chunked_": "gpu_opt_pure_chunked",
        "gpu_opt_pure__": "gpu_opt_pure",
        "gpu_opt__": "gpu_opt",
    }
    if lowered in mapping:
        return mapping[lowered]

    if lowered == "gpu_opt_pure_chunked":
        return "gpu_opt_pure_chunked"
    if lowered == "gpu_opt_pure":
        return "gpu_opt_pure"
    if lowered == "gpu_opt":
        return "gpu_opt"

    compact = lowered.replace(" ", "").replace("-", "_")
    if compact == "gpu_opt_pure_chunked":
        return "gpu_opt_pure_chunked"
    if compact == "gpu_opt_pure":
        return "gpu_opt_pure"
    if compact == "gpu_opt":
        return "gpu_opt"

    title_map = {
        "gpu_opt": "gpu_opt",
        "gpu_opt_pure": "gpu_opt_pure",
        "gpu_opt_pure_chunked": "gpu_opt_pure_chunked",
        "gpu_opt_pure_on": "gpu_opt_pure",
        "gpu_opt_on": "gpu_opt",
    }
    title_key = name.strip().lower()
    if title_key in title_map:
        return title_map[title_key]

    pretty = name.strip()
    if pretty == "GPU_Opt":
        return "gpu_opt"
    if pretty == "GPU_Opt_Pure":
        return "gpu_opt_pure"
    if pretty == "GPU_Opt_Pure_Chunked":
        return "gpu_opt_pure_chunked"

    return lowered


def _resolve_files(path: str, pattern: str) -> list:
    """path がディレクトリなら pattern を再帰収集、ファイルなら単一。"""
    import glob as _glob
    if os.path.isdir(path):
        return sorted(_glob.glob(os.path.join(path, "**", pattern), recursive=True))
    return [path]


def load_tsv(path: str) -> dict[tuple[str, int], dict[str, object]]:
    data: dict[tuple[str, int], dict[str, object]] = defaultdict(
        lambda: {
            "gteps": [],
            "time": [],
            "n_ok": 0,
            "n_fail": 0,
            "n_total": 0,
            "statuses": [],
        }
    )

    for fp in _resolve_files(path, "oversubscribe_results_*.tsv"):
        with open(fp, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                impl = normalize_impl_name(row["Implementation"])
                batch = int(row["BatchSize"])
                status = row["Status"].strip()
                key = (impl, batch)
                entry = data[key]
                entry["n_total"] += 1
                entry["statuses"].append(status)

                if status == SUCCESS:
                    entry["n_ok"] += 1
                    entry["gteps"].append(float(row["GTEPS"]))
                    entry["time"].append(float(row["Time_sec"]))
                else:
                    entry["n_fail"] += 1

    return data


def parse_log(path: str) -> dict[tuple[str, int], dict[str, object]]:
    phase_data: dict[tuple[str, int], dict[str, object]] = defaultdict(
        lambda: {
            "bfs_wall_s": [],
            "back_wall_s": [],
            "prefetch_s": [],
            "kernel_s": [],
            "prefetch_ratio": [],
            "n_phase_lines": 0,
            "n_prefetch_lines": 0,
        }
    )

    current_impl: str | None = None
    current_batch: int | None = None

    for _fp in _resolve_files(path, "um_experiment_*.log"):
      current_impl = None
      current_batch = None
      with open(_fp, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            m = RUN_HEADER_RE.match(line)
            if m:
                current_impl = normalize_impl_name(m.group(1))
                current_batch = int(m.group(2))
                continue

            m = RUNNING_RE.match(line)
            if m:
                current_impl = normalize_impl_name(m.group(1))
                current_batch = None
                continue

            m = MEM_BATCH_RE.search(line)
            if m:
                current_batch = int(m.group(1))
                continue

            m = MEM_BATCH_PER_STREAM_RE.search(line)
            if m:
                current_batch = int(m.group(1))
                continue

            m = GPU_PHASE_RE.search(line)
            if m and current_impl and current_batch is not None:
                bfs_wall_s = float(m.group(1))
                back_wall_s = float(m.group(2))
                prefetch_s = float(m.group(3)) if m.group(3) is not None else None
                kernel_s = bfs_wall_s + back_wall_s

                entry = phase_data[(current_impl, current_batch)]
                entry["bfs_wall_s"].append(bfs_wall_s)
                entry["back_wall_s"].append(back_wall_s)
                entry["kernel_s"].append(kernel_s)
                entry["n_phase_lines"] += 1

                if prefetch_s is not None:
                    entry["prefetch_s"].append(prefetch_s)
                    entry["n_prefetch_lines"] += 1
                    if kernel_s > 0.0:
                        entry["prefetch_ratio"].append(prefetch_s / kernel_s)

    return phase_data


def _status_label(entry: dict[str, object]) -> str:
    n_ok = int(entry["n_ok"])
    n_fail = int(entry["n_fail"])
    n_total = int(entry["n_total"])
    statuses = sorted(set(entry["statuses"]))

    if n_total == 0:
        return "N/A"
    if n_ok == n_total:
        return SUCCESS
    if n_fail == n_total and len(statuses) == 1:
        return statuses[0]
    return "PARTIAL"


def _fmt_float(val: float | None, width: int, digits: int) -> str:
    if val is None:
        return f"{'—':>{width}}"
    return f"{val:>{width}.{digits}f}"


def print_summary_table(data: dict[tuple[str, int], dict[str, object]], all_batches: list[int]) -> None:
    col_w = 22
    num_w = 10
    stat_w = 12
    sep = "─" * (8 + (col_w + num_w * 3 + stat_w + 9) * len(IMPL_ORDER))

    header_impl = "".join(
        f"  {IMPL_DISPLAY.get(impl, impl):^{col_w + num_w * 3 + stat_w + 7}}"
        for impl in IMPL_ORDER
    )
    header_sub = "".join(
        f"  {'GTEPS med':>{num_w}}  {'std':>{num_w}}  {'Time med':>{num_w}}  {'Status':<{stat_w}}"
        for _ in IMPL_ORDER
    )

    print()
    print("=== BATCH × 実装サマリ ===")
    print(f"{'BATCH':>8}{header_impl}")
    print(f"{'':>8}{header_sub}")
    print(sep)

    for batch in all_batches:
        row = f"{batch:>8}"
        for impl in IMPL_ORDER:
            entry = data.get(
                (impl, batch),
                {"gteps": [], "time": [], "n_ok": 0, "n_fail": 0, "n_total": 0, "statuses": []},
            )
            gteps = entry["gteps"]
            times = entry["time"]
            med_g = _median(gteps) if gteps else None
            std_g = _std(gteps) if gteps else None
            med_t = _median(times) if times else None
            row += (
                f"  {_fmt_float(med_g, num_w, 4)}"
                f"  {_fmt_float(std_g, num_w, 4)}"
                f"  {_fmt_float(med_t, num_w, 2)}"
                f"  {_status_label(entry):<{stat_w}}"
            )
        print(row)
    print(sep)


def print_prefetch_ratio_table(
    phase_data: dict[tuple[str, int], dict[str, object]],
    all_batches: list[int],
) -> None:
    print()
    print("=== gpu_opt: prefetch / kernel 比率 ([GPU Phase] 由来) ===")
    print(
        f"{'BATCH':>8}  {'Prefetch med(s)':>15}  {'Kernel med(s)':>15}  "
        f"{'Ratio med(%)':>14}  {'Samples':>8}"
    )
    print("─" * 72)

    for batch in all_batches:
        entry = phase_data.get(("gpu_opt", batch))
        if not entry:
            print(f"{batch:>8}  {'—':>15}  {'—':>15}  {'—':>14}  {'0':>8}")
            continue

        prefetch_vals = entry["prefetch_s"]
        kernel_vals = entry["kernel_s"]
        ratio_vals = entry["prefetch_ratio"]

        prefetch_med = _median(prefetch_vals) if prefetch_vals else None
        kernel_med = _median(kernel_vals) if kernel_vals else None
        ratio_med = (_median(ratio_vals) * 100.0) if ratio_vals else None
        samples = int(entry["n_prefetch_lines"]) if prefetch_vals else 0

        print(
            f"{batch:>8}  {_fmt_float(prefetch_med, 15, 4)}"
            f"  {_fmt_float(kernel_med, 15, 4)}"
            f"  {_fmt_float(ratio_med, 14, 3)}"
            f"  {samples:>8}"
        )

    print("─" * 72)
    print("Note: Prefetch cum がログに無い BATCH は '—' を表示します。")


def print_oom_boundary(data: dict[tuple[str, int], dict[str, object]]) -> None:
    print()
    print("=== OOM / FAIL 境界 ===")
    for impl in IMPL_ORDER:
        ok_batches = sorted(
            batch
            for (impl_name, batch), entry in data.items()
            if impl_name == impl and int(entry["n_ok"]) > 0
        )
        fail_batches = sorted(
            batch
            for (impl_name, batch), entry in data.items()
            if impl_name == impl and int(entry["n_ok"]) == 0 and int(entry["n_total"]) > 0
        )

        label = IMPL_DISPLAY.get(impl, impl)
        if ok_batches:
            msg = f"  {label}: 最大成功 BATCH = {max(ok_batches)}"
        else:
            msg = f"  {label}: 成功なし"

        if fail_batches:
            msg += f" / 失敗開始 BATCH = {min(fail_batches)}"
        print(msg)


def build_parser() -> argparse.ArgumentParser:
    default_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "raw_data",
        "memory_scalability",
    )
    parser = argparse.ArgumentParser(description="HBM3 oversubscribe 実験の集計")
    parser.add_argument(
        "tsv_path",
        nargs="?",
        default=default_dir,
        help="oversubscribe_results_*.tsv のパス、または raw_data/memory_scalability ディレクトリ（impl別を再帰収集）",
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        default=default_dir,
        help="um_experiment_*.log のパス、または raw_data/memory_scalability ディレクトリ（impl別を再帰収集）",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not os.path.exists(args.tsv_path):
        print(f"[ERROR] TSV not found: {args.tsv_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.log_path):
        print(f"[ERROR] Log not found: {args.log_path}", file=sys.stderr)
        sys.exit(1)

    data = load_tsv(args.tsv_path)
    phase_data = parse_log(args.log_path)
    all_batches = sorted({batch for (_, batch) in data.keys()} | {batch for (_, batch) in phase_data.keys()})

    if not all_batches:
        print("[ERROR] No batch data found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n入力 TSV : {args.tsv_path}")
    print(f"入力 Log : {args.log_path}")
    print(f"BATCH 範囲: {min(all_batches)} – {max(all_batches)}")

    print_summary_table(data, all_batches)
    print_prefetch_ratio_table(phase_data, all_batches)
    print_oom_boundary(data)


if __name__ == "__main__":
    main()
