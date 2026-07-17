#!/usr/bin/env python3
"""Validate Series C as an exact 8-configuration factorial result set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys


EXPECTED_CONFIGS = {
    f"Ablation_H{h}_W{w}_A{a}"
    for h in (0, 1)
    for w in (0, 1)
    for a in (0, 1)
}
FAILURE_RE = re.compile(
    r"(?:\bFAIL(?:ED)?\b|\bOOM\b|out of memory|\bTIMEOUT\b|CUDA\s+error|cudaError|\bFATAL\b)",
    re.IGNORECASE,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate complete Series C TSV")
    parser.add_argument("tsv")
    parser.add_argument("--expected-trials", type=int, default=5)
    parser.add_argument("--expected-graph", default=None)
    parser.add_argument("--stderr-log", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    errors = []
    if args.expected_trials <= 0:
        parser.error("--expected-trials must be positive")
    required = {
        "Config",
        "Graph",
        "Trial",
        "Time_sec",
        "GTEPS",
        "RunnerExit",
        "Status",
    }
    try:
        with open(args.tsv, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            fields = set(reader.fieldnames or [])
            if not required.issubset(fields):
                errors.append(
                    "missing columns: " + ",".join(sorted(required - fields))
                )
            rows = list(reader)
    except OSError as error:
        print(f"ERROR: cannot read {args.tsv}: {error}", file=sys.stderr)
        return 2

    expected_total = len(EXPECTED_CONFIGS) * args.expected_trials
    if len(rows) != expected_total:
        errors.append(f"row_count expected={expected_total} actual={len(rows)}")

    configs = {row.get("Config", "") for row in rows}
    if configs != EXPECTED_CONFIGS:
        errors.append(
            "config_set mismatch missing="
            + repr(sorted(EXPECTED_CONFIGS - configs))
            + " unexpected="
            + repr(sorted(configs - EXPECTED_CONFIGS))
        )

    expected_trials = set(range(1, args.expected_trials + 1))
    seen_pairs = set()
    trials_by_config = {config: set() for config in EXPECTED_CONFIGS}
    duplicate_trials = 0
    for line_number, row in enumerate(rows, 2):
        config = row.get("Config", "")
        graph = row.get("Graph", "")
        if args.expected_graph is not None and graph != args.expected_graph:
            errors.append(
                f"line {line_number}: graph expected={args.expected_graph!r} actual={graph!r}"
            )
        try:
            trial = int(row.get("Trial", ""))
        except ValueError:
            errors.append(f"line {line_number}: malformed Trial={row.get('Trial')!r}")
            continue
        pair = (config, trial)
        if pair in seen_pairs:
            duplicate_trials += 1
        seen_pairs.add(pair)
        if config in trials_by_config:
            trials_by_config[config].add(trial)

        for column in ("Time_sec", "GTEPS"):
            try:
                value = float(row.get(column, ""))
            except ValueError:
                errors.append(
                    f"line {line_number}: malformed {column}={row.get(column)!r}"
                )
                continue
            if not math.isfinite(value) or value <= 0:
                errors.append(
                    f"line {line_number}: {column} must be finite and >0, actual={row.get(column)!r}"
                )
        try:
            runner_exit = int(row.get("RunnerExit", ""))
        except ValueError:
            errors.append(
                f"line {line_number}: malformed RunnerExit={row.get('RunnerExit')!r}"
            )
        else:
            if runner_exit != 0:
                errors.append(f"line {line_number}: RunnerExit expected=0 actual={runner_exit}")
        if row.get("Status") != "SUCCESS":
            errors.append(
                f"line {line_number}: Status expected='SUCCESS' actual={row.get('Status')!r}"
            )
        marker_text = "\t".join(str(value) for value in row.values())
        if FAILURE_RE.search(marker_text):
            errors.append(f"line {line_number}: failure marker in TSV row")

    if duplicate_trials:
        errors.append(f"duplicate_trial_count={duplicate_trials}")
    for config in sorted(EXPECTED_CONFIGS):
        actual = trials_by_config[config]
        if actual != expected_trials:
            errors.append(
                f"{config}: trial_set expected={sorted(expected_trials)} actual={sorted(actual)}"
            )

    if args.stderr_log:
        try:
            with open(args.stderr_log, encoding="utf-8", errors="replace") as stream:
                stderr_text = stream.read()
        except OSError as error:
            print(f"ERROR: cannot read {args.stderr_log}: {error}", file=sys.stderr)
            return 2
        match = FAILURE_RE.search(stderr_text)
        if match:
            errors.append(f"failure marker in stderr log: {match.group(0)!r}")

    payload = {
        "status": "PASS" if not errors else "FAIL",
        "row_count": len(rows),
        "expected_row_count": expected_total,
        "configuration_count": len(configs),
        "expected_configurations": sorted(EXPECTED_CONFIGS),
        "duplicate_trial_count": duplicate_trials,
        "errors": errors,
    }
    if args.json:
        try:
            with open(args.json, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
        except OSError as error:
            print(f"ERROR: cannot write {args.json}: {error}", file=sys.stderr)
            return 2
    if not args.quiet:
        print(f"status={payload['status']}")
        print(f"row_count={len(rows)} expected={expected_total}")
        print(f"configuration_count={len(configs)} expected=8")
        print(f"duplicate_trial_count={duplicate_trials}")
        for error in errors:
            print(f"detail={error}")
    return 0 if not errors else 3


if __name__ == "__main__":
    sys.exit(main())
