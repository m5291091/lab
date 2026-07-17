#!/usr/bin/env python3
"""BC vector dump integrity validator.

The parser never repairs, rounds, deduplicates, or replaces a recorded value.
Comments beginning with ``#`` and blank lines are metadata; every other line
must contain exactly ``index value``.  Exit 0 means complete and valid, exit 3
means an integrity violation, and exit 2 means an I/O/usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class VectorScan:
    path: str
    expected_length: Optional[int]
    sha256: str = "not_recorded"
    header: str = ""
    values: Dict[int, float] = field(default_factory=dict)
    seen_indices: set[int] = field(default_factory=set)
    data_rows: int = 0
    parse_errors: int = 0
    value_column_missing: int = 0
    duplicate_indices: int = 0
    out_of_range_indices: int = 0
    nan_values: int = 0
    positive_inf_values: int = 0
    negative_inf_values: int = 0
    missing_indices: int = 0
    examples: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        expected_ok = (
            self.expected_length is None
            or (
                len(self.seen_indices) == self.expected_length
                and self.missing_indices == 0
                and self.out_of_range_indices == 0
            )
        )
        return (
            self.data_rows > 0
            and self.parse_errors == 0
            and self.value_column_missing == 0
            and self.duplicate_indices == 0
            and self.nan_values == 0
            and self.positive_inf_values == 0
            and self.negative_inf_values == 0
            and expected_ok
        )

    def summary(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "expected_length": (
                self.expected_length if self.expected_length is not None else "not_recorded"
            ),
            "data_rows": self.data_rows,
            "unique_index_count": len(self.seen_indices),
            "missing_indices": self.missing_indices,
            "duplicate_indices": self.duplicate_indices,
            "out_of_range_indices": self.out_of_range_indices,
            "nan_values": self.nan_values,
            "positive_inf_values": self.positive_inf_values,
            "negative_inf_values": self.negative_inf_values,
            "parse_errors": self.parse_errors,
            "value_column_missing": self.value_column_missing,
            "status": "PASS" if self.valid else "FAIL",
            "examples": self.examples,
        }


def _remember(scan: VectorScan, message: str) -> None:
    if len(scan.examples) < 10:
        scan.examples.append(message)


def scan_vector(path: str, expected_length: Optional[int] = None) -> VectorScan:
    if expected_length is not None and expected_length <= 0:
        raise ValueError("expected_length must be positive")

    scan = VectorScan(path=path, expected_length=expected_length)
    scan.sha256 = sha256_of(path)

    with open(path, "r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                scan.header = line.lstrip("# ").strip()
                continue

            scan.data_rows += 1
            parts = line.split()
            if len(parts) < 2:
                scan.value_column_missing += 1
                scan.parse_errors += 1
                _remember(scan, f"line {line_number}: missing value column")
                continue
            if len(parts) != 2:
                scan.parse_errors += 1
                _remember(scan, f"line {line_number}: expected 2 columns, actual {len(parts)}")
                continue

            try:
                index = int(parts[0], 10)
            except ValueError:
                scan.parse_errors += 1
                _remember(scan, f"line {line_number}: malformed index {parts[0]!r}")
                continue
            try:
                value = float(parts[1])
            except ValueError:
                scan.parse_errors += 1
                _remember(scan, f"line {line_number}: malformed value {parts[1]!r}")
                continue

            if index in scan.seen_indices:
                # Preserve the first recorded value; never hide a duplicate by overwrite.
                scan.duplicate_indices += 1
                _remember(scan, f"line {line_number}: duplicate index {index}")
            else:
                scan.seen_indices.add(index)
                scan.values[index] = value

            if index < 0 or (expected_length is not None and index >= expected_length):
                scan.out_of_range_indices += 1
                upper = expected_length - 1 if expected_length is not None else "unbounded"
                _remember(
                    scan,
                    f"line {line_number}: index out of range, expected [0,{upper}], actual {index}",
                )

            if math.isnan(value):
                scan.nan_values += 1
                _remember(scan, f"line {line_number}: NaN at index {index}")
            elif value == math.inf:
                scan.positive_inf_values += 1
                _remember(scan, f"line {line_number}: +Inf at index {index}")
            elif value == -math.inf:
                scan.negative_inf_values += 1
                _remember(scan, f"line {line_number}: -Inf at index {index}")

    if expected_length is not None:
        in_range = sum(1 for index in scan.seen_indices if 0 <= index < expected_length)
        scan.missing_indices = expected_length - in_range
    if scan.data_rows == 0:
        _remember(scan, "no data rows (empty or header-only vector)")
    return scan


def write_json(path: str, payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path == "-":
        sys.stdout.write(text)
    else:
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BC vector completeness and values")
    parser.add_argument("vector")
    parser.add_argument("--expected-length", type=int, default=None)
    parser.add_argument("--json", default=None, help="write machine-readable summary (use - for stdout)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        scan = scan_vector(args.vector, args.expected_length)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"ERROR: {args.vector}: {error}", file=sys.stderr)
        return 2

    summary = scan.summary()
    if args.json:
        try:
            write_json(args.json, summary)
        except OSError as error:
            print(f"ERROR: cannot write {args.json}: {error}", file=sys.stderr)
            return 2
    if not args.quiet:
        for key, value in summary.items():
            if key != "examples":
                print(f"{key}={value}")
        for example in scan.examples:
            print(f"detail={example}")
    return 0 if scan.valid else 3


if __name__ == "__main__":
    sys.exit(main())
