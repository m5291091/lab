#!/usr/bin/env python3
"""Parse one validation log with an implementation-specific contract.

No parser expression or allocation formula is shared across implementations.
Recorded decimal-GB log values are converted to bytes with 1 GB = 10^9 bytes.
Code-derived allocations are labelled ``estimated`` and are never presented as
runtime memory measurements.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import re
import sys


COLUMNS = [
    "Implementation",
    "RequestedBatch",
    "EffectiveBatch",
    "RequestedNS",
    "EffectiveNS",
    "SubBatch",
    "NumSubs",
    "HBMCapacityBytes",
    "FreeHBMBeforeBytes",
    "PerSourceStateBytes",
    "CodeDerivedAllocationBytes",
    "AllocationFormula",
    "MemoryMode",
    "PrefetchMode",
    "ExitCode",
    "Status",
    "FailureReason",
    "ValueSource",
]


def gb_to_bytes(value: str) -> str:
    try:
        return str(int(Decimal(value) * Decimal(1_000_000_000)))
    except (InvalidOperation, ValueError):
        return "not_recorded"


def last_match(pattern: re.Pattern[str], text: str):
    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def proposed_per_source(n: int, max_depth: int) -> int:
    # d_d, d_Q_curr, d_Q_next, d_S (int); d_sigma, d_delta (double);
    # d_S_ends[max_depth+1] (int); d_depth (int).
    return n * (4 * 4 + 2 * 8) + (max_depth + 1) * 4 + 4


def parse_gpu_opt(args, text):
    hbm = last_match(
        re.compile(r"^\s*> \[Mem\] GPU HBM3: total=([0-9]+(?:\.[0-9]+)?) GB, free_before=([0-9]+(?:\.[0-9]+)?) GB\s*$", re.M),
        text,
    )
    memory = last_match(
        re.compile(
            r"^\s*> \[Mem\] topology\(CPU/HBM3\)=[0-9]+(?:\.[0-9]+)? GB, "
            r"dynamic\(UM\)=[0-9]+(?:\.[0-9]+)? GB, BATCH=(\d+), "
            r"SUB_BATCH=(\d+), num_subs=(\d+), NS_eff=(\d+)\s*$",
            re.M,
        ),
        text,
    )
    effective_batch = memory.group(1) if memory else "not_recorded"
    sub_batch = memory.group(2) if memory else "not_recorded"
    num_subs = memory.group(3) if memory else "not_recorded"
    effective_ns = memory.group(4) if memory else "not_recorded"
    per_source = proposed_per_source(args.nodes, args.max_depth)
    allocation = "not_recorded"
    formula = "not_recorded"
    if memory:
        allocation_value = int(effective_ns) * int(effective_batch) * per_source
        allocation = str(allocation_value)
        formula = (
            f"estimated: EffectiveNS({effective_ns}) * EffectiveBatch({effective_batch} sources) * "
            f"PerSourceStateBytes({per_source} bytes/source)"
        )
    streaming = re.search(r"^\s*> \[Mode\] HBM3 streaming:", text, re.M) is not None
    return {
        "EffectiveBatch": effective_batch,
        "RequestedNS": "2",
        "EffectiveNS": effective_ns,
        "SubBatch": sub_batch,
        "NumSubs": num_subs,
        "HBMCapacityBytes": gb_to_bytes(hbm.group(1)) if hbm else "not_recorded",
        "FreeHBMBeforeBytes": gb_to_bytes(hbm.group(2)) if hbm else "not_recorded",
        "PerSourceStateBytes": str(per_source),
        "CodeDerivedAllocationBytes": allocation,
        "AllocationFormula": formula,
        "MemoryMode": "managed_unified_memory",
        "PrefetchMode": "hbm3_streaming_prefetch" if streaming else "not_recorded",
        "ValueSource": (
            "RequestedNS=code_constant:src/proposed/host_um.cu:264; "
            "EffectiveBatch/SubBatch/NumSubs/EffectiveNS=recorded:implementation-specific dynamic(UM) log; "
            "HBMCapacityBytes/FreeHBMBeforeBytes=recorded:GPU_HBM3_decimal_GB*1000000000; "
            "PerSourceStateBytes/CodeDerivedAllocationBytes=estimated:src/proposed/host_um.cu:270-275,345-353"
        ),
    }


def parse_gpu_opt_pure(args, text):
    hbm = last_match(
        re.compile(r"^\s*> \[Mem\] GPU: total=([0-9]+(?:\.[0-9]+)?) GB, free_before=([0-9]+(?:\.[0-9]+)?) GB\s*$", re.M),
        text,
    )
    memory = last_match(
        re.compile(
            r"^\s*> \[Mem\] topology\(GPU\)=[0-9]+(?:\.[0-9]+)? GB, "
            r"dynamic\(GPU\)=[0-9]+(?:\.[0-9]+)? GB, batch_per_stream=(\d+)\s*$",
            re.M,
        ),
        text,
    )
    effective_batch = memory.group(1) if memory else "not_recorded"
    per_source = proposed_per_source(args.nodes, args.max_depth)
    allocation = "not_recorded"
    formula = "not_recorded"
    if memory:
        allocation = str(2 * int(effective_batch) * per_source)
        formula = (
            f"estimated: EffectiveNS(code_constant=2) * EffectiveBatch({effective_batch} sources) * "
            f"PerSourceStateBytes({per_source} bytes/source)"
        )
    return {
        "EffectiveBatch": effective_batch,
        "RequestedNS": "2",
        "EffectiveNS": "2",
        "SubBatch": "not_applicable",
        "NumSubs": "not_applicable",
        "HBMCapacityBytes": gb_to_bytes(hbm.group(1)) if hbm else "not_recorded",
        "FreeHBMBeforeBytes": gb_to_bytes(hbm.group(2)) if hbm else "not_recorded",
        "PerSourceStateBytes": str(per_source),
        "CodeDerivedAllocationBytes": allocation,
        "AllocationFormula": formula,
        "MemoryMode": "explicit_device_memory",
        "PrefetchMode": "not_applicable",
        "ValueSource": (
            "RequestedNS/EffectiveNS=code_constant:src/proposed/host_pure.cu:92; "
            "EffectiveBatch=recorded:implementation-specific dynamic(GPU) log; "
            "HBMCapacityBytes/FreeHBMBeforeBytes=recorded:GPU_decimal_GB*1000000000; "
            "PerSourceStateBytes/CodeDerivedAllocationBytes=estimated:src/proposed/host_pure.cu:141-157"
        ),
    }


def parse_gpu_opt_pure_chunked(args, text):
    hbm = last_match(
        re.compile(r"^\s*> \[Mem\] GPU: total=([0-9]+(?:\.[0-9]+)?) GB, free_before=([0-9]+(?:\.[0-9]+)?) GB\s*$", re.M),
        text,
    )
    memory = last_match(
        re.compile(
            r"^\s*> \[Mem\] topology\(GPU\)=[0-9]+(?:\.[0-9]+)? GB, "
            r"dynamic\(SUB_BATCH alloc\)=[0-9]+(?:\.[0-9]+)? GB, BATCH=(\d+), "
            r"SUB_BATCH=(\d+), num_subs=(\d+), NS_eff=(\d+)\s*$",
            re.M,
        ),
        text,
    )
    effective_batch = memory.group(1) if memory else "not_recorded"
    sub_batch = memory.group(2) if memory else "not_recorded"
    num_subs = memory.group(3) if memory else "not_recorded"
    effective_ns = memory.group(4) if memory else "not_recorded"
    per_source = proposed_per_source(args.nodes, args.max_depth)
    allocation = "not_recorded"
    formula = "not_recorded"
    if memory:
        allocation = str(int(effective_ns) * int(sub_batch) * per_source)
        formula = (
            f"estimated: EffectiveNS({effective_ns}) * SubBatch({sub_batch} sources) * "
            f"PerSourceStateBytes({per_source} bytes/source)"
        )
    chunking = re.search(r"^\s*> \[Mode\] Manual chunking:", text, re.M) is not None
    return {
        "EffectiveBatch": effective_batch,
        "RequestedNS": "2",
        "EffectiveNS": effective_ns,
        "SubBatch": sub_batch,
        "NumSubs": num_subs,
        "HBMCapacityBytes": gb_to_bytes(hbm.group(1)) if hbm else "not_recorded",
        "FreeHBMBeforeBytes": gb_to_bytes(hbm.group(2)) if hbm else "not_recorded",
        "PerSourceStateBytes": str(per_source),
        "CodeDerivedAllocationBytes": allocation,
        "AllocationFormula": formula,
        "MemoryMode": "explicit_device_memory_chunked" if chunking else "explicit_device_memory",
        "PrefetchMode": "not_applicable",
        "ValueSource": (
            "RequestedNS=code_constant:src/proposed/host_chunked.cu:93; "
            "EffectiveBatch/SubBatch/NumSubs/EffectiveNS=recorded:implementation-specific dynamic(SUB_BATCH alloc) log; "
            "HBMCapacityBytes/FreeHBMBeforeBytes=recorded:GPU_decimal_GB*1000000000; "
            "PerSourceStateBytes/CodeDerivedAllocationBytes=estimated:src/proposed/host_chunked.cu:145-151,176-184"
        ),
    }


def parse_pathmerge(args, text):
    memory = last_match(
        re.compile(
            r"^\s*> \[PathMerge\] free_mem=([0-9]+(?:\.[0-9]+)?) GB, "
            r"batch_size=(\d+), num_sources=(\d+), num_batches=(\d+)\s*$",
            re.M,
        ),
        text,
    )
    effective_batch = memory.group(2) if memory else "not_recorded"
    # PathMerge's own layout: 44*n bytes/source plus one source ID, with
    # n doubles for BC and three scalar ints.  GPU_Opt formulas are not used.
    per_source = 44 * args.nodes + 4
    allocation = "not_recorded"
    formula = "not_recorded"
    if memory:
        allocation = str(8 * args.nodes + int(effective_batch) * per_source + 3 * 4)
        formula = (
            f"estimated PathMerge-only: 8*n({args.nodes}) + EffectiveBatch({effective_batch}) * "
            f"(44*n+4)({per_source} bytes/source) + 3*4 bytes"
        )
    return {
        "EffectiveBatch": effective_batch,
        "RequestedNS": "not_applicable",
        "EffectiveNS": "not_applicable",
        "SubBatch": "not_applicable",
        "NumSubs": "not_applicable",
        "HBMCapacityBytes": "not_recorded",
        "FreeHBMBeforeBytes": gb_to_bytes(memory.group(1)) if memory else "not_recorded",
        "PerSourceStateBytes": str(per_source),
        "CodeDerivedAllocationBytes": allocation,
        "AllocationFormula": formula,
        "MemoryMode": "managed_memory_pathmerge",
        "PrefetchMode": "not_applicable",
        "ValueSource": (
            "EffectiveBatch/FreeHBMBeforeBytes=recorded:implementation-specific PathMerge log; "
            "RequestedNS/EffectiveNS/SubBatch/NumSubs/PrefetchMode=not_applicable; "
            "PerSourceStateBytes/CodeDerivedAllocationBytes=estimated_PathMerge_formula:"
            "src/baseline/pathmerge.cu:67-93,120-125;src/baseline/galliot.cu:33-63"
        ),
    }


PARSERS = {
    "GPU_Opt": parse_gpu_opt,
    "GPU_Opt_Pure": parse_gpu_opt_pure,
    "GPU_Opt_Pure_Chunked": parse_gpu_opt_pure_chunked,
    "PathMerge": parse_pathmerge,
}


def clean(value: str) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description="Implementation-specific log parser")
    parser.add_argument("--implementation", required=True, choices=sorted(PARSERS))
    parser.add_argument("--requested-batch", required=True, type=int)
    parser.add_argument("--nodes", type=int, default=325557)
    parser.add_argument("--max-depth", type=int, default=256)
    parser.add_argument("--log", required=True)
    parser.add_argument("--exit-code", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--failure-reason", default="not_applicable")
    parser.add_argument("--no-header", action="store_true")
    args = parser.parse_args()

    try:
        with open(args.log, "r", encoding="utf-8", errors="replace") as stream:
            text = stream.read()
    except OSError as error:
        print(f"ERROR: cannot read {args.log}: {error}", file=sys.stderr)
        return 2
    if args.requested_batch <= 0 or args.nodes <= 0 or args.max_depth < 0:
        print("ERROR: requested batch/nodes must be positive and max-depth non-negative", file=sys.stderr)
        return 2

    parsed = PARSERS[args.implementation](args, text)
    row = {
        "Implementation": args.implementation,
        "RequestedBatch": str(args.requested_batch),
        **parsed,
        "ExitCode": args.exit_code,
        "Status": args.status,
        "FailureReason": args.failure_reason,
    }
    writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
    if not args.no_header:
        writer.writeheader()
    writer.writerow({key: clean(row.get(key, "not_recorded")) for key in COLUMNS})
    return 0


if __name__ == "__main__":
    sys.exit(main())
