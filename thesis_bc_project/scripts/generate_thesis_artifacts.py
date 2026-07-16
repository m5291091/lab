#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_thesis_artifacts.py -- Gate K0

Generate a reproducible, English-only set of figures and tables for the
master's thesis and presentation, using ONLY canonical Git-tracked data under
thesis_bc_project/{raw_data,result,docs}.

Design rules (see result/figures/thesis/README.md for the full policy):
  * Every displayed value is recomputed from a canonical input file.
  * median = numpy.median; speedup = median(PathMerge_tuned) / median(GPU_Opt).
  * OOM configurations are NEVER represented as zero seconds -- they are drawn
    as a distinct out-of-memory marker, not on the runtime axis.
  * The canonical memory-path stress result stays visible as "Core Fail".
  * No estimation, interpolation, reverse-calculation, or fastest-trial cherry
    picking. Missing measurements are not connected as if they existed.
  * All in-figure / in-table text is English. Graph names and implementation
    names are kept verbatim (not translated).
  * No dependency on build_miyabi/. Deterministic output (fixed SOURCE_DATE_EPOCH,
    no embedded timestamps, fixed SVG hash salt).

Run:  python3 scripts/generate_thesis_artifacts.py
"""

import os
# Determinism: fixed epoch so matplotlib does not embed the wall-clock time.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-thesis-bc-gate-k0")

import csv
import re
import subprocess
import sys
import statistics
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# Paths (resolved relative to this script; NO build_miyabi dependency).
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent          # thesis_bc_project/
RAW = ROOT / "raw_data"
RES = ROOT / "result"
DOCS = ROOT / "docs"
FIG_DIR = RES / "figures" / "thesis"
TAB_DIR = RES / "tables" / "thesis"

# --------------------------------------------------------------------------- #
# Global style: colorblind-safe (Okabe-Ito), consistent colors & order.
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "pdf.fonttype": 42,          # embed TrueType (Type 42) subsets in PDF
    "ps.fonttype": 42,
    "svg.fonttype": "none",      # keep SVG text as extractable text (English only)
    "svg.hashsalt": "thesis_bc_gate_k0",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": "--",
})

OK = {  # Okabe-Ito colorblind-safe palette
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#999999",
}

# Consistent implementation colors / markers / hatches across ALL figures.
IMPL_STYLE = {
    "GPU_Opt":               dict(color=OK["blue"],       marker="o", hatch="//"),
    "GPU_Opt_Pure":          dict(color=OK["orange"],     marker="s", hatch="\\\\"),
    "GPU_Opt_Pure_Chunked":  dict(color=OK["green"],      marker="^", hatch="xx"),
    "PathMerge":             dict(color=OK["vermillion"], marker="D", hatch=".."),
}
FACTOR_STYLE = {
    "Hybrid BFS":                     dict(color=OK["blue"],   hatch="//"),
    "Warp-Cooperative Accumulation":  dict(color=OK["green"],  hatch="xx"),
    "Dual Streams":                   dict(color=OK["purple"], hatch=".."),
}
PHASE_STYLE = {
    "BFS":      dict(color=OK["skyblue"],    hatch="//"),
    "Backward": dict(color=OK["vermillion"], hatch="xx"),
    "Other":    dict(color=OK["grey"],       hatch=".."),
}

# Consistent graph order for the headline (4-graph) figures.
HEADLINE_ORDER = ["email-EuAll", "roadNet-PA", "roadNet-TX", "roadNet-CA"]

# Provenance labels (SourceSnapshotID) used in manifests / captions.
SNAP_BLOCK = "phase_def_block_20260710"
SNAP_LEGACY = "oldtree_f05ec52_20260512"

# Accumulators for validation + manifest.
INPUTS_USED = set()
VALIDATION = []   # list of (name, ok, detail)


def note(name, ok, detail):
    VALIDATION.append((name, ok, detail))


# --------------------------------------------------------------------------- #
# Small IO helpers.
# --------------------------------------------------------------------------- #
def input_path(relpath):
    """Resolve and record a canonical input path relative to ROOT."""
    p = (ROOT / relpath) if not str(relpath).startswith("/") else Path(relpath)
    INPUTS_USED.add(str(p.relative_to(ROOT)))
    return p


def read_tsv(relpath):
    """Read a tab-separated file (relative to ROOT) into a list of dict rows."""
    p = input_path(relpath)
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_text(relpath):
    """Read and record a canonical text input relative to ROOT."""
    with open(input_path(relpath)) as f:
        return f.read()


def median(vals):
    return float(np.median(np.asarray(list(vals), dtype=float)))


def sample_sd(vals):
    vals = list(vals)
    return float(statistics.stdev(vals)) if len(vals) >= 2 else None


def save_fig(fig, stem):
    """Save PDF + PNG(300dpi) + SVG deterministically (no timestamps)."""
    base = FIG_DIR / stem
    fig.savefig(str(base) + ".pdf",
                metadata={"Creator": "generate_thesis_artifacts.py",
                          "Producer": "matplotlib"})
    fig.savefig(str(base) + ".png", dpi=300,
                metadata={"Software": "generate_thesis_artifacts.py"})
    fig.savefig(str(base) + ".svg",
                metadata={"Creator": "generate_thesis_artifacts.py", "Date": None})
    plt.close(fig)
    return {"pdf": stem + ".pdf", "png": stem + ".png", "svg": stem + ".svg"}


def bottom_caption(ax, text, y=-0.17, fontsize=7.5, italic=False):
    """Place a caption BELOW the x-axis label (axes-relative, so it never
    collides with the x-label regardless of tick-label height)."""
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=fontsize, style=("italic" if italic else "normal"))


def write_table(stem, header, rows, title, note_lines):
    """Write a presentation-ready Markdown table and a TSV with the same data."""
    tsv = TAB_DIR / (stem + ".tsv")
    with open(tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    md = TAB_DIR / (stem + ".md")
    with open(md, "w") as f:
        f.write("# " + title + "\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(c) for c in r) + " |\n")
        if note_lines:
            f.write("\n")
            for ln in note_lines:
                f.write("> " + ln + "\n")
    return {"md": stem + ".md", "tsv": stem + ".tsv"}


# --------------------------------------------------------------------------- #
# Canonical data loaders (recompute everything from raw).
# --------------------------------------------------------------------------- #
def load_graph_metadata():
    stats = {r["graph"]: r for r in read_tsv("docs/graph_stats.tsv")}
    cat = {r["graph"]: r for r in read_tsv("result/datasets/graph_catalog.tsv")}
    return stats, cat


def load_gpu_opt_results(graph):
    """GPU_Opt median runtime and median GTEPS from proposed_variants (b512)."""
    rel = (f"raw_data/main_performance/proposed_variants/{graph}/_run/"
           f"job_2357334_20260711/results.tsv")
    rows = [r for r in read_tsv(rel) if r["Implementation"] == "GPU_Opt"]
    t = [float(r["Time_sec"]) for r in rows]
    g = [float(r["GTEPS"]) for r in rows]
    return dict(time_med=median(t), time_sd=sample_sd(t), gteps_med=median(g),
                n=len(rows), times=t)


def load_pathmerge_tuned(graph):
    """Tuned PathMerge median runtime / GTEPS from the canonical source per graph.

    Provenance follows result/main_performance/proposed_vs_pathmerge/comparison.tsv:
      email-EuAll -> pathmerge sweep, b2048
      roadNet-PA  -> legacy_partial/large results_no_gpu_opt.tsv, b64
      roadNet-TX  -> legacy_partial/large results_no_gpu_opt.tsv, b64
      roadNet-CA  -> pathmerge sweep, b32
    """
    if graph == "email-EuAll":
        rel = ("raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/"
               "job_multi_20260710/pathmerge_sweep_results.tsv")
        rows = [r for r in read_tsv(rel) if r["Config"] == "PathMerge_b2048"]
        batch = "b2048"
        t = [float(r["Time_sec"]) for r in rows]
        g = [float(r["GTEPS"]) for r in rows]
    elif graph == "roadNet-CA":
        rel = ("raw_data/tuning/pathmerge/roadNet-CA/pathmerge_bc/"
               "job_multi_20260710/pathmerge_sweep_results.tsv")
        rows = [r for r in read_tsv(rel) if r["Config"] == "PathMerge_b32"]
        batch = "b32"
        t = [float(r["Time_sec"]) for r in rows]
        g = [float(r["GTEPS"]) for r in rows]
    else:  # roadNet-PA / roadNet-TX -> legacy large, b64
        rel = ("raw_data/main_performance/seven_implementations/legacy_partial/"
               "large/no_gpu_opt/job_notrecorded_legacy/results_no_gpu_opt.tsv")
        rows = [r for r in read_tsv(rel)
                if r["Implementation"] == "PathMerge_BC" and r["Graph"] == graph]
        batch = "b64"
        t = [float(r["Time_sec"]) for r in rows]
        g = [float(r["GTEPS"]) for r in rows]
    return dict(time_med=median(t), time_sd=sample_sd(t), gteps_med=median(g),
                n=len(rows), batch=batch)


def load_main_performance():
    data = {}
    for gname in HEADLINE_ORDER:
        prop = load_gpu_opt_results(gname)
        pm = load_pathmerge_tuned(gname)
        speedup = pm["time_med"] / prop["time_med"]
        data[gname] = dict(prop=prop, pm=pm, speedup=speedup)
    return data


def load_pathmerge_sweep(graph):
    """Return measured PathMerge sweep points and recorded clamp metadata."""
    # The sweep directory for 325557_3216152 is named "325557".
    sweep_dir = {"325557_3216152": "325557"}.get(graph, graph)
    rel = (f"raw_data/tuning/pathmerge/{sweep_dir}/pathmerge_bc/"
           f"job_multi_20260710/pathmerge_sweep_results.tsv")
    rows = read_tsv(rel)
    log_rel = (f"raw_data/tuning/pathmerge/{sweep_dir}/pathmerge_bc/"
               f"job_multi_20260710/pathmerge_sweep.log")
    effective = {}
    if (ROOT / log_rel).exists():
        current_requested = None
        for line in read_text(log_rel).splitlines():
            m = re.match(r"=== PathMerge batch_size=(\d+) ===", line)
            if m:
                current_requested = int(m.group(1))
                continue
            m = re.search(r"WARNING: batch_size=(\d+).*clamping to (\d+)", line)
            if m:
                requested, actual = int(m.group(1)), int(m.group(2))
                if requested != current_requested:
                    raise ValueError(f"inconsistent clamp record in {log_rel}: {line}")
                effective[requested] = actual
    by_batch = {}
    for r in rows:
        b = int(r["Config"].replace("PathMerge_b", ""))
        by_batch.setdefault(b, []).append(float(r["Time_sec"]))
    out = {}
    for b, ts in by_batch.items():
        out[b] = dict(times=ts, n=len(ts), med=median(ts), sd=sample_sd(ts),
                      effective=effective.get(b), clamped=(b in effective))
    return out


def load_ablation():
    """Recompute factorial main effects from canonical per-trial TSV files.

    For each factor, pair configurations that differ only in that factor,
    compute T(F=0)/T(F=1) from per-configuration medians, then take the
    geometric mean across the four pairs.  The archived contribution TSVs are
    used only as independent cross-checks.
    """
    sources = [
        ("synthetic",
         "raw_data/ablation/synthetic/job_2354994_20260710/ablation_results.tsv",
         "result/ablation/synthetic_2354994/ablation_contributions.tsv"),
        ("email",
         "raw_data/ablation/email-EuAll/job_2354999_20260710/ablation_results.tsv",
         "result/ablation/email_2354999/ablation_contributions.tsv"),
    ]
    factor_key = {"H": "Hybrid BFS", "W": "Warp-Cooperative Accumulation",
                  "A": "Dual Streams"}
    factor_pos = {"H": 0, "W": 1, "A": 2}
    per_graph = {}
    trial_counts = {}
    syn_graphs = []
    for group, raw_rel, archived_rel in sources:
        raw_rows = read_tsv(raw_rel)
        by_graph = {}
        for r in raw_rows:
            match = re.fullmatch(r"Ablation_H([01])_W([01])_A([01])", r["Config"])
            if not match:
                raise ValueError(f"unexpected ablation config: {r['Config']}")
            config = tuple(int(v) for v in match.groups())
            by_graph.setdefault(r["Graph"], {}).setdefault(config, []).append(
                float(r["Time_sec"]))
        if group == "synthetic":
            syn_graphs = list(by_graph)
        counts = {len(ts) for cfgs in by_graph.values() for ts in cfgs.values()}
        if len(counts) != 1:
            raise ValueError(f"inconsistent ablation trial counts in {raw_rel}: {counts}")
        trial_counts[group] = counts.pop()
        for graph, configs in by_graph.items():
            medians = {cfg: median(ts) for cfg, ts in configs.items()}
            if len(medians) != 8:
                raise ValueError(f"incomplete 2^3 ablation for {graph}")
            effects = {}
            for factor, label in factor_key.items():
                pos = factor_pos[factor]
                ratios = []
                for cfg0 in sorted(c for c in medians if c[pos] == 0):
                    cfg1 = list(cfg0)
                    cfg1[pos] = 1
                    ratios.append(medians[cfg0] / medians[tuple(cfg1)])
                effects[label] = float(np.exp(np.mean(np.log(ratios))))
            per_graph[graph] = effects

        archived = read_tsv(archived_rel)
        for row in archived:
            got = per_graph[row["Graph"]][factor_key[row["Factor"]]]
            expected = float(row["MainEffect"])
            note(f"ablation_main_effect[{row['Graph']},{row['Factor']}]",
                 round(got, 4) == round(expected, 4),
                 f"recomputed={got:.4f} archived={expected:.4f}")

    geo = {}
    for fk in factor_key.values():
        vals = [per_graph[g][fk] for g in syn_graphs]
        geo[fk] = float(np.exp(np.mean(np.log(vals))))
    email_eff = per_graph["email-EuAll"]
    return dict(per_graph=per_graph, syn_graphs=syn_graphs, syn_geo=geo,
                email=email_eff, factor_key=factor_key, trials=trial_counts)


def load_memory_scalability():
    impls = [("gpu_opt", "GPU_Opt"),
             ("gpu_opt_pure", "GPU_Opt_Pure"),
             ("gpu_opt_pure_chunked", "GPU_Opt_Pure_Chunked")]
    out = {}
    for key, label in impls:
        rel = (f"raw_data/memory_scalability/325557_3216152/{key}/"
               f"job_notrecorded_20260512/oversubscribe_results_{key}.tsv")
        rows = read_tsv(rel)
        log_rel = (f"raw_data/memory_scalability/325557_3216152/{key}/"
                   f"job_notrecorded_20260512/um_experiment_{key}.log")
        meta = {}
        current_batch = None
        for line in read_text(log_rel).splitlines():
            m = re.match(r"=== \S+ batch=(\d+) trial=\d+ rc=\d+ ===", line)
            if m:
                current_batch = int(m.group(1))
                continue
            m = re.search(r"BATCH=(\d+), SUB_BATCH=(\d+), num_subs=(\d+)", line)
            if m and current_batch is not None:
                values = tuple(int(v) for v in m.groups())
                if values[0] != current_batch:
                    raise ValueError(f"inconsistent memory metadata in {log_rel}: {line}")
                meta.setdefault(current_batch, set()).add(values)
        by_batch = {}
        for r in rows:
            b = int(r["BatchSize"])
            by_batch.setdefault(b, {"succ": [], "status": set(), "trials": 0})
            by_batch[b]["status"].add(r["Status"])
            by_batch[b]["trials"] += 1
            if r["Status"] == "SUCCESS":
                by_batch[b]["succ"].append(float(r["Time_sec"]))
        pts = {}
        for b, d in by_batch.items():
            success = "SUCCESS" in d["status"] and len(d["succ"]) > 0
            observed_meta = meta.get(b, set())
            if len(observed_meta) > 1:
                raise ValueError(f"inconsistent BATCH/SUB_BATCH metadata for {label} b{b}")
            recorded = next(iter(observed_meta)) if observed_meta else None
            pts[b] = dict(
                success=success,
                med=median(d["succ"]) if success else None,
                sd=sample_sd(d["succ"]) if success and len(d["succ"]) >= 2 else None,
                oom=("OOM_OR_FAIL" in d["status"]) and not success,
                n=len(d["succ"]),
                trials=d["trials"],
                effective=(recorded[0] if recorded else None),
                sub_batch=(recorded[1] if recorded else None),
                num_subs=(recorded[2] if recorded else None),
            )
        out[label] = pts
    return out


def load_kernel_selection():
    out = {}
    jobmap = {"roadNet-PA": "job_2354329_20260710", "roadNet-TX": "job_2354330_20260710"}
    for g, job in jobmap.items():
        rel = f"raw_data/tuning/kernel_selection/{g}/gpu_opt_forced/{job}/kernel_selection_results.tsv"
        rows = read_tsv(rel)
        sh = [float(r["Time_sec"]) for r in rows if r["Kernel"] == "shared"]
        bl = [float(r["Time_sec"]) for r in rows if r["Kernel"] == "block"]
        out[g] = dict(shared_med=median(sh), block_med=median(bl),
                      shared_sd=sample_sd(sh), block_sd=sample_sd(bl),
                      shared_n=len(sh), block_n=len(bl),
                      speedup=median(sh) / median(bl))
    return out


def load_phase_breakdown():
    """Parse phase_timing.log for GPU_Opt trials on the 4 headline graphs.

    Each GPU_Opt trial block yields (BFS wall, Backward wall, total Elapse).
    Other := total - BFS - Backward  (init / CSR load / copy-out / host overhead).
    """
    out = {}
    for g in HEADLINE_ORDER:
        rel = (f"raw_data/main_performance/proposed_variants/{g}/_run/"
               f"job_2357334_20260711/phase_timing.log")
        p = input_path(rel)
        samples = []
        cur = None
        pending = None
        with open(p) as f:
            for line in f:
                s = line.strip()
                if s.startswith("Running: GPU_Opt on"):
                    cur = "GPU_Opt"
                elif s.startswith("Running: GPU_Opt_Pure") or s.startswith("Running: GPU_Opt_Pure_Chunked"):
                    cur = "other"
                elif cur == "GPU_Opt" and s.startswith("> [GPU Phase]"):
                    # "> [GPU Phase] BFS wall=9.0829 s (cum=...), Backward wall=20.0790 s (...)"
                    bfs_v = float(s.split("BFS wall=")[1].split(" s")[0])
                    back_v = float(s.split("Backward wall=")[1].split(" s")[0])
                    pending = (bfs_v, back_v)
                elif cur == "GPU_Opt" and s.startswith("> Elapse time"):
                    if pending is None:
                        raise ValueError(f"missing phase record before total in {rel}")
                    total_v = float(s.split("=")[1].strip())
                    bfs_v, back_v = pending
                    samples.append((bfs_v, back_v, total_v,
                                    total_v - bfs_v - back_v))
                    cur = None
                    pending = None
        if not samples:
            raise ValueError(f"no GPU_Opt phase samples found in {rel}")
        bfs_m = median(s[0] for s in samples)
        back_m = median(s[1] for s in samples)
        total_m = median(s[2] for s in samples)
        other_m = median(s[3] for s in samples)
        out[g] = dict(bfs=bfs_m, backward=back_m, other=other_m,
                      total=total_m, component_sum=bfs_m + back_m + other_m,
                      n=len(samples))
    return out


def comparison_nonfinite(relpath):
    """Extract the two archived non-finite counts from a comparison report."""
    counts = [int(v) for v in re.findall(r"\| 非有限値数 [AB] \| (\d+) \|",
                                         read_text(relpath))]
    if len(counts) != 2:
        raise ValueError(f"could not parse non-finite counts from {relpath}")
    return sum(counts)


def load_correctness():
    small = read_tsv("result/correctness/small_full_vector/correctness_summary.tsv")
    mem = read_tsv("result/correctness/memory_paths/canonical_job_2368587/comparison_matrix.tsv")
    small_nf = {}
    for graph in ("benchmark_7000_41459", "benchmark_11023_62184", "chain_200"):
        small_nf[graph] = comparison_nonfinite(
            f"result/correctness/small_full_vector/{graph}/comparison.md")
    mem_pairs = [
        ("gpu_opt_b1024", "gpu_opt_pure_b1024"),
        ("gpu_opt_b1024", "gpu_opt_pure_chunked_b1024"),
        ("gpu_opt_pure_b1024", "gpu_opt_pure_chunked_b1024"),
        ("gpu_opt_b9792", "gpu_opt_b1024"),
        ("gpu_opt_pure_chunked_b16384", "gpu_opt_pure_chunked_b1024"),
        ("pathmerge_b4096", "gpu_opt_b1024"),
    ]
    mem_nf = {}
    for a, b in mem_pairs:
        rel = ("result/correctness/memory_paths/canonical_job_2368587/"
               f"{a}__vs__{b}.md")
        mem_nf[(a, b)] = comparison_nonfinite(rel)
    return small, mem, small_nf, mem_nf


def load_environment():
    """Parse bandwidth and verify normalized specs against the environment archive."""
    environment = read_text("result/environment/environment.md")
    required_tokens = {
        "gpu": "| GPU | NVIDIA GH200 |",
        "nominal_hbm3": "| 公称 HBM3 | 96 GB |",
        "recorded_memory": "97,871 MiB（約95.6 GiB、約102.6 decimal GB",
        "runtime_memory": "total 約102.0 GB、free (`free_before`) 約101.4 GB",
        "driver": "595.58.03",
        "cuda": "release 13.0, V13.0.48",
        "cmake": "4.3.4",
        "compiler": "g++ (GCC) 11.4.1",
        "nsys": "2025.5.1.121",
        "pbs_system": "Miyabi-G PBS batch system",
        "group": "| Group | `gj17` |",
        "queue": "Not independently verifiable from retained job logs",
        "memory_resource": "Host-memory-limited 100 GiB configuration",
    }
    for key, token in required_tokens.items():
        note(f"environment[{key}]", token in environment,
             f"archived specification contains {token!r}")

    runtime_memory_log = read_text(
        "raw_data/main_performance/proposed_variants/email-EuAll/_run/"
        "job_2357334_20260711/phase_timing.log")
    runtime_memory_token = "GPU HBM3: total=102.0 GB, free_before=101.4 GB"
    note("runtime_memory_at_launch", runtime_memory_token in runtime_memory_log,
         f"retained run log contains {runtime_memory_token!r}")

    p = input_path("raw_data/profiling/job_2359175_20260711/bandwidth.log")
    bw = {}
    gpu_name = None
    with open(p) as f:
        for line in f:
            s = line.strip()
            if s.startswith("GPU:"):
                gpu_name = s.split("GPU:")[1].strip()
            parts = s.split("\t")
            if len(parts) == 6 and parts[0] in (
                    "HBM3_DtoD", "Pinned_HtoD", "Pinned_DtoH", "NVLink_C2C_Prefetch"):
                bw[parts[0]] = dict(gbs=parts[3], ratio=parts[5])
    return dict(gpu_name=gpu_name, bw=bw)


# --------------------------------------------------------------------------- #
# Figures.
# --------------------------------------------------------------------------- #
def fig_F1(mp):
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x = np.arange(len(HEADLINE_ORDER)); w = 0.38
    prop = [mp[g]["prop"]["time_med"] for g in HEADLINE_ORDER]
    pm = [mp[g]["pm"]["time_med"] for g in HEADLINE_ORDER]
    prop_sd = [mp[g]["prop"]["time_sd"] for g in HEADLINE_ORDER]
    pm_sd = [mp[g]["pm"]["time_sd"] for g in HEADLINE_ORDER]
    s1 = IMPL_STYLE["GPU_Opt"]; s2 = IMPL_STYLE["PathMerge"]
    b1 = ax.bar(x - w / 2, prop, w, yerr=prop_sd, capsize=3, label="GPU_Opt (b512)",
                color=s1["color"], hatch=s1["hatch"], edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w / 2, pm, w, yerr=pm_sd, capsize=3, label="PathMerge (Tuned)",
                color=s2["color"], hatch=s2["hatch"], edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Median Runtime (s)  [log scale]")
    ax.set_xlabel("Graph")
    ax.set_title("F1  Main Runtime Comparison: GPU_Opt (b512) vs Tuned PathMerge")
    ax.set_xticks(x); ax.set_xticklabels(HEADLINE_ORDER)
    ax.set_ylim(10, 6000)
    for i, g in enumerate(HEADLINE_ORDER):
        ax.text(x[i] - w / 2, prop[i] * 1.05, f"{prop[i]:.1f}s",
                ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + w / 2, pm[i] * 1.05,
                f"{pm[i]:.1f}s\n({mp[g]['pm']['batch']})",
                ha="center", va="bottom", fontsize=8)
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    bottom_caption(ax,
                   "Median of per-trial runtimes; error bars = sample SD. Batch annotations "
                   "give the tuned PathMerge batch. Log y-axis (values span ~2 orders).",
                   y=-0.15)
    return save_fig(fig, "main_runtime_comparison")


def fig_F2(mp):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(HEADLINE_ORDER))
    sp = [mp[g]["speedup"] for g in HEADLINE_ORDER]
    s1 = IMPL_STYLE["GPU_Opt"]
    bars = ax.bar(x, sp, 0.55, color=s1["color"], hatch=s1["hatch"],
                  edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color=OK["vermillion"], linestyle="--", linewidth=1.4,
               label="Parity (1.0x, Tuned PathMerge)")
    ax.set_ylabel("Speedup over Tuned PathMerge")
    ax.set_xlabel("Graph")
    ax.set_title("F2  GPU_Opt (b512) Speedup over Tuned PathMerge")
    ax.set_xticks(x); ax.set_xticklabels(HEADLINE_ORDER)
    ax.set_ylim(0, max(sp) * 1.25)
    for i in range(len(x)):
        ax.text(x[i], sp[i] + 0.03, f"{sp[i]:.2f}x", ha="center", va="bottom", fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    bottom_caption(ax, "Results are limited to the four evaluated graphs.",
                   y=-0.13, fontsize=8.5, italic=True)
    return save_fig(fig, "main_speedup_over_tuned_pathmerge")


def fig_F3(sweeps, tuned_batch):
    graphs = [g for g in ["email-EuAll", "roadNet-PA", "roadNet-TX", "roadNet-CA",
                          "325557_3216152"] if g in sweeps]
    n = len(graphs)
    ncol = 3; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for idx, g in enumerate(graphs):
        ax = axes[idx]
        pts = sweeps[g]
        batches = sorted(pts.keys())
        xs = list(batches)
        # Draw a line only between adjacent log2 sweep positions.  A missing
        # power-of-two batch starts a new segment instead of implying a sample.
        segments = []
        for b in batches:
            if not segments or b != 2 * segments[-1][-1]:
                segments.append([b])
            else:
                segments[-1].append(b)
        for segment in segments:
            if len(segment) >= 2:
                ax.plot(segment, [pts[b]["med"] for b in segment], "-",
                        color=OK["vermillion"], linewidth=1.3, zorder=1)
        for b in batches:
            p = pts[b]
            if p["n"] == 1:
                marker, size, fill = "o", 6, "white"
            elif p["n"] == 2:
                marker, size, fill = "^", 7, "white"
            else:
                marker, size, fill = "D", 7, OK["vermillion"]
            ax.errorbar(
                b, p["med"], yerr=p["sd"], fmt=marker, markersize=size,
                color=OK["vermillion"],
                markerfacecolor=fill,
                markeredgecolor=OK["vermillion"], capsize=3, zorder=3)
            if p["clamped"]:
                ax.scatter([b], [p["med"]], s=185, marker="s", facecolors="none",
                           edgecolors=OK["purple"], linewidths=1.8, zorder=4)
                ax.annotate(f"Requested {b}\nEffective {p['effective']}\n(clamped)",
                            xy=(b, p["med"]), xytext=(-8, 32),
                            textcoords="offset points", ha="right", va="bottom",
                            fontsize=7.5, color=OK["purple"],
                            arrowprops=dict(arrowstyle="-", color=OK["purple"],
                                            linewidth=0.8))
        # highlight tuned batch used in the main comparison, if present
        tb = tuned_batch.get(g)
        if tb in pts:
            ax.scatter([tb], [pts[tb]["med"]], s=180, facecolors="none",
                       edgecolors=OK["blue"], linewidths=1.8, zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_title(g, fontsize=11)
        ax.set_xlabel("Requested Batch Size (log2)")
        ax.set_ylabel("Median Runtime (s)")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(b) for b in xs], rotation=45, fontsize=8)
        ax.get_xaxis().set_minor_locator(plt.NullLocator())
    for j in range(n, len(axes)):
        axes[j].axis("off")
    legend_ax = axes[-1] if n < len(axes) else None
    handles = [
        Line2D([0], [0], marker="D", color=OK["vermillion"], linestyle="-",
               markerfacecolor=OK["vermillion"],
               label="Confirmation (n>=3; error bar = sample SD)"),
        Line2D([0], [0], marker="^", color=OK["vermillion"], linestyle="None",
               markerfacecolor="white", label="Two recorded trials (n=2; sample SD)"),
        Line2D([0], [0], marker="o", color=OK["vermillion"], linestyle="None",
               markerfacecolor="white", label="Screening only (n=1; no error bar)"),
        Line2D([0], [0], marker="o", color=OK["blue"], linestyle="None",
               markerfacecolor="none", markersize=11, markeredgewidth=1.8,
               label="Tuned batch (used in F1/F2/T2)"),
        Line2D([0], [0], marker="s", color=OK["purple"], linestyle="None",
               markerfacecolor="none", markersize=11, markeredgewidth=1.8,
               label="Clamped (requested and effective annotated)"),
    ]
    tgt = legend_ax if legend_ax is not None else axes[0]
    if legend_ax is not None:
        legend_ax.legend(handles=handles, loc="center", fontsize=10, frameon=True)
        legend_ax.set_title("Legend")
    else:
        axes[0].legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("F3  PathMerge Batch-Size Sweep (Median Runtime)", fontsize=12)
    fig.text(0.5, 0.012,
             "The x-axis shows requested batch size. Effective batch is annotated where "
             "canonical logs record clamping. Lines stop at missing batch sizes.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    return save_fig(fig, "pathmerge_batch_sweep")


def fig_F4(abl):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.8))
    # Panel A: main effects (synthetic geomean vs email) for the three factors
    factors = ["Hybrid BFS", "Warp-Cooperative Accumulation", "Dual Streams"]
    groups = [("Synthetic (geomean, 4 graphs)", abl["syn_geo"]),
              ("email-EuAll (hub)", abl["email"])]
    x = np.arange(len(factors)); w = 0.38
    for gi, (glabel, eff) in enumerate(groups):
        vals = [eff[f] for f in factors]
        color = OK["blue"] if gi == 0 else OK["orange"]
        hatch = "//" if gi == 0 else "\\\\"
        axA.bar(x + (gi - 0.5) * w, vals, w, label=glabel, color=color,
                hatch=hatch, edgecolor="black", linewidth=0.6)
        for xi, v in zip(x + (gi - 0.5) * w, vals):
            axA.text(xi, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axA.axhline(1.0, color=OK["vermillion"], linestyle="--", linewidth=1.3,
                label="No effect (1.0x)")
    axA.set_xticks(x)
    axA.set_xticklabels(["Hybrid BFS", "Warp-Cooperative\nAccumulation", "Dual Streams"])
    axA.set_ylabel("Main Effect (geomean speedup)")
    axA.set_title("(a) Ablation main effects")
    axA.set_ylim(0, 2.35)
    # place legend above the short center (Warp) group so it never overlaps the
    # tall Hybrid / Dual-Stream bars or their value labels
    axA.legend(loc="upper center", fontsize=8, ncol=1, framealpha=0.95)
    axA.grid(axis="x", visible=False)
    # Panel B: per-graph Warp-Cooperative effect (shows graph dependence)
    order = ["benchmark_7000_41459", "benchmark_11023_62184", "56438_300801",
             "325557_3216152", "email-EuAll"]
    order = [g for g in order if g in abl["per_graph"]]
    wvals = [abl["per_graph"][g]["Warp-Cooperative Accumulation"] for g in order]
    colors = [OK["green"] if v >= 1.0 else OK["vermillion"] for v in wvals]
    xb = np.arange(len(order))
    axB.bar(xb, wvals, 0.6, color=colors, hatch="xx", edgecolor="black", linewidth=0.6)
    axB.axhline(1.0, color=OK["black"], linestyle="--", linewidth=1.3)
    for xi, v in zip(xb, wvals):
        axB.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axB.set_xticks(xb); axB.set_xticklabels(order, rotation=30, ha="right", fontsize=8.5)
    axB.set_ylabel("Warp-Cooperative Main Effect")
    axB.set_title("(b) Warp-Cooperative Accumulation is graph-dependent")
    axB.set_ylim(0.9, 1.25)
    axB.legend(handles=[Patch(facecolor=OK["green"], hatch="xx", edgecolor="black",
                              label="Beneficial (>= 1.0x)"),
                        Patch(facecolor=OK["vermillion"], hatch="xx", edgecolor="black",
                              label="Harmful (< 1.0x)")],
               loc="upper right", fontsize=8)
    axB.grid(axis="x", visible=False)
    fig.suptitle("F4  Ablation Contributions (5 measured graphs; not generalized to roadNet)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_fig(fig, "ablation_contributions")


def fig_F5(mem):
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    all_succ = [p["med"] for pts in mem.values() for p in pts.values() if p["med"]]
    ymax = max(all_succ)
    oom_y = ymax * 1.25   # OOM band position -- NOT zero seconds
    ax.axhspan(ymax * 1.12, oom_y * 1.10, color=OK["vermillion"], alpha=0.08, zorder=0)
    ax.axhline(ymax * 1.12, color=OK["vermillion"], linestyle=":", linewidth=1.0)
    ax.text(mem_min_batch(mem), oom_y, "Out of Memory band (run did not complete; "
            "not a runtime value)", fontsize=8, color=OK["vermillion"], va="center")
    for i, (label, pts) in enumerate(mem.items()):
        st = IMPL_STYLE[label]
        batches = sorted(pts.keys())
        sb = [b for b in batches if pts[b]["success"]]
        sy = [pts[b]["med"] for b in sb]
        ssd = [pts[b]["sd"] for b in sb]
        ax.errorbar(sb, sy, yerr=ssd, fmt=st["marker"] + "-", color=st["color"],
                    markerfacecolor=st["color"], markeredgecolor="black",
                    markersize=7, linewidth=1.4, capsize=3, label=label, zorder=3)
        ob = [b for b in batches if pts[b]["oom"]]
        if ob:
            # multiplicative offset on the log2 axis so that OOM markers from
            # different implementations at the same batch do not hide each other
            off = 2.0 ** ((i - 1) * 0.05)
            ax.scatter([b * off for b in ob], [oom_y] * len(ob), marker="X", s=110,
                       color=st["color"], edgecolors="black", linewidths=0.8, zorder=4)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Requested Batch Size (log2)")
    ax.set_ylabel("Median Runtime (s)  /  Out of Memory marker")
    ax.set_title("F5  Memory Scalability on 325557_3216152 (Legacy Feasibility)")
    all_batches = sorted({b for pts in mem.values() for b in pts.keys()})
    ax.set_xticks(all_batches)
    ax.set_xticklabels([str(b) for b in all_batches], rotation=45, fontsize=8)
    ax.get_xaxis().set_minor_locator(plt.NullLocator())
    ax.set_ylim(0, oom_y * 1.12)
    succ_handles, labels = ax.get_legend_handles_labels()
    succ_handles.append(Line2D([0], [0], marker="X", color="black", linestyle="None",
                               markersize=9, label="Out of Memory (no runtime)"))
    ax.legend(handles=succ_handles, loc="center right", fontsize=9)
    ax.grid(axis="x", visible=False)
    bottom_caption(ax,
                   "Legacy feasibility result on 325557_3216152; not a current block-kernel "
                   "performance comparison.\nGPU_Opt and GPU_Opt_Pure_Chunked extend the "
                   "observed feasible range but do not provide unlimited capacity.", y=-0.30)
    return save_fig(fig, "memory_scalability_325557")


def mem_min_batch(mem):
    return min(b for pts in mem.values() for b in pts.keys())


def fig_F6(ks):
    graphs = ["roadNet-PA", "roadNet-TX"]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(graphs)); w = 0.36
    shared = [ks[g]["shared_med"] for g in graphs]
    block = [ks[g]["block_med"] for g in graphs]
    shared_sd = [ks[g]["shared_sd"] for g in graphs]
    block_sd = [ks[g]["block_sd"] for g in graphs]
    ax.bar(x - w / 2, shared, w, yerr=shared_sd, capsize=3,
           label="Shared Kernel", color=OK["orange"],
           hatch="\\\\", edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, block, w, yerr=block_sd, capsize=3,
           label="Block Kernel", color=OK["blue"],
           hatch="//", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(graphs)
    ax.set_ylabel("Median Runtime (s)")
    ax.set_xlabel("Graph")
    ax.set_title("F6  Shared vs Block BFS Kernel (forced comparison)")
    ax.set_ylim(0, max(shared) * 1.38)
    for i, g in enumerate(graphs):
        ax.text(x[i] - w / 2, shared[i] * 1.01, f"{shared[i]:.1f}s", ha="center",
                va="bottom", fontsize=8)
        ax.text(x[i] + w / 2, block[i] * 1.01, f"{block[i]:.1f}s", ha="center",
                va="bottom", fontsize=8)
        ax.annotate(f"Block {ks[g]['speedup']:.2f}x faster",
                    xy=(x[i], max(shared[i], block[i]) * 1.06),
                    ha="center", fontsize=9, fontweight="bold", color=OK["blue"])
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.text(0.01, 0.01,
             "Error bars = sample SD (n=3 per kernel). "
             "No generalization is made to unmeasured graphs.",
             fontsize=8.5, ha="left", style="italic")
    return save_fig(fig, "shared_vs_block_kernel")


def fig_F7(ph):
    fig, (ax_email, ax_road) = plt.subplots(
        1, 2, figsize=(10.2, 5.1), gridspec_kw={"width_ratios": [1, 3]})

    def stacked_panel(ax, graphs, title):
        x = np.arange(len(graphs)); w = 0.58
        bfs = [ph[g]["bfs"] for g in graphs]
        back = [ph[g]["backward"] for g in graphs]
        other = [ph[g]["other"] for g in graphs]
        ax.bar(x, bfs, w, label="BFS", color=PHASE_STYLE["BFS"]["color"],
               hatch=PHASE_STYLE["BFS"]["hatch"], edgecolor="black", linewidth=0.6)
        ax.bar(x, back, w, bottom=bfs, label="Backward",
               color=PHASE_STYLE["Backward"]["color"],
               hatch=PHASE_STYLE["Backward"]["hatch"],
               edgecolor="black", linewidth=0.6)
        bottom = [a + b for a, b in zip(bfs, back)]
        ax.bar(x, other, w, bottom=bottom, label="Other",
               color=PHASE_STYLE["Other"]["color"],
               hatch=PHASE_STYLE["Other"]["hatch"],
               edgecolor="black", linewidth=0.6)
        sums = [ph[g]["component_sum"] for g in graphs]
        ax.set_ylim(0, max(sums) * 1.20)
        ax.set_xticks(x)
        ax.set_xticklabels(graphs)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Graph")
        ax.grid(axis="x", visible=False)
        for i, g in enumerate(graphs):
            ax.text(x[i], sums[i] * 1.02, f"n={ph[g]['n']}",
                    ha="center", va="bottom", fontsize=8)

    stacked_panel(ax_email, ["email-EuAll"], "(a) email-EuAll")
    stacked_panel(ax_road, ["roadNet-PA", "roadNet-TX", "roadNet-CA"],
                  "(b) Road networks")
    ax_email.set_ylabel("Median Component Time (s)")
    ax_road.set_ylabel("Median Component Time (s)")
    handles, labels = ax_email.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92),
               ncol=3, fontsize=9)
    fig.suptitle("F7  GPU_Opt (b512) Phase Breakdown", fontsize=12)
    fig.text(0.01, 0.008,
             "Measured components: BFS and Backward wall-clock timers. Other is computed "
             "per trial as total - BFS - Backward; bars show component medians. "
             "Descriptive of measured configurations only; not causal evidence.",
             fontsize=7.5, ha="left")
    fig.tight_layout(rect=[0, 0.07, 1, 0.86])
    return save_fig(fig, "phase_breakdown")


# --------------------------------------------------------------------------- #
# Tables.
# --------------------------------------------------------------------------- #
def table_T1(stats, cat):
    order = ["email-EuAll", "roadNet-PA", "roadNet-TX", "roadNet-CA",
             "325557_3216152", "56438_300801", "benchmark_7000", "benchmark_11023",
             "benchmark_85830", "chain_200", "random"]
    header = ["Graph", "Nodes", "Edges", "Average Degree", "Maximum Degree",
              "Directed Input", "Symmetrized"]
    rows = []
    for g in order:
        s = stats[g]; c = cat[g]
        rows.append([g, s["n"], s["m"], s["avg_deg"], s["max_deg"],
                     c["DirectedOriginal"], c["Symmetrized"]])
    notes = ["Nodes / Edges / degrees from docs/graph_stats.tsv (undirected edge count m).",
             "Directed Input / Symmetrized from result/datasets/graph_catalog.tsv "
             "(\"unknown\" = not recorded for generated graphs)."]
    return write_table("T1_graph_metadata", header, rows, "T1  Graph Metadata", notes)


def table_T2(mp):
    header = ["Graph", "GPU_Opt Batch", "GPU_Opt Median (s)", "PathMerge Tuned Batch",
              "PathMerge Median (s)", "Speedup", "GPU_Opt GTEPS", "PathMerge GTEPS",
              "Trials"]
    rows = []
    for g in HEADLINE_ORDER:
        d = mp[g]
        rows.append([g, "b512", f"{d['prop']['time_med']:.2f}", d["pm"]["batch"],
                     f"{d['pm']['time_med']:.2f}", f"{d['speedup']:.2f}",
                     f"{d['prop']['gteps_med']:.2f}", f"{d['pm']['gteps_med']:.2f}",
                     f"{d['prop']['n']} / {d['pm']['n']}"])
    notes = ["Speedup = median(PathMerge Tuned) / median(GPU_Opt). GPU_Opt fixed at b512.",
             "PathMerge tuned batch per graph (email b2048, roadNet-PA/TX b64, roadNet-CA b32).",
             "Trials are listed as GPU_Opt / PathMerge. Exact canonical source paths are "
             "listed in TABLE_MANIFEST.tsv."]
    return write_table("T2_main_performance", header, rows, "T2  Main Performance", notes)


def table_T3(abl):
    header = ["Graph Group", "Hybrid BFS Effect", "Warp-Cooperative Effect",
              "Dual-Stream Effect", "Trials", "Limitation"]
    g = abl["syn_geo"]; e = abl["email"]
    rows = [
        ["Synthetic (geomean, 4 graphs)", f"{g['Hybrid BFS']:.3f}x",
         f"{g['Warp-Cooperative Accumulation']:.3f}x", f"{g['Dual Streams']:.3f}x",
         f"{abl['trials']['synthetic']} per configuration",
         "4 synthetic graphs; not generalized to roadNet"],
        ["email-EuAll (hub, real)", f"{e['Hybrid BFS']:.3f}x",
         f"{e['Warp-Cooperative Accumulation']:.3f}x", f"{e['Dual Streams']:.3f}x",
         f"{abl['trials']['email']} per configuration",
         "Single hub graph; Warp-Cooperative < 1.0x (harmful here)"],
    ]
    notes = ["Per-factor main effects are recomputed from configuration medians in the "
             "canonical raw ablation TSVs and checked against the archived contribution TSVs.",
             "Warp-Cooperative Accumulation is graph-dependent (range ~0.970x-1.175x across "
             "the 5 measured graphs)."]
    return write_table("T3_ablation_summary", header, rows, "T3  Ablation Summary", notes)


def table_T4(mem):
    """Memory feasibility table using only matching legacy TSVs and logs."""
    header = ["Implementation", "Batch Size", "Effective Batch", "Sub-Batch",
              "Number of Sub-Batches", "Median Runtime (s)", "Status", "Limitation"]
    rows = []
    for label in ["GPU_Opt_Pure", "GPU_Opt", "GPU_Opt_Pure_Chunked"]:
        pts = mem[label]
        for b in sorted(pts.keys()):
            p = pts[b]
            if p["effective"] is not None:
                eff = str(p["effective"])
                sub = str(p["sub_batch"])
                nsub = str(p["num_subs"])
            elif label == "GPU_Opt_Pure":
                eff = "Not Recorded"
                sub = "Not Applicable"
                nsub = "Not Applicable"
            else:
                eff = sub = nsub = "Not Recorded"
            if p["success"]:
                rt = f"{p['med']:.2f}"; status = "Success"
            else:
                rt = "N/A (OOM)"; status = "Out of Memory"
            lim = ("Legacy feasibility only (oldtree_f05ec52_20260512); "
                   "not current block-kernel performance")
            if label == "GPU_Opt" and b == 12288:
                lim += "; Out of Memory n=1 (sweep stopped)"
            rows.append([label, b, eff, sub, nsub, rt, status, lim])
    notes = ["Runtime and status are recomputed from the matching legacy feasibility TSVs. "
             "Successful medians use n=5; Out of Memory uses n=5 except GPU_Opt b12288 "
             "(n=1; sweep stopped). OOM is N/A, never 0 s.",
             "Effective Batch, Sub-Batch, and Number of Sub-Batches come from the matching "
             "legacy experiment logs when recorded. GPU_Opt_Pure does not record those fields.",
             "Observed feasibility in the tested range: GPU_Opt_Pure (maximum successful "
             "requested batch 4096) < GPU_Opt (10240) < GPU_Opt_Pure_Chunked (16384). "
             "This does not imply unlimited capacity."]
    return write_table("T4_memory_scalability", header, rows, "T4  Memory Scalability", notes)


def table_T5(small, mem, small_nf, mem_nf):
    header = ["Validation Scope", "Reference", "Candidate", "Graph", "Comparison Level",
              "Mismatches", "Missing Values", "Non-Finite Values",
              "Maximum Relative Error", "Status", "Limitation"]
    rows = []
    gmap = {"data/benchmark_7000_41459": "benchmark_7000_41459",
            "data/benchmark_11023_62184": "benchmark_11023_62184",
            "data/chain_200": "chain_200"}
    for r in small:
        graph = gmap[r["graph_path"]]
        rows.append([
            "Small full-vector", "Sequential (CPU)", "GPU_Opt", graph,
            "Full Vector", r["mismatched_elements"],
            str(int(r["missing_reference_only"]) + int(r["missing_candidate_only"])),
            str(small_nf[graph]), f"{float(r['max_rel_error']):.2e}",
            ("Pass" if r["status"] == "PASS" else "Core Fail"),
            "Small graph; independent CPU reference; n=1"])
    # memory-path comparison matrix
    def find(subclass, a, b):
        for r in mem:
            if r["comparison_subclass"] == subclass and r["label_a"] == a and r["label_b"] == b:
                return r
        return None
    same_path = [("gpu_opt_b1024", "gpu_opt_pure_b1024", "GPU_Opt (b1024)", "GPU_Opt_Pure (b1024)"),
                 ("gpu_opt_b1024", "gpu_opt_pure_chunked_b1024", "GPU_Opt (b1024)", "GPU_Opt_Pure_Chunked (b1024)"),
                 ("gpu_opt_pure_b1024", "gpu_opt_pure_chunked_b1024", "GPU_Opt_Pure (b1024)", "GPU_Opt_Pure_Chunked (b1024)")]
    for a, b, la, lb in same_path:
        r = find("same_batch_diff_path", a, b)
        rows.append(["Memory-path same-batch/different-path", la, lb, "325557_3216152",
                     "Full Vector", r["mismatched_elements"],
                     str(int(r["missing_a"]) + int(r["missing_b"])),
                     str(mem_nf[(a, b)]), f"{float(r['max_rel_error']):.2e}",
                     ("Pass" if r["status"] == "PASS" else "Core Fail"),
                     "Non-byte-identical (SHA256 differ) but within mixed tolerance; n=1"])
    stress = [("gpu_opt_b9792", "gpu_opt_b1024", "GPU_Opt (b9792)", "GPU_Opt (b1024)"),
              ("gpu_opt_pure_chunked_b16384", "gpu_opt_pure_chunked_b1024",
               "GPU_Opt_Pure_Chunked (b16384)", "GPU_Opt_Pure_Chunked (b1024)")]
    for a, b, la, lb in stress:
        r = find("same_impl_diff_batch", a, b)
        rows.append(["Memory-path stress (same-impl/different-batch)", la, lb, "325557_3216152",
                     "Full Vector", r["mismatched_elements"],
                     str(int(r["missing_a"]) + int(r["missing_b"])),
                     str(mem_nf[(a, b)]),
                     f"{float(r['max_rel_error']):.2e}", "Core Fail",
                     "Exceeds rel_tol 1e-6; cause not determined; not relabeled as Pass"])
    r = find("pathmerge_cross", "pathmerge_b4096", "gpu_opt_b1024")
    rows.append(["PathMerge cross-implementation diagnostic",
                 "PathMerge (b4096)", "GPU_Opt (b1024)",
                 "325557_3216152", "Full Vector", r["mismatched_elements"],
                 str(int(r["missing_a"]) + int(r["missing_b"])),
                 str(mem_nf[("pathmerge_b4096", "gpu_opt_b1024")]),
                 f"{float(r['max_rel_error']):.2e}", "Supported with Limitations",
                 "Observed difference only: external comparator is not ground truth; "
                 "correctness is undetermined"])
    notes = ["abs_tol = 1e-3, rel_tol = 1e-6 (canonical; unchanged). "
             "Non-Finite Values = count of NaN/Inf (0 = all vectors finite/valid).",
             "The canonical memory-path stress divergence is preserved as Core Fail and is "
             "not hidden or relabeled. Sources: result/correctness/small_full_vector/"
             "correctness_summary.tsv; result/correctness/memory_paths/canonical_job_2368587/"
             "comparison_matrix.tsv."]
    return write_table("T5_correctness_summary", header, rows, "T5  Correctness Summary", notes)


def table_T6(env):
    bw = env["bw"]
    header = ["Component", "Specification"]
    rows = [
        ["GPU", "NVIDIA GH200"],
        ["Nominal HBM3", "96 GB"],
        ["Recorded Device Memory", "97,871 MiB (approximately 95.6 GiB or 102.6 GB)"],
        ["Runtime-Reported Total Memory at Launch", "approximately 102.0 GB (decimal GB)"],
        ["Runtime Free Memory at Launch",
         "approximately 101.4 GB (decimal GB; memory-budget basis, not total capacity)"],
        ["NVIDIA Driver", "595.58.03"],
        ["CUDA Toolkit (nvcc)", "release 13.0, V13.0.48"],
        ["Host C++ Compiler", "g++ (GCC) 11.4.1"],
        ["CMake", "4.3.4"],
        ["Nsight Systems (nsys)", "2025.5.1.121"],
        ["PBS System", "Miyabi-G PBS batch system"],
        ["Group", "gj17"],
        ["Queue", "Not independently verifiable from retained job logs"],
        ["Resource Configuration - Memory-Path Experiments",
         "Host-memory-limited 100 GiB configuration"],
        ["HBM3 Bandwidth (Device-to-Device)", f"{bw['HBM3_DtoD']['gbs']} GB/s "
         f"({bw['HBM3_DtoD']['ratio']}% of theoretical)"],
        ["Pinned Host-to-Device Bandwidth", f"{bw['Pinned_HtoD']['gbs']} GB/s "
         f"({bw['Pinned_HtoD']['ratio']}% of theoretical)"],
        ["Pinned Device-to-Host Bandwidth", f"{bw['Pinned_DtoH']['gbs']} GB/s "
         f"({bw['Pinned_DtoH']['ratio']}% of theoretical)"],
        ["NVLink-C2C Prefetch Bandwidth", f"{bw['NVLink_C2C_Prefetch']['gbs']} GB/s "
         f"({bw['NVLink_C2C_Prefetch']['ratio']}% of theoretical)"],
        ["Main-Experiment Aggregation", "Median of all recorded trials"],
        ["Main-Experiment Warmup", "None; no recorded trial was discarded"],
    ]
    notes = [
        "GPU model, nominal HBM3, recorded device memory, software, PBS system, and group "
        "from result/environment/environment.md.",
        "The nominal 96 GB, recorded 97,871 MiB, and runtime-reported approximately 102.0 GB "
        "refer to the same HBM3 through different units or query methods, not separate memory tiers.",
        "Runtime total and free memory at launch from raw_data/main_performance/proposed_variants/"
        "email-EuAll/_run/job_2357334_20260711/phase_timing.log; free memory is the launch-time "
        "available amount used as the memory-budget basis, not total capacity.",
        "The retained job logs do not independently verify the actual queue name; it is not an "
        "evaluation control variable.",
        "Bandwidth from raw_data/profiling/job_2359175_20260711/bandwidth.log.",
    ]
    return write_table("T6_experimental_environment", header, rows,
                       "T6  Experimental Environment", notes)


# --------------------------------------------------------------------------- #
# Manifests + READMEs.
# --------------------------------------------------------------------------- #
def rel_inputs(*paths):
    return ";".join(paths)


def write_manifests_and_readmes(fig_out, tab_out, mp, meta):
    # ---- FIGURE_MANIFEST.tsv -------------------------------------------------
    fm_header = ["FigureID", "Title", "Claim", "InputFiles", "GenerationScript",
                 "Metric", "Aggregation", "Trials", "PDF", "PNG", "SVG", "Limitations"]
    gs = "scripts/generate_thesis_artifacts.py"
    main_inputs = [
        "raw_data/main_performance/proposed_variants/email-EuAll/_run/job_2357334_20260711/results.tsv",
        "raw_data/main_performance/proposed_variants/roadNet-PA/_run/job_2357334_20260711/results.tsv",
        "raw_data/main_performance/proposed_variants/roadNet-TX/_run/job_2357334_20260711/results.tsv",
        "raw_data/main_performance/proposed_variants/roadNet-CA/_run/job_2357334_20260711/results.tsv",
        "raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv",
        "raw_data/tuning/pathmerge/roadNet-CA/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv",
        "raw_data/main_performance/seven_implementations/legacy_partial/large/no_gpu_opt/"
        "job_notrecorded_legacy/results_no_gpu_opt.tsv",
    ]
    sweep_inputs = [
        "raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv",
        "raw_data/tuning/pathmerge/roadNet-PA/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv",
        "raw_data/tuning/pathmerge/roadNet-TX/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv",
        "raw_data/tuning/pathmerge/roadNet-CA/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv",
        "raw_data/tuning/pathmerge/325557/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv",
        "raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_multi_20260710/pathmerge_sweep.log",
        "raw_data/tuning/pathmerge/roadNet-PA/pathmerge_bc/job_multi_20260710/pathmerge_sweep.log",
        "raw_data/tuning/pathmerge/325557/pathmerge_bc/job_multi_20260710/pathmerge_sweep.log",
    ]
    ablation_inputs = [
        "raw_data/ablation/synthetic/job_2354994_20260710/ablation_results.tsv",
        "raw_data/ablation/email-EuAll/job_2354999_20260710/ablation_results.tsv",
        "result/ablation/synthetic_2354994/ablation_contributions.tsv",
        "result/ablation/email_2354999/ablation_contributions.tsv",
    ]
    memory_tsv_inputs = [
        "raw_data/memory_scalability/325557_3216152/gpu_opt/"
        "job_notrecorded_20260512/oversubscribe_results_gpu_opt.tsv",
        "raw_data/memory_scalability/325557_3216152/gpu_opt_pure/"
        "job_notrecorded_20260512/oversubscribe_results_gpu_opt_pure.tsv",
        "raw_data/memory_scalability/325557_3216152/gpu_opt_pure_chunked/"
        "job_notrecorded_20260512/oversubscribe_results_gpu_opt_pure_chunked.tsv",
    ]
    memory_log_inputs = [
        "raw_data/memory_scalability/325557_3216152/gpu_opt/"
        "job_notrecorded_20260512/um_experiment_gpu_opt.log",
        "raw_data/memory_scalability/325557_3216152/gpu_opt_pure/"
        "job_notrecorded_20260512/um_experiment_gpu_opt_pure.log",
        "raw_data/memory_scalability/325557_3216152/gpu_opt_pure_chunked/"
        "job_notrecorded_20260512/um_experiment_gpu_opt_pure_chunked.log",
    ]
    kernel_inputs = [
        "raw_data/tuning/kernel_selection/roadNet-PA/gpu_opt_forced/"
        "job_2354329_20260710/kernel_selection_results.tsv",
        "raw_data/tuning/kernel_selection/roadNet-TX/gpu_opt_forced/"
        "job_2354330_20260710/kernel_selection_results.tsv",
    ]
    phase_inputs = [
        f"raw_data/main_performance/proposed_variants/{g}/_run/"
        f"job_2357334_20260711/phase_timing.log" for g in HEADLINE_ORDER
    ]
    correctness_inputs = [
        "result/correctness/small_full_vector/correctness_summary.tsv",
        "result/correctness/memory_paths/canonical_job_2368587/comparison_matrix.tsv",
        "result/correctness/small_full_vector/benchmark_7000_41459/comparison.md",
        "result/correctness/small_full_vector/benchmark_11023_62184/comparison.md",
        "result/correctness/small_full_vector/chain_200/comparison.md",
        "result/correctness/memory_paths/canonical_job_2368587/"
        "gpu_opt_b1024__vs__gpu_opt_pure_b1024.md",
        "result/correctness/memory_paths/canonical_job_2368587/"
        "gpu_opt_b1024__vs__gpu_opt_pure_chunked_b1024.md",
        "result/correctness/memory_paths/canonical_job_2368587/"
        "gpu_opt_pure_b1024__vs__gpu_opt_pure_chunked_b1024.md",
        "result/correctness/memory_paths/canonical_job_2368587/"
        "gpu_opt_b9792__vs__gpu_opt_b1024.md",
        "result/correctness/memory_paths/canonical_job_2368587/"
        "gpu_opt_pure_chunked_b16384__vs__gpu_opt_pure_chunked_b1024.md",
        "result/correctness/memory_paths/canonical_job_2368587/"
        "pathmerge_b4096__vs__gpu_opt_b1024.md",
    ]
    fm = [
        ["F1", "Main Runtime Comparison",
         "GPU_Opt (b512) is faster than tuned PathMerge on all four evaluated graphs",
         rel_inputs(*main_inputs), gs, "Median Runtime (s)", "median",
         "email-EuAll 5/3; roadNet-PA/TX/CA 3/3 (GPU_Opt/PathMerge)",
         fig_out["F1"]["pdf"], fig_out["F1"]["png"], fig_out["F1"]["svg"],
         "Four graphs only; GPU_Opt fixed b512; PathMerge tuned (conservative); log y-axis"],
        ["F2", "Main Speedup over Tuned PathMerge",
         "Median/median speedup is 3.17 / 1.31 / 1.51 / 1.45",
         rel_inputs(*(main_inputs + ["result/main_performance/proposed_vs_pathmerge/comparison.tsv"])),
         gs, "Speedup over Tuned PathMerge", "median/median",
         "email-EuAll 5/3; roadNet-PA/TX/CA 3/3 (GPU_Opt/PathMerge)",
         fig_out["F2"]["pdf"], fig_out["F2"]["png"], fig_out["F2"]["svg"],
         "Results are limited to the four evaluated graphs"],
        ["F3", "PathMerge Batch Sweep",
         "PathMerge runtime depends on requested batch; recorded clamping is explicit",
         rel_inputs(*sweep_inputs), gs, "Median Runtime (s) vs Requested Batch Size",
         "median; sample SD where n>=2", "per batch n=1-4",
         fig_out["F3"]["pdf"], fig_out["F3"]["png"], fig_out["F3"]["svg"],
         "Clamped: email-EuAll 8192->7393; 325557_3216152 8192->6018; gaps not connected"],
        ["F4", "Ablation Contributions",
         "Hybrid BFS and Dual Streams help; Warp-Cooperative Accumulation is graph-dependent",
         rel_inputs(*ablation_inputs), gs, "Factor Main-Effect Speedup",
         "configuration medians; factorial and graph geometric means",
         "synthetic 5/configuration; email 3/configuration",
         fig_out["F4"]["pdf"], fig_out["F4"]["png"], fig_out["F4"]["svg"],
         "5 measured graphs; not generalized to roadNet"],
        ["F5", "Memory Scalability (325557_3216152)",
         "Observed feasible batch: GPU_Opt_Pure < GPU_Opt < GPU_Opt_Pure_Chunked",
         rel_inputs(*memory_tsv_inputs), gs, "Median Runtime (s) and Out-of-Memory Status",
         "median; sample SD for successful runs",
         "5/configuration except GPU_Opt b12288 Out of Memory n=1",
         fig_out["F5"]["pdf"], fig_out["F5"]["png"], fig_out["F5"]["svg"],
         "Legacy feasibility only; 325557_3216152 only; no unlimited-capacity claim"],
        ["F6", "Shared vs Block Kernel",
         "Block kernel is faster than shared: roadNet-PA 1.52x, roadNet-TX 1.66x",
         rel_inputs(*kernel_inputs), gs, "Median Runtime (s)", "median; sample SD", "3/kernel/graph",
         fig_out["F6"]["pdf"], fig_out["F6"]["png"], fig_out["F6"]["svg"],
         "Two graphs only; no generalization to unmeasured graphs"],
        ["F7", "Phase Breakdown",
         "GPU_Opt runtime splits into BFS, Backward, and Other (descriptive)",
         rel_inputs(*phase_inputs), gs, "Median Component Time (s)",
         "median of per-trial components", "email-EuAll 5; roadNet-PA/TX/CA 3",
         fig_out["F7"]["pdf"], fig_out["F7"]["png"], fig_out["F7"]["svg"],
         "Descriptive of measured b512 runs only; not causal evidence"],
    ]
    write_raw_manifest(FIG_DIR / "FIGURE_MANIFEST.tsv", fm_header, fm)

    # ---- TABLE_MANIFEST.tsv --------------------------------------------------
    tm_header = ["TableID", "Title", "Claim", "InputFiles", "GenerationScript",
                 "Aggregation", "Trials", "Markdown", "TSV", "Limitations"]
    tm = [
        ["T1", "Graph Metadata", "Structural properties of the evaluated graphs",
         "docs/graph_stats.tsv;result/datasets/graph_catalog.tsv", gs, "n/a", "n/a",
         tab_out["T1"]["md"], tab_out["T1"]["tsv"],
         "\"unknown\" fields not recorded for generated graphs"],
        ["T2", "Main Performance",
         "Median/median speedup 3.17 / 1.31 / 1.51 / 1.45 (GPU_Opt b512 vs tuned PathMerge)",
         rel_inputs(*main_inputs), gs, "median (speedup = median/median)",
         "email-EuAll 5/3; roadNet-PA/TX/CA 3/3 (GPU_Opt/PathMerge)",
         tab_out["T2"]["md"], tab_out["T2"]["tsv"], "Four graphs only"],
        ["T3", "Ablation Summary", "Per-factor main effects; Warp-Cooperative is graph-dependent",
         rel_inputs(*ablation_inputs), gs,
         "configuration medians; factorial and graph geometric means",
         "synthetic 5/configuration; email 3/configuration",
         tab_out["T3"]["md"], tab_out["T3"]["tsv"],
         "5 graphs; not generalized to roadNet"],
        ["T4", "Memory Scalability",
         "Observed feasible batch ordering GPU_Opt_Pure < GPU_Opt < GPU_Opt_Pure_Chunked",
         rel_inputs(*(memory_tsv_inputs + memory_log_inputs)), gs,
         "median runtime; recorded execution metadata",
         "5/configuration except GPU_Opt b12288 Out of Memory n=1",
         tab_out["T4"]["md"], tab_out["T4"]["tsv"],
         "Legacy feasibility; 325557 only; OOM shown as N/A not 0 s"],
        ["T5", "Correctness Summary",
         "Small full-vector Pass; memory-path stress Core Fail preserved",
         rel_inputs(*correctness_inputs), gs, "single-run full-vector comparison", "1/comparison",
         tab_out["T5"]["md"], tab_out["T5"]["tsv"],
         "Core Fail not relabeled; PathMerge cross-implementation correctness undetermined"],
        ["T6", "Experimental Environment", "Hardware/software environment and bandwidth",
         "result/environment/environment.md;raw_data/main_performance/proposed_variants/email-EuAll/"
         "_run/job_2357334_20260711/phase_timing.log;"
         "raw_data/profiling/job_2359175_20260711/bandwidth.log",
         gs, "n/a for specifications; single bandwidth measurement", "1 bandwidth measurement",
         tab_out["T6"]["md"], tab_out["T6"]["tsv"], "Only archived, supported specifications"],
    ]
    write_raw_manifest(TAB_DIR / "TABLE_MANIFEST.tsv", tm_header, tm)

    write_fig_readme(mp)
    write_tab_readme()


def write_raw_manifest(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def write_fig_readme(mp):
    lines = []
    lines.append("# Thesis / Presentation Figures (Gate K0)\n")
    lines.append("Generated by `scripts/generate_thesis_artifacts.py` from canonical, "
                 "Git-tracked data only. **All in-figure text is English.** Graph and "
                 "implementation names are kept verbatim.\n")
    lines.append("Regenerate:\n")
    lines.append("```bash\npython3 scripts/generate_thesis_artifacts.py\n```\n")
    lines.append("Each figure is exported as PDF (embedded fonts), PNG (300 dpi), and SVG. "
                 "See `FIGURE_MANIFEST.tsv` for per-figure inputs, metric, aggregation, "
                 "trials, and limitations.\n")
    lines.append("## Policy\n")
    lines.append("- median = numpy.median; speedup = median(PathMerge tuned) / median(GPU_Opt).\n"
                 "- OOM configurations are drawn as an explicit Out-of-Memory marker, never as 0 s.\n"
                 "- Missing / invalid measurements are not connected as if they existed.\n"
                 "- Colorblind-safe (Okabe-Ito) palette; series also distinguished by markers/hatching.\n"
                 "- Consistent graph order and implementation colors across figures.\n"
                 "- Deterministic output (fixed SOURCE_DATE_EPOCH, no embedded timestamps).\n")
    lines.append("## Figures\n")
    fig_desc = {
        "F1": ("main_runtime_comparison", "Grouped bars (log y): GPU_Opt b512 vs tuned PathMerge."),
        "F2": ("main_speedup_over_tuned_pathmerge", "Speedup bars with 1.0x parity line."),
        "F3": ("pathmerge_batch_sweep", "Per-graph sweep; screening/confirmation and clamping shown."),
        "F4": ("ablation_contributions", "Main effects + per-graph Warp-Cooperative dependence."),
        "F5": ("memory_scalability_325557", "Feasibility on 325557; OOM band (not 0 s)."),
        "F6": ("shared_vs_block_kernel", "Shared vs block kernel; block 1.52x / 1.66x faster."),
        "F7": ("phase_breakdown", "Stacked BFS / Backward / Other wall-clock time."),
    }
    for fid, (stem, desc) in fig_desc.items():
        lines.append(f"- **{fid}** `{stem}.{{pdf,png,svg}}` -- {desc}")
    lines.append("")
    lines.append("## Key values (recomputed)\n")
    lines.append("| Graph | GPU_Opt b512 median (s) | Tuned PathMerge median (s) | Speedup |")
    lines.append("|---|---|---|---|")
    for g in HEADLINE_ORDER:
        d = mp[g]
        lines.append(f"| {g} | {d['prop']['time_med']:.2f} | {d['pm']['time_med']:.2f} "
                     f"({d['pm']['batch']}) | {d['speedup']:.2f}x |")
    lines.append("")
    with open(FIG_DIR / "README.md", "w") as f:
        f.write("\n".join(lines))


def write_tab_readme():
    lines = []
    lines.append("# Thesis / Presentation Tables (Gate K0)\n")
    lines.append("Generated by `scripts/generate_thesis_artifacts.py`. Each table is written "
                 "as presentation-ready Markdown (`.md`) and machine-readable `.tsv`. "
                 "**All table text is English.** See `TABLE_MANIFEST.tsv` for provenance.\n")
    lines.append("Regenerate:\n```bash\npython3 scripts/generate_thesis_artifacts.py\n```\n")
    lines.append("## Tables\n")
    td = {
        "T1": "Graph Metadata (nodes, edges, degrees, directedness).",
        "T2": "Main Performance (GPU_Opt vs tuned PathMerge, speedup, GTEPS, trials).",
        "T3": "Ablation Summary (Hybrid BFS / Warp-Cooperative / Dual-Stream effects).",
        "T4": "Memory Scalability (feasible batch, effective/sub-batch, status).",
        "T5": "Correctness Summary (small Pass; memory-path stress Core Fail preserved).",
        "T6": "Experimental Environment (hardware, software, bandwidth).",
    }
    for tid, desc in td.items():
        lines.append(f"- **{tid}** `{tid}_*.md` / `.tsv` -- {desc}")
    lines.append("")
    lines.append("## Notes\n")
    lines.append("- Status vocabulary: Success, Out of Memory, Pass, Core Fail, "
                 "Supported with Limitations.\n"
                 "- The canonical memory-path stress result is preserved as **Core Fail** "
                 "and is never relabeled as Pass.\n"
                 "- OOM is reported as `N/A (OOM)`, never as 0 seconds.\n")
    with open(TAB_DIR / "README.md", "w") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    stats, cat = load_graph_metadata()
    mp = load_main_performance()
    sweeps = {g: load_pathmerge_sweep(g)
              for g in ["email-EuAll", "roadNet-PA", "roadNet-TX", "roadNet-CA", "325557_3216152"]}
    tuned_batch = {"email-EuAll": 2048, "roadNet-PA": 64, "roadNet-TX": 64,
                   "roadNet-CA": 32, "325557_3216152": None}
    abl = load_ablation()
    mem = load_memory_scalability()
    ks = load_kernel_selection()
    ph = load_phase_breakdown()
    small, memc, small_nf, mem_nf = load_correctness()
    env = load_environment()
    thesis_values = {r["ValueID"]: r for r in read_tsv("docs/thesis/thesis_values.tsv")}
    evidence = {r["ClaimID"]: r for r in read_tsv("docs/thesis/evidence_matrix.tsv")}

    # ---- validation checks ----
    expected_speedup = {"email-EuAll": 3.17, "roadNet-PA": 1.31,
                        "roadNet-TX": 1.51, "roadNet-CA": 1.45}
    for g, exp in expected_speedup.items():
        got = round(mp[g]["speedup"], 2)
        note(f"speedup[{g}]", got == exp, f"recomputed={mp[g]['speedup']:.4f} -> {got} (expect {exp})")
    speedup_value_ids = {
        "email-EuAll": "PERF-EMAIL-SPEEDUP", "roadNet-PA": "PERF-PA-SPEEDUP",
        "roadNet-TX": "PERF-TX-SPEEDUP", "roadNet-CA": "PERF-CA-SPEEDUP",
    }
    for g, value_id in speedup_value_ids.items():
        indexed = float(thesis_values[value_id]["Value"])
        recomputed = round(mp[g]["speedup"], 2)
        note(f"thesis_values_index[{g}]", recomputed == indexed,
             f"recomputed={recomputed:.2f} indexed={indexed:.2f}")
    comp = {r["Graph"]: r for r in read_tsv("result/main_performance/proposed_vs_pathmerge/comparison.tsv")}
    for g in HEADLINE_ORDER:
        cp = round(mp[g]["prop"]["time_med"], 2); cc = float(comp[g]["Proposed_block_s"])
        note(f"gpu_opt_median[{g}]", cp == cc, f"recomputed={cp} canonical={cc}")
        mm = round(mp[g]["pm"]["time_med"], 2); mc = float(comp[g]["PathMerge_tuned_s"])
        note(f"pathmerge_median[{g}]", mm == mc, f"recomputed={mm} canonical={mc}")
    # OOM never zero: assert every plotted memory point is a real success time
    oom_ok = all((p["med"] is None) == (not p["success"]) for pts in mem.values() for p in pts.values())
    note("oom_not_zero", oom_ok, "all OOM points have med=None (drawn as OOM marker, not 0 s)")
    # Core Fail present
    core_fail = any(r["comparison_subclass"] == "same_impl_diff_batch" and r["status"] == "FAIL"
                    for r in memc)
    note("core_fail_present", core_fail, "same_impl_diff_batch FAIL present in comparison_matrix")
    note("core_fail_evidence_status",
         evidence["C-CORR-STRESS"]["Status"] == "NOT_YET_SUPPORTED",
         f"evidence status={evidence['C-CORR-STRESS']['Status']}")
    note("clamp[email-EuAll,b8192]",
         sweeps["email-EuAll"][8192]["effective"] == 7393,
         f"effective={sweeps['email-EuAll'][8192]['effective']}")
    note("clamp[325557_3216152,b8192]",
         sweeps["325557_3216152"][8192]["effective"] == 6018,
         f"effective={sweeps['325557_3216152'][8192]['effective']}")
    note("phase_other_nonnegative", all(d["other"] >= 0 for d in ph.values()),
         "per-trial Other medians are non-negative")
    memory_trial_counts_ok = all(
        p["trials"] == (1 if label == "GPU_Opt" and b == 12288 else 5)
        for label, pts in mem.items() for b, p in pts.items())
    note("memory_trials", memory_trial_counts_ok,
         "n=5 except GPU_Opt b12288 Out of Memory n=1 (sweep stopped)")

    # Confirm that generation reads only canonical paths, never build_miyabi,
    # and that every recorded input is Git-tracked.
    allowed_inputs = all(
        rel.startswith("raw_data/") or rel.startswith("result/") or
        rel in {"docs/graph_stats.tsv", "docs/thesis/evidence_matrix.tsv",
                "docs/thesis/thesis_values.tsv"}
        for rel in INPUTS_USED)
    note("canonical_input_scope", allowed_inputs,
         "all inputs are under raw_data/, result/, or indexed thesis metadata sources")
    no_build_input = all("build_miyabi" not in rel for rel in INPUTS_USED)
    note("no_build_miyabi_dependency", no_build_input,
         "no recorded input path contains build_miyabi")
    untracked_inputs = []
    for rel in sorted(INPUTS_USED):
        repo_rel = str(Path(ROOT.name) / rel)
        proc = subprocess.run(
            ["git", "-C", str(ROOT.parent), "ls-files", "--error-unmatch", "--", repo_rel],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if proc.returncode != 0:
            untracked_inputs.append(rel)
    note("all_inputs_git_tracked", not untracked_inputs,
         "all inputs tracked" if not untracked_inputs else f"untracked={untracked_inputs}")

    # ---- figures ----
    fig_out = {
        "F1": fig_F1(mp), "F2": fig_F2(mp), "F3": fig_F3(sweeps, tuned_batch),
        "F4": fig_F4(abl), "F5": fig_F5(mem), "F6": fig_F6(ks), "F7": fig_F7(ph),
    }
    # ---- tables ----
    tab_out = {
        "T1": table_T1(stats, cat), "T2": table_T2(mp), "T3": table_T3(abl),
        "T4": table_T4(mem),
        "T5": table_T5(small, memc, small_nf, mem_nf), "T6": table_T6(env),
    }
    write_manifests_and_readmes(fig_out, tab_out, mp, dict())
    core_fail_rows = sum(1 for line in (TAB_DIR / "T5_correctness_summary.tsv").read_text().splitlines()
                         if "\tCore Fail\t" in line)
    note("core_fail_table_rows", core_fail_rows == 2,
         f"T5 contains {core_fail_rows} Core Fail rows")

    # ---- report ----
    print("=== Gate K0 generation: validation ===")
    all_ok = True
    for name, ok, detail in VALIDATION:
        flag = "OK " if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{flag}] {name}: {detail}")
    print(f"\nFigures: {sorted(fig_out)}  Tables: {sorted(tab_out)}")
    print(f"Distinct canonical inputs used: {len(INPUTS_USED)}")
    print("ALL_VALIDATION_OK" if all_ok else "VALIDATION_FAILURES_PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
