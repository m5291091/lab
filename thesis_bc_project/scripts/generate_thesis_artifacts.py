#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_thesis_artifacts.py -- Gate K0 / Gate W7.4.1

Generate a reproducible, English-only set of figures and tables for the
master's thesis and presentation, using ONLY canonical data under
thesis_bc_project/{raw_data,result,docs}.

Gate W7.4.1 switches T3/F4 (ablation), T4/F5 (memory feasibility), and T5
(correctness) to the corrected-325557 official inputs (jobs 2404743 / 2406254,
checkpoint 45352a3). The old malformed-325557 values are retained only as
historical and are never the current main value.

Design rules (see result/figures/thesis/README.md for the full policy):
  * Every displayed value is recomputed from a canonical input file and
    cross-checked against the audited derived TSVs / thesis_values index; no
    fallback to stored, rounded, interpolated, or reverse-computed values.
  * median = numpy.median; speedup = median(PathMerge_tuned) / median(GPU_Opt).
  * Failed configurations are NEVER represented as zero seconds -- they are
    drawn as distinct failure markers. In the corrected memory feasibility
    boundary, a CUDA (GPU-device) out-of-memory and a cgroup host-memory OOM
    kill (SIGKILL, exit 137) are separate classes with separate markers and are
    never conflated.
  * Corrected-325557 correctness comparisons are numerically consistent within
    the mixed tolerance (abs_tol 1e-3, rel_tol 1e-6) with mismatch 0, but are
    NOT byte-identical; PathMerge is an external comparator, not ground truth.
  * All in-figure / in-table text is English. Graph and implementation names
    are kept verbatim (not translated).
  * No dependency on build_miyabi/. Deterministic output (fixed SOURCE_DATE_EPOCH,
    no embedded timestamps, fixed SVG hash salt). Matplotlib binaries are only
    reproducible run-to-run within one toolchain, so THESIS_FIGS selects which
    figures to (re)render; this gate regenerates only the corrected F4 and F5.

Run:  THESIS_FIGS=F4,F5 python3 scripts/generate_thesis_artifacts.py
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

# --------------------------------------------------------------------------- #
# Corrected 325557 (Gate W7.4) OFFICIAL inputs for T3/F4, T4/F5, T5.
# These supersede the old malformed 325557 as the CURRENT thesis values.
# --------------------------------------------------------------------------- #
CORRECTED_GRAPH = "325557_3216152_corrected_v1"
CORRECTED_GRAPH_SHA256 = \
    "8373244f209a3ee489fe72a7b237a5639d142e3a10ac451a2c81b09194eeaa22"
CORRECTED_CHECKPOINT = "45352a344aaac463283a647467b790be9b45bfb8"
CORRECTED_JOB_CORRECTNESS_MEM = "2404743"   # Series A/B: correctness + feasibility
CORRECTED_JOB_ABLATION = "2406254"          # Series C: ablation
CORRECTNESS_ABS_TOL = "1e-3"
CORRECTNESS_REL_TOL = "1e-6"
GEN_COMMAND = "THESIS_FIGS=F4,F5 python3 scripts/generate_thesis_artifacts.py"

# --------------------------------------------------------------------------- #
# UM b12288 failure evidence (Gate T1B1.1): TWO layers, deliberately separate.
#
#   1. Runtime classifier record -- retained raw provenance, never rewritten.
#      During the run, run_corrected_325557_validation.sh classified each
#      configuration by scanning ONLY that configuration's stdout/stderr
#      (classify_observed "${CFG_STDERR}" "${CFG_OUTPUT}"). For um_b12288 it
#      recorded OOMEvidenceClass=none with runner_exit=137 (SIGKILL-compatible).
#
#   2. Post-hoc archive audit -- this record. The PBS epilogue appends a direct
#      cgroup OOM line at JOB END, after "=== Complete ===" and after the
#      validation script had already classified every configuration. It was
#      therefore outside the running classifier's scan scope and could not have
#      been observed by it.
#
# Layer 1 is NOT an error corrected by layer 2: the two differ in what they were
# able to inspect. The runtime `none` is a scan-scope artifact and is reported
# alongside the post-hoc class, never replaced by it.
UM_B12288_EVIDENCE_PATH = (
    f"raw_data/corrected_325557/job_{CORRECTED_JOB_CORRECTNESS_MEM}/pbs_stdout.log")
UM_B12288_EVIDENCE_SHA256 = \
    "3c4c46680f9432b94fef79ca9344027ad77195973d075b8019379f934feb8ec5"
UM_B12288_EVIDENCE_LINE = 146
UM_B12288_EVIDENCE_TOKEN = "oom-kill:constraint=CONSTRAINT_MEMCG"
UM_B12288_EVIDENCE_CLASS = "kernel_oom_kill"   # scripts/oom_evidence.sh vocabulary
UM_B12288_RUNTIME_CLASSIFIER = "none"          # what layer 1 recorded
UM_B12288_CONFIG_MARKER = "[B 4/5] um_b12288:"

# Directories whose files are Gate-W7.4 corrected-input artifacts that are
# legitimately still pending commit at gate time (Gate W7.4.1 does not commit).
# They are canonical (under result/ or raw_data/) but not yet Git-tracked, so
# the "all inputs Git tracked" audit treats them as a known pending-commit set.
PENDING_COMMIT_INPUT_PREFIXES = (
    "raw_data/corrected_325557/",
    "result/ablation/corrected_325557/",
    "result/memory_scalability/corrected_325557/",
    "result/correctness/corrected_325557/",
)

# Stable figure file stems (used so the manifest can reference figures whose
# binaries are NOT regenerated in this run -- see FIGURES_TO_GENERATE).
FIG_STEMS = {
    "F1": "main_runtime_comparison",
    "F2": "main_speedup_over_tuned_pathmerge",
    "F3": "pathmerge_batch_sweep",
    "F4": "ablation_contributions",
    "F5": "memory_scalability_325557",
    "F6": "shared_vs_block_kernel",
    "F7": "phase_breakdown",
}
# Matplotlib binary output is only reproducible run-to-run WITHIN one toolchain
# environment, not byte-identically across matplotlib builds. The committed
# F1/F2/F3/F6/F7 came from the original toolchain, so this gate regenerates ONLY
# the corrected-data figures (F4, F5) and leaves the others byte-invariant.
# THESIS_FIGS (comma-separated IDs) selects which figures to (re)render; unset
# means "all" (full from-scratch regeneration in the original toolchain).
_ENV_FIGS = os.environ.get("THESIS_FIGS", "").strip()
FIGURES_TO_GENERATE = (set(FIG_STEMS) if not _ENV_FIGS
                       else {f.strip() for f in _ENV_FIGS.split(",") if f.strip()})

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


def _normalize_svg(path):
    """Deterministically normalize an SVG *in place*, touching ONLY line-terminal
    whitespace:

      * strip trailing spaces / tabs / CRs from the end of every line,
      * force LF line endings,
      * end the file with exactly one trailing newline.

    matplotlib emits multi-line ``<path d="...">`` data where each coordinate line
    ends with a space before the newline; removing that trailing space leaves the
    newline as a valid SVG token separator, so no number, command, element, or
    attribute value changes meaning. XML structure, path data, numeric values,
    element order, and attribute values are preserved verbatim otherwise.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized = "\n".join(ln.rstrip(" \t") for ln in lines).rstrip("\n") + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(normalized)


def save_fig(fig, stem):
    """Save PDF + PNG(300dpi) + SVG deterministically (no timestamps)."""
    base = FIG_DIR / stem
    fig.savefig(str(base) + ".pdf",
                metadata={"Creator": "generate_thesis_artifacts.py",
                          "Producer": "matplotlib"})
    fig.savefig(str(base) + ".png", dpi=300,
                metadata={"Software": "generate_thesis_artifacts.py"})
    svg_path = str(base) + ".svg"
    fig.savefig(svg_path,
                metadata={"Creator": "generate_thesis_artifacts.py", "Date": None})
    _normalize_svg(svg_path)   # strip line-terminal whitespace only (git diff --check clean)
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


FACTOR_KEY = {"H": "Hybrid BFS", "W": "Warp-Cooperative Accumulation",
              "A": "Dual Streams"}
FACTOR_POS = {"H": 0, "W": 1, "A": 2}


def _config_medians(rows):
    """Group per-trial ablation rows into per-configuration medians."""
    by_cfg = {}
    for r in rows:
        m = re.fullmatch(r"Ablation_H([01])_W([01])_A([01])", r["Config"])
        if not m:
            raise ValueError(f"unexpected ablation config: {r['Config']}")
        by_cfg.setdefault(tuple(int(v) for v in m.groups()), []).append(
            float(r["Time_sec"]))
    counts = {len(ts) for ts in by_cfg.values()}
    if len(by_cfg) != 8 or len(counts) != 1:
        raise ValueError(f"incomplete/ragged 2^3 ablation: {sorted(by_cfg)} counts={counts}")
    return {cfg: median(ts) for cfg, ts in by_cfg.items()}, counts.pop()


def _main_effects(config_medians):
    """Per-factor main effect = geomean of T(F=0)/T(F=1) over the 4 pairings."""
    eff = {}
    for factor, label in FACTOR_KEY.items():
        pos = FACTOR_POS[factor]
        ratios = []
        for cfg0 in sorted(c for c in config_medians if c[pos] == 0):
            cfg1 = list(cfg0)
            cfg1[pos] = 1
            ratios.append(config_medians[cfg0] / config_medians[tuple(cfg1)])
        eff[label] = float(np.exp(np.mean(np.log(ratios))))
    return eff


def load_ablation():
    """Recompute the CORRECTED-325557 ablation and the updated synthetic-4
    aggregate from canonical per-trial TSVs.

    The synthetic-4 aggregate is a MIXED-CHECKPOINT geometric mean: the three
    non-325557 synthetic graphs come from job 2354994 (unchanged raw), and the
    old malformed 325557 is replaced by the corrected-325557 re-measurement
    (job 2406254, checkpoint 45352a3). Each per-graph main effect is recomputed
    from raw and cross-checked against the audited result TSVs; nothing falls
    back to a stored/rounded value.
    """
    # --- 3 synthetic graphs (exclude the OLD malformed 325557_3216152) -------
    syn_rows = read_tsv(
        "raw_data/ablation/synthetic/job_2354994_20260710/ablation_results.tsv")
    other3 = ["benchmark_7000_41459", "benchmark_11023_62184", "56438_300801"]
    by_graph = {g: [] for g in other3}
    for r in syn_rows:
        if r["Graph"] in by_graph:
            by_graph[r["Graph"]].append(r)
    per_graph = {}
    trial_set = set()
    for g in other3:
        medians, n = _config_medians(by_graph[g])
        per_graph[g] = _main_effects(medians)
        trial_set.add(n)

    # --- corrected 325557 (job 2406254) --------------------------------------
    cor_rows = read_tsv(
        f"raw_data/corrected_325557/job_{CORRECTED_JOB_ABLATION}/ablation_results.tsv")
    if {r["Graph"] for r in cor_rows} != {CORRECTED_GRAPH}:
        raise ValueError("corrected ablation raw contains unexpected graph name")
    cor_medians, cor_n = _config_medians(cor_rows)
    trial_set.add(cor_n)
    per_graph[CORRECTED_GRAPH] = _main_effects(cor_medians)
    if len(trial_set) != 1:
        raise ValueError(f"inconsistent ablation trial counts across graphs: {trial_set}")
    trials = trial_set.pop()

    order = other3 + [CORRECTED_GRAPH]
    mixed_geo = {label: float(np.exp(np.mean(np.log([per_graph[g][label] for g in order]))))
                 for label in FACTOR_KEY.values()}
    base_full = {"H0W0A0": cor_medians[(0, 0, 0)], "H1W1A1": cor_medians[(1, 1, 1)]}

    # --- cross-checks against the audited derived TSVs (hard fail on drift) ---
    agg = {r["Factor"]: r for r in read_tsv(
        "result/ablation/corrected_325557/synthetic4_aggregate.tsv")}
    per_graph_expected = {}       # (graph, factor) -> audited MainEffect
    for factor, row in agg.items():
        label = FACTOR_KEY[factor]
        note(f"ablation_mixed_geomean[{factor}]",
             round(mixed_geo[label], 4) == round(float(row["NewGeomean_mixed_checkpoint"]), 4),
             f"recomputed={mixed_geo[label]:.4f} audited={row['NewGeomean_mixed_checkpoint']}")
        for item in row["PerGraphMainEffects"].split(";"):
            gname, val = item.split("=")
            gkey = CORRECTED_GRAPH if gname == "325557_corrected" else gname
            per_graph_expected[(gkey, factor)] = float(val)
    for g in order:
        for factor, label in FACTOR_KEY.items():
            exp = per_graph_expected[(g, factor)]
            note(f"ablation_main_effect[{g},{factor}]",
                 round(per_graph[g][label], 4) == round(exp, 4),
                 f"recomputed={per_graph[g][label]:.4f} audited={exp:.4f}")
    contrib = {r["Factor"]: float(r["MainEffect"]) for r in read_tsv(
        "result/ablation/corrected_325557/ablation_contributions.tsv")}
    for factor, label in FACTOR_KEY.items():
        note(f"ablation_corrected325557[{factor}]",
             round(per_graph[CORRECTED_GRAPH][label], 4) == round(contrib[factor], 4),
             f"recomputed={per_graph[CORRECTED_GRAPH][label]:.4f} audited={contrib[factor]:.4f}")

    return dict(per_graph=per_graph, order=order, mixed_geo=mixed_geo,
                corrected=per_graph[CORRECTED_GRAPH], base_full=base_full,
                trials=trials, factor_key=FACTOR_KEY)


def verify_um_b12288_posthoc_evidence():
    """Re-verify, from the retained archive itself, the direct cgroup OOM record
    for the UM b12288 configuration of job 2404743.

    This is a POST-HOC ARCHIVE AUDIT layered on top of -- never a replacement
    for -- the runtime classifier record (see the UM_B12288_* constants). The
    archive is only ever READ here; nothing under raw_data/ is modified.

    Every check is hard. A missing file, a changed hash, a moved evidence line,
    a foreign job or configuration context, a missing cgroup token, or an
    ambiguous kill attribution aborts generation. The published claim is never
    emitted from a hard-coded string alone, and a failed verification is never
    silently downgraded to "no evidence".
    """
    path = input_path(UM_B12288_EVIDENCE_PATH)
    if not path.is_file():
        raise RuntimeError(
            f"post-hoc OOM evidence missing: {UM_B12288_EVIDENCE_PATH}")

    got_sha = _sha256_file(path)
    if got_sha != UM_B12288_EVIDENCE_SHA256:
        raise RuntimeError(
            f"post-hoc OOM evidence hash mismatch for {UM_B12288_EVIDENCE_PATH}: "
            f"expected {UM_B12288_EVIDENCE_SHA256}, got {got_sha}")

    with open(path) as f:
        lines = f.read().splitlines()
    if len(lines) < UM_B12288_EVIDENCE_LINE:
        raise RuntimeError(
            f"post-hoc OOM evidence line {UM_B12288_EVIDENCE_LINE} is beyond the end "
            f"of {UM_B12288_EVIDENCE_PATH} ({len(lines)} lines)")

    exact = lines[UM_B12288_EVIDENCE_LINE - 1]
    if UM_B12288_EVIDENCE_TOKEN not in exact:
        raise RuntimeError(
            f"{UM_B12288_EVIDENCE_PATH}:{UM_B12288_EVIDENCE_LINE} does not carry the "
            f"cgroup OOM token {UM_B12288_EVIDENCE_TOKEN!r}")
    if CORRECTED_JOB_CORRECTNESS_MEM not in exact:
        raise RuntimeError(
            f"{UM_B12288_EVIDENCE_PATH}:{UM_B12288_EVIDENCE_LINE} does not name the "
            f"expected job {CORRECTED_JOB_CORRECTNESS_MEM}")

    # Configuration context: the kill must be attributable to um_b12288 and to no
    # other configuration recorded in the same job.
    announce = [i for i, ln in enumerate(lines)
                if ln.startswith(UM_B12288_CONFIG_MARKER)]
    killed = [i for i, ln in enumerate(lines) if " Killed " in ln]
    cgroup = [i for i, ln in enumerate(lines) if UM_B12288_EVIDENCE_TOKEN in ln]
    complete = [i for i, ln in enumerate(lines) if ln.startswith("=== Complete ===")]
    if len(announce) != 1:
        raise RuntimeError(
            f"expected exactly one {UM_B12288_CONFIG_MARKER!r} block in "
            f"{UM_B12288_EVIDENCE_PATH}, found {len(announce)}")
    if len(killed) != 1:
        raise RuntimeError(
            f"kill attribution is ambiguous: expected exactly one SIGKILL record in "
            f"{UM_B12288_EVIDENCE_PATH}, found {len(killed)}")
    if len(cgroup) != 1 or cgroup[0] != UM_B12288_EVIDENCE_LINE - 1:
        raise RuntimeError(
            f"expected exactly one cgroup OOM record at line {UM_B12288_EVIDENCE_LINE} "
            f"of {UM_B12288_EVIDENCE_PATH}, found {len(cgroup)} at "
            f"{[i + 1 for i in cgroup]}")

    # The single SIGKILL record must sit inside the um_b12288 block, i.e. between
    # its announcement and its own outcome line.
    outcome_idx = next((i for i in range(announce[0] + 1, len(lines))
                        if lines[i].lstrip().startswith("observed=")), None)
    if outcome_idx is None:
        raise RuntimeError(
            f"no outcome line follows {UM_B12288_CONFIG_MARKER!r} in "
            f"{UM_B12288_EVIDENCE_PATH}")
    if not announce[0] < killed[0] < outcome_idx:
        raise RuntimeError(
            f"the SIGKILL record at line {killed[0] + 1} of {UM_B12288_EVIDENCE_PATH} "
            f"does not lie inside the um_b12288 block "
            f"(lines {announce[0] + 1}..{outcome_idx + 1})")
    if "RUNTIME_FAILED" not in lines[outcome_idx]:
        raise RuntimeError(
            f"um_b12288 outcome line {outcome_idx + 1} of {UM_B12288_EVIDENCE_PATH} is "
            f"not RUNTIME_FAILED: {lines[outcome_idx].strip()!r}")

    # Epilogue position: the cgroup record is appended after the validation script
    # finished, which is why the running classifier could not observe it.
    if not complete or cgroup[0] < complete[0]:
        raise RuntimeError(
            f"the cgroup OOM record at line {cgroup[0] + 1} of "
            f"{UM_B12288_EVIDENCE_PATH} does not follow '=== Complete ==='; the "
            f"post-job epilogue position could not be confirmed")

    note("um_b12288_posthoc_cgroup_oom", True,
         f"{UM_B12288_EVIDENCE_PATH}:{UM_B12288_EVIDENCE_LINE} verified "
         f"({UM_B12288_EVIDENCE_CLASS}); sha256 {UM_B12288_EVIDENCE_SHA256[:16]}...; "
         f"job {CORRECTED_JOB_CORRECTNESS_MEM}; sole SIGKILL (line {killed[0] + 1}) lies "
         f"inside the um_b12288 block (lines {announce[0] + 1}..{outcome_idx + 1}); "
         f"appended after '=== Complete ===' (line {complete[0] + 1}), hence outside the "
         f"runtime per-config classifier scope that recorded "
         f"OOMEvidenceClass={UM_B12288_RUNTIME_CLASSIFIER}")

    return dict(path=UM_B12288_EVIDENCE_PATH, line=UM_B12288_EVIDENCE_LINE,
                sha256=UM_B12288_EVIDENCE_SHA256,
                evidence_class=UM_B12288_EVIDENCE_CLASS,
                runtime_classifier=UM_B12288_RUNTIME_CLASSIFIER,
                exact_line=exact, sigkill_line=killed[0] + 1,
                epilogue_after_line=complete[0] + 1)


def load_memory_scalability():
    """Load the CORRECTED-325557 targeted feasibility-boundary validation.

    This is a targeted 5-point boundary confirmation (each configuration n=1),
    NOT a batch sweep and NOT a performance comparison. Successful wall-clock
    times are reported verbatim; failures carry a distinct, evidence-backed
    class -- CUDA (GPU-device) out-of-memory vs a host/cgroup memory OOM kill
    (SIGKILL, exit 137) -- and are NEVER represented as a runtime.
    """
    boundary = read_tsv(
        "result/memory_scalability/corrected_325557/feasibility_boundary.tsv")
    # Post-hoc archive audit for the um_b12288 SIGKILL (verified against the
    # retained PBS stdout; aborts generation if the evidence does not check out).
    posthoc = verify_um_b12288_posthoc_evidence()
    # Independent raw cross-checks (same job 2404743).
    raw_feas = {r["Config"]: r for r in read_tsv(
        f"raw_data/corrected_325557/job_{CORRECTED_JOB_CORRECTNESS_MEM}/feasibility_results.tsv")}
    raw_oom = {r["Config"]: r for r in read_tsv(
        f"raw_data/corrected_325557/job_{CORRECTED_JOB_CORRECTNESS_MEM}/oom_evidence.tsv")}

    order = ["pure_b4096", "pure_b8192", "um_b10240", "um_b12288", "chunked_b16384"]
    out = []
    for row in boundary:
        cfg = row["Config"]
        impl = row["Implementation"]
        batch = int(row["RequestedBatch"])
        observed = row["Observed"]
        exit_code = int(row["RunnerExit"])
        evid_class = row["OOMEvidenceClass"]
        runtime = None if row["RuntimeSec"] == "not_recorded" else float(row["RuntimeSec"])

        # Classify the outcome into the three current-thesis categories.
        if observed == "SUCCESS":
            outcome, fail_class = "SUCCESS", None
        elif observed == "OOM_CONFIRMED" and evid_class == "cuda_oom" and exit_code == 1:
            outcome, fail_class = "CUDA_OOM", "cuda_oom"
        elif observed == "RUNTIME_FAILED" and exit_code == 137 and evid_class == "none":
            outcome, fail_class = "CGROUP_OOM_KILL", "cgroup_host_oom_kill"
        else:
            raise ValueError(f"unclassifiable feasibility outcome for {cfg}: {row}")

        # Cross-check against the raw feasibility + OOM-evidence tables.
        rf, ro = raw_feas[cfg], raw_oom[cfg]
        note(f"memory_boundary[{cfg}]",
             (rf["Observed"] == observed and int(rf["RunnerExit"]) == exit_code
              and ro["OOMEvidenceClass"] == evid_class),
             f"boundary vs raw feasibility/oom_evidence agree ({observed}, exit {exit_code}, {evid_class})")

        out.append(dict(
            config=cfg, impl=impl, batch=batch, outcome=outcome,
            fail_class=fail_class, runtime=runtime, exit=exit_code,
            evidence_class=evid_class,
            evidence_line=(ro["ExactMatchedLine"] if fail_class == "cuda_oom" else None),
            posthoc=(posthoc if cfg == "um_b12288" else None),
            success=(outcome == "SUCCESS"), trials=1))
    got_order = [d["config"] for d in out]
    if got_order != order:
        raise ValueError(f"unexpected feasibility-boundary ordering: {got_order}")
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


import json


def _parse_impl_batch(label):
    """gpu_opt_pure_chunked_b16384 -> ('GPU_Opt_Pure_Chunked', 16384)."""
    m = re.fullmatch(r"(gpu_opt_pure_chunked|gpu_opt_pure|gpu_opt|pathmerge)_b(\d+)", label)
    if not m:
        raise ValueError(f"unrecognized comparison label: {label}")
    impl = {"gpu_opt": "GPU_Opt", "gpu_opt_pure": "GPU_Opt_Pure",
            "gpu_opt_pure_chunked": "GPU_Opt_Pure_Chunked",
            "pathmerge": "PathMerge"}[m.group(1)]
    return impl, int(m.group(2))


def load_small_correctness():
    """Tier A -- Independent CPU reference. Load the Sequential-vs-GPU_Opt
    full-vector correctness for the three small graphs from the AUDITED canonical
    summary (result/correctness/small_full_vector/correctness_summary.tsv; PBS job
    2367583; abs_tol 1e-3, rel_tol 1e-6). Every displayed value is read from that
    TSV, never transcribed from prose. This is a comparison against an INDEPENDENT
    Sequential CPU reference (unlike the corrected-325557 cross-implementation
    consistency check)."""
    rows = read_tsv("result/correctness/small_full_vector/correctness_summary.tsv")
    graph_order = ["benchmark_7000_41459", "benchmark_11023_62184", "chain_200"]
    by_graph = {Path(r["graph_path"]).name: r for r in rows}
    if set(by_graph) != set(graph_order):
        raise ValueError(f"unexpected small-correctness graphs: {sorted(by_graph)}")
    out = []
    for g in graph_order:
        r = by_graph[g]
        missing = int(r["missing_reference_only"]) + int(r["missing_candidate_only"])
        mismatch = int(r["mismatched_elements"])
        byte_identical = (r["sequential_vector_sha256"] == r["gpu_opt_vector_sha256"])
        vec_ok = int(r["reference_vector_length"]) == int(r["candidate_vector_length"])
        checks = (missing == 0 and mismatch == 0 and r["status"] == "PASS"
                  and int(r["sequential_exit"]) == 0 and int(r["gpu_opt_exit"]) == 0
                  and int(r["comparison_exit"]) == 0 and vec_ok and not byte_identical)
        note(f"small_correctness[{g}]", checks,
             f"missing={missing} mismatch={mismatch} status={r['status']} "
             f"byte_identical={byte_identical}")
        out.append(dict(
            graph=g, ref_impl="Sequential", cand_impl="GPU_Opt",
            ref_batch="N/A", cand_batch=int(r["effective_batch"]),
            vec_len=int(r["reference_vector_length"]),
            missing=missing, mismatch=mismatch,
            max_abs=float(r["max_abs_error"]), max_rel=float(r["max_rel_error"]),
            byte_identical=byte_identical, tol_result=r["status"]))
    if len(out) != 3:
        raise ValueError(f"expected 3 small-correctness rows, got {len(out)}")
    return out


def load_correctness():
    """Load the CORRECTED-325557 full-vector correctness: 6 validated vectors
    and all 10 cross-implementation comparisons (job 2404743).

    Every displayed value is recomputed from the per-comparison raw JSON and
    cross-checked against the audited comparison_summary / vector_summary TSVs.
    All 10 comparisons are numerically consistent within the mixed tolerance
    (abs_tol 1e-3, rel_tol 1e-6) with mismatched_elements == 0, but the vectors
    are NOT byte-identical (per-implementation SHA256 differ). PathMerge is an
    external comparator, not an independent ground truth.
    """
    summary = read_tsv("result/correctness/corrected_325557/comparison_summary.tsv")
    vecs = {r["Config"]: r for r in read_tsv(
        "result/correctness/corrected_325557/vector_summary.tsv")}
    if any(v["Status"] != "PASS" for v in vecs.values()) or len(vecs) != 6:
        raise ValueError("corrected-325557 vector_summary is not 6 PASS vectors")
    if len({v["SHA256"] for v in vecs.values()}) != 6:
        raise ValueError("expected 6 distinct per-implementation SHA256 (non-byte-identical)")

    comps = []
    for r in summary:
        a, b = r["LabelA"], r["LabelB"]
        jpath = (f"raw_data/corrected_325557/job_{CORRECTED_JOB_CORRECTNESS_MEM}/"
                 f"comparisons/{a}__vs__{b}.json")
        with open(input_path(jpath)) as f:
            j = json.load(f)
        # Recompute-and-verify against the audited summary row.
        checks = (
            j["mismatched_elements"] == int(r["MismatchedElements"])
            and j["status"] == r["Status"]
            and j["length_a"] == j["length_b"] == j["expected_length"]
            and j["sha256_a"] != j["sha256_b"])
        note(f"correctness_comparison[{a}__vs__{b}]", checks,
             f"mismatch={j['mismatched_elements']} status={j['status']} "
             f"len={j['expected_length']} byte_identical={j['sha256_a'] == j['sha256_b']}")
        ra = j["vector_a"]; rb = j["vector_b"]
        nonfinite = sum(v[k] for v in (ra, rb)
                        for k in ("nan_values", "positive_inf_values", "negative_inf_values"))
        ref_impl, ref_batch = _parse_impl_batch(a)
        cand_impl, cand_batch = _parse_impl_batch(b)
        comps.append(dict(
            klass=r["ComparisonClass"], ref_label=a, cand_label=b,
            ref_impl=ref_impl, ref_batch=ref_batch,
            cand_impl=cand_impl, cand_batch=cand_batch,
            vec_len=j["expected_length"],
            missing=j["missing_a"] + j["missing_b"],
            mismatch=j["mismatched_elements"], nonfinite=nonfinite,
            max_abs=j["max_abs_error"], max_rel=j["max_rel_error"],
            byte_identical=(j["sha256_a"] == j["sha256_b"]),
            tol_result=("PASS" if j["status"] == "PASS" else j["status"])))
    if len(comps) != 10:
        raise ValueError(f"expected 10 corrected-325557 comparisons, got {len(comps)}")
    return dict(comparisons=comps, vectors=vecs)


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


GRAPH_DISPLAY = {
    "benchmark_7000_41459": "benchmark_7000_41459",
    "benchmark_11023_62184": "benchmark_11023_62184",
    "56438_300801": "56438_300801",
    CORRECTED_GRAPH: "325557_3216152\n(corrected)",
}


def fig_F4(abl):
    factors = ["Hybrid BFS", "Warp-Cooperative Accumulation", "Dual Streams"]
    short = {"Hybrid BFS": "Hybrid BFS",
             "Warp-Cooperative Accumulation": "Warp-Cooperative\nAccumulation",
             "Dual Streams": "Dual Streams"}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 5.0),
                                   gridspec_kw={"width_ratios": [1, 1.5]})

    # Panel (a): synthetic-4 mixed-checkpoint aggregate main effects.
    x = np.arange(len(factors)); w = 0.6
    for xi, f in zip(x, factors):
        v = abl["mixed_geo"][f]
        st = FACTOR_STYLE[f]
        axA.bar(xi, v, w, color=st["color"], hatch=st["hatch"],
                edgecolor="black", linewidth=0.6)
        axA.text(xi, v + 0.02, f"{v:.3f}x", ha="center", va="bottom", fontsize=9)
    axA.axhline(1.0, color=OK["vermillion"], linestyle="--", linewidth=1.3,
                label="No effect (1.0x)")
    axA.set_xticks(x); axA.set_xticklabels([short[f] for f in factors], fontsize=9)
    axA.set_ylabel("Main Effect (geometric-mean speedup)")
    axA.set_title("(a) Synthetic-4 aggregate (mixed-checkpoint)")
    axA.set_ylim(0, 2.05)
    axA.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    axA.grid(axis="x", visible=False)

    # Panel (b): per-graph main effects (4 synthetic graphs; corrected 325557).
    order = abl["order"]
    xb = np.arange(len(order)); bw = 0.26
    for fi, f in enumerate(factors):
        st = FACTOR_STYLE[f]
        vals = [abl["per_graph"][g][f] for g in order]
        pos = xb + (fi - 1) * bw
        axB.bar(pos, vals, bw, color=st["color"], hatch=st["hatch"],
                edgecolor="black", linewidth=0.6,
                label=f.replace(" Accumulation", ""))
        for xi, v in zip(pos, vals):
            axB.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.8)
    axB.axhline(1.0, color=OK["vermillion"], linestyle="--", linewidth=1.3)
    axB.set_xticks(xb)
    axB.set_xticklabels([GRAPH_DISPLAY[g] for g in order], rotation=20, ha="right",
                        fontsize=7.8)
    axB.set_ylabel("Per-Graph Main Effect (speedup)")
    axB.set_title("(b) Per-graph main effects (4 synthetic graphs)")
    axB.set_ylim(0, 2.35)
    axB.legend(loc="upper left", fontsize=8, ncol=1, framealpha=0.95)
    axB.grid(axis="x", visible=False)

    fig.suptitle("F4  Ablation Contributions (4 synthetic graphs; mixed-checkpoint aggregate)",
                 fontsize=12)
    fig.text(0.5, 0.02,
             "Aggregate (a) is a MIXED-CHECKPOINT geometric mean: 3 graphs from job 2354994; "
             "corrected 325557 from job 2406254 (checkpoint 45352a3).\n"
             "Per-graph values (b) and the aggregate are distinct and not interchangeable. "
             "n=5 per configuration; the per-invocation untimed H1W1A1 warm-up is excluded from the "
             "40 formal rows.\n"
             "Warp-Cooperative Accumulation is graph-dependent (56438_300801 < 1.0x). "
             "Not generalized to roadNet.",
             fontsize=7.5, ha="center", va="bottom")
    fig.tight_layout(rect=[0, 0.14, 1, 0.95])
    return save_fig(fig, "ablation_contributions")


def fig_F5(mem):
    """Targeted feasibility-boundary validation on the corrected 325557 graph.

    Not a sweep: each point is a single tested (implementation, requested-batch)
    configuration (n=1). Successful runs are placed at their wall-clock time;
    the two failures sit in a distinct failure band (never at 0 s) with separate
    markers for a GPU-device CUDA OOM vs a host/cgroup memory OOM kill. Points
    are never connected, because unmeasured batches were not measured.
    """
    succ = [d for d in mem if d["success"]]
    ymax = max(d["runtime"] for d in succ)
    fail_y = ymax * 1.25            # failure band position -- NOT zero seconds
    band_lo = ymax * 1.12
    ax = plt.subplots(figsize=(9.2, 5.6))[1]
    fig = ax.figure
    ax.axhspan(band_lo, fail_y * 1.10, color=OK["vermillion"], alpha=0.08, zorder=0)
    ax.axhline(band_lo, color=OK["vermillion"], linestyle=":", linewidth=1.0)
    batches = [d["batch"] for d in mem]
    ax.text(min(batches), fail_y * 1.05,
            "Failure band (run did not complete; not a runtime value)",
            fontsize=8, color=OK["vermillion"], va="bottom")

    impl_seen = set()
    fail_markers = {"cuda_oom": "X", "cgroup_host_oom_kill": "P"}
    seen_fail = set()
    for d in mem:
        st = IMPL_STYLE[d["impl"]]
        lbl = d["impl"] if d["impl"] not in impl_seen else None
        impl_seen.add(d["impl"])
        if d["success"]:
            ax.scatter([d["batch"]], [d["runtime"]], marker=st["marker"], s=115,
                       color=st["color"], edgecolors="black", linewidths=0.8,
                       label=lbl, zorder=3)
            ax.annotate(f"{d['runtime']:.1f}s", xy=(d["batch"], d["runtime"]),
                        xytext=(0, 8), textcoords="offset points", ha="center",
                        fontsize=8)
        else:
            if lbl:   # ensure the implementation appears in the legend
                ax.scatter([], [], marker=st["marker"], s=115, color=st["color"],
                           edgecolors="black", linewidths=0.8, label=lbl)
            seen_fail.add(d["fail_class"])
            ax.scatter([d["batch"]], [fail_y], marker=fail_markers[d["fail_class"]],
                       s=150, color=st["color"], edgecolors="black", linewidths=1.0,
                       zorder=4)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Requested Batch Size (log2; targeted boundary points only)")
    ax.set_ylabel("Single-run wall-clock time (s)  /  failure marker")
    ax.set_title("F5  Memory Feasibility Boundary Validation (corrected 325557)")
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches], rotation=45, fontsize=8.5)
    ax.get_xaxis().set_minor_locator(plt.NullLocator())
    ax.set_xlim(2 ** (np.log2(min(batches)) - 0.4), 2 ** (np.log2(max(batches)) + 0.4))
    ax.set_ylim(0, fail_y * 1.14)

    handles, _ = ax.get_legend_handles_labels()
    if "cuda_oom" in seen_fail:
        handles.append(Line2D([0], [0], marker="X", color="black", linestyle="None",
                              markersize=10,
                              label="CUDA out-of-memory (GPU device; exit 1; no runtime)"))
    if "cgroup_host_oom_kill" in seen_fail:
        handles.append(Line2D([0], [0], marker="P", color="black", linestyle="None",
                              markersize=10,
                              label="Cgroup host-memory OOM kill (SIGKILL exit 137; not CUDA/HBM)"))
    ax.legend(handles=handles, loc="center left", fontsize=8.3, framealpha=0.95)
    ax.grid(axis="x", visible=False)
    bottom_caption(ax,
                   "Targeted feasibility boundary on the corrected 325557 graph (job 2404743); "
                   "each configuration n=1. Wall-clock times are single-run feasibility values at "
                   "different requested batches and are NOT a performance comparison.\n"
                   "Max successful requested batch is within the tested range only "
                   "(Pure 4096 < UM 10240 < Chunked 16384); Chunked was tested to 16384 and this is "
                   "no unlimited-capacity claim. Points are not connected (unmeasured batches were "
                   "not measured).", y=-0.30)
    return save_fig(fig, "memory_scalability_325557")


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
    # Graph -> presentation role. The corrected 325557 graph is used ONLY for
    # ablation (RQ2), memory scalability (RQ3), and correctness (RQ4); it is NOT
    # part of the RQ1 main-performance comparison. The old malformed 325557 input
    # is retained only as a historical, superseded input.
    used_for = {
        "email-EuAll": "Main performance (RQ1); Ablation",
        "roadNet-PA": "Main performance (RQ1); Kernel selection",
        "roadNet-TX": "Main performance (RQ1); Kernel selection",
        "roadNet-CA": "Main performance (RQ1)",
        "325557_3216152_corrected_v1": "Ablation; Memory scalability; Correctness",
        "56438_300801": "Ablation",
        "benchmark_7000": "Ablation; Correctness",
        "benchmark_11023": "Ablation; Correctness",
        "benchmark_85830": "Auxiliary",
        "chain_200": "Correctness",
        "random": "Auxiliary",
        "325557_3216152": "Historical (superseded by corrected_v1)",
    }
    order = ["email-EuAll", "roadNet-PA", "roadNet-TX", "roadNet-CA",
             "325557_3216152_corrected_v1", "56438_300801", "benchmark_7000",
             "benchmark_11023", "benchmark_85830", "chain_200", "random",
             "325557_3216152"]
    header = ["Graph", "Nodes", "Edges", "Average Degree",
              "Input File [MiB]", "CSR [MiB]", "Used For"]
    rows = []
    for g in order:
        c = cat[g]
        csr_mib = int(c["CSRBytes"]) / 1_048_576
        rows.append([g, c["n"], c["m"], c["avg_deg"],
                     c["FileSizeMiB"], f"{csr_mib:.2f}", used_for[g]])
    notes = [
        "Nodes / Edges (undirected edge count m) / Average Degree and file sizes from "
        "result/datasets/graph_catalog.tsv. Input File [MiB] is the on-disk CSR text input "
        "file (FileSizeBytes / 1,048,576); CSR [MiB] is the in-memory CSR array "
        "((n + 1) + 2m) x 4 bytes / 1,048,576.",
        "Input File [MiB] and CSR [MiB] are the STATIC graph representation on disk / in "
        "host memory; they are NOT the GPU working set. The GPU working set is the "
        "batch-dependent per-source state (Chapter 8), not the graph file size.",
        "The corrected 325557 graph (325557_3216152_corrected_v1, SHA256 8373244f..., "
        "checkpoint 45352a3) is used for Ablation / Memory scalability / Correctness only, "
        "NOT for the RQ1 main-performance comparison. The old malformed 325557_3216152 "
        "(SHA256 a095b2e7...) is retained only as a historical, superseded input.",
    ]
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
    header = ["Row", "Hybrid BFS Effect", "Warp-Cooperative Effect",
              "Dual-Stream Effect", "Trials", "Note"]

    def cells(eff):
        return [f"{eff['Hybrid BFS']:.4f}x",
                f"{eff['Warp-Cooperative Accumulation']:.4f}x",
                f"{eff['Dual Streams']:.4f}x"]

    trials = f"{abl['trials']} per configuration"
    rows = [["Synthetic-4 aggregate (mixed-checkpoint)"] + cells(abl["mixed_geo"]) +
            [trials, "Geometric mean of the 4 per-graph effects; mixed checkpoints (see notes)"]]
    for g in abl["order"]:
        if g == CORRECTED_GRAPH:
            label = "325557_3216152_corrected_v1"
            gnote = f"Corrected re-measurement (job {CORRECTED_JOB_ABLATION}, checkpoint 45352a3); supersedes old malformed 325557"
        else:
            label = g
            gnote = "job 2354994 (unchanged raw)"
        rows.append([label] + cells(abl["per_graph"][g]) + [trials, gnote])
    notes = [
        "The synthetic-4 aggregate is a MIXED-CHECKPOINT geometric mean: three graphs from "
        "job 2354994 and the corrected 325557 from job 2406254 (checkpoint 45352a3). It is not "
        "a single-checkpoint re-measurement of all four graphs.",
        "Per-graph and aggregate effects are distinct. Prose rounding of the aggregate is "
        "H = 1.679x, W = 1.066x, A = 1.391x. The old malformed-325557 headline "
        "(H = 1.655, W = 1.065, A = 1.396) is retained only as a historical value and is not the "
        "current main value.",
        f"n=5 per configuration; the per-invocation untimed H1W1A1 warm-up is excluded from the "
        f"40 formal rows (corrected 325557: H0W0A0 median {abl['base_full']['H0W0A0']:.2f} s, "
        f"H1W1A1 median {abl['base_full']['H1W1A1']:.2f} s). Warp-Cooperative Accumulation is "
        "graph-dependent (56438_300801 = 0.9916x < 1.0). Not generalized to roadNet.",
    ]
    return write_table("T3_ablation_summary", header, rows,
                       "T3  Ablation Summary (corrected 325557; mixed-checkpoint aggregate)", notes)


def _cuda_oom_snippet(evidence_line):
    """host_pure.cu:144: out of memory  (strip the absolute source prefix)."""
    if not evidence_line:
        return ""
    m = re.search(r"([^/\s]+\.cu:\d+:.*)$", evidence_line)
    return m.group(1) if m else evidence_line


def table_T4(mem):
    """Corrected-325557 targeted feasibility-boundary validation (n=1 each)."""
    header = ["Implementation", "Requested Batch", "Observed Outcome",
              "Failure Class", "Runtime (s)", "Runner Exit", "OOM Evidence", "Note"]
    outcome_label = {"SUCCESS": "Success",
                     "CUDA_OOM": "CUDA out-of-memory",
                     "CGROUP_OOM_KILL": "Cgroup host-memory OOM kill"}
    fail_label = {None: "None", "cuda_oom": "CUDA OOM",
                  "cgroup_host_oom_kill": "Cgroup host-memory OOM kill"}
    runtime_cell = {"SUCCESS": None, "CUDA_OOM": "N/A (CUDA OOM)",
                    "CGROUP_OOM_KILL": "N/A (cgroup host-memory OOM kill)"}
    per_note = {
        "pure_b4096": "Feasibility run (n=1)",
        "pure_b8192": "Confirmed CUDA out-of-memory (host_pure.cu:144)",
        "um_b10240": "Feasibility run (n=1); migration volume and memory placement not measured",
        "um_b12288": "Host/cgroup memory limit exceeded; not a CUDA or HBM out-of-memory",
        "chunked_b16384": "Tested upper limit (no unlimited-capacity claim)",
    }
    rows = []
    for d in mem:
        if d["success"]:
            rt = f"{d['runtime']:.2f}"
            evidence = "none"
        else:
            rt = runtime_cell[d["outcome"]]
            if d["fail_class"] == "cuda_oom":
                evidence = f"cuda_oom ({_cuda_oom_snippet(d['evidence_line'])})"
            elif d["posthoc"]:
                # Both evidence layers, in one cell: what the runtime classifier
                # recorded, and the post-hoc PBS-epilogue record that backs the
                # cgroup class. Neither is hidden behind the other.
                ph = d["posthoc"]
                evidence = (f"runtime classifier: {ph['runtime_classifier']}; "
                            f"post-hoc PBS epilogue: {ph['evidence_class']} "
                            f"({Path(ph['path']).name}:{ph['line']})")
            else:
                evidence = "none (SIGKILL, exit 137)"
        rows.append([d["impl"], d["batch"], outcome_label[d["outcome"]],
                     fail_label[d["fail_class"]], rt, d["exit"], evidence,
                     per_note[d["config"]]])
    ph = next((d["posthoc"] for d in mem if d["posthoc"]), None)
    if ph is None:
        raise RuntimeError("T4: the post-hoc um_b12288 evidence record is missing")
    notes = [
        f"Targeted feasibility-boundary validation on the corrected 325557 graph "
        f"(job {CORRECTED_JOB_CORRECTNESS_MEM}, checkpoint 45352a3); each configuration n=1. "
        "This confirms feasibility ordering, not performance. Runtimes are single-run wall-clock "
        "values at different requested batches and are not a performance comparison. Failures are "
        "shown as N/A, never 0 s.",
        "Two failure classes are kept distinct: GPU_Opt_Pure b8192 is a confirmed CUDA "
        "(GPU-device) out-of-memory (runner exit 1, host_pure.cu:144: out of memory); GPU_Opt "
        "b12288 is a host/cgroup memory OOM kill (SIGKILL, exit 137) and is NOT a CUDA or HBM "
        "out-of-memory. The two are never conflated.",
        f"UM b12288 evidence is recorded in two layers. (1) Runtime classifier: during the run the "
        f"per-configuration classifier scanned only that configuration's stdout/stderr and recorded "
        f"OOMEvidenceClass={ph['runtime_classifier']} with runner exit 137 (SIGKILL-compatible); "
        f"that record is retained unchanged. (2) Post-hoc archive audit: the PBS epilogue of the "
        f"same job carries a direct cgroup OOM record at {ph['path']}:{ph['line']} "
        f"(class {ph['evidence_class']}; file SHA256 {ph['sha256']}). The epilogue is appended at "
        f"job end -- after '=== Complete ===' (line {ph['epilogue_after_line']}) -- so it was "
        f"outside the running classifier's scan scope. The runtime "
        f"{ph['runtime_classifier']} is therefore a scan-scope artifact, not a value contradicted "
        f"or corrected by the audit. The kill is attributable to this configuration: the job's sole "
        f"SIGKILL record (line {ph['sigkill_line']}) lies inside the um_b12288 block.",
        "Observed feasible ordering within the tested range only: GPU_Opt_Pure (maximum "
        "successful requested batch 4096) < GPU_Opt (10240) < GPU_Opt_Pure_Chunked (16384). "
        "Chunked was tested to 16384; this is no unlimited-capacity claim. The input file is "
        "about 43.25 MiB; capacity pressure is the batch-dependent working set, not the input "
        "graph. Corrected 325557 only; not generalized to other graphs or GPUs.",
    ]
    return write_table("T4_memory_scalability", header, rows,
                       "T4  Memory Feasibility Boundary Validation (corrected 325557)", notes)


def table_T5(corr, small):
    """Full-vector correctness assembled from TWO distinct kinds of evidence:

      * Panel A -- Independent CPU reference (Tier A): 3 small graphs validated
        against an independent Sequential CPU reference (job 2367583).
      * Panel B -- Cross-implementation consistency (Tier B): all 10 corrected-
        325557 cross-implementation comparisons (job 2404743).

    Tier A is a comparison against an independent ground-truth reference; Tier B is
    an implementation-consistency check on the same corrected input, NOT a
    comparison against an independent ground truth. Every value is read from
    audited canonical artifacts. Total = 13 rows."""
    header = ["EvidenceTier", "Graph", "Reference", "Candidate", "ReferenceBatch",
              "CandidateBatch", "VectorLength", "MissingIndices", "MismatchedElements",
              "MaxAbsoluteError", "MaxRelativeError", "ByteIdentical",
              "ToleranceResult", "CorrectnessScope"]
    tier_a = "Independent CPU reference"
    tier_b = "Cross-implementation consistency"
    scope_a = "Full-vector comparison against independent Sequential CPU reference"
    scope_b = "Full-vector cross-implementation comparison on corrected 325557 graph"
    rows = []
    # Panel A -- independent CPU reference (Sequential vs GPU_Opt) on 3 small graphs.
    for s in small:
        rows.append([
            tier_a, s["graph"], s["ref_impl"], s["cand_impl"],
            s["ref_batch"], s["cand_batch"], s["vec_len"], s["missing"], s["mismatch"],
            f"{s['max_abs']:.3e}", f"{s['max_rel']:.3e}",
            "No" if not s["byte_identical"] else "Yes",
            s["tol_result"], scope_a])
    # Panel B -- cross-implementation consistency on the corrected 325557 graph.
    for c in corr["comparisons"]:
        rows.append([
            tier_b, CORRECTED_GRAPH, c["ref_impl"], c["cand_impl"],
            c["ref_batch"], c["cand_batch"], c["vec_len"], c["missing"], c["mismatch"],
            f"{c['max_abs']:.3e}", f"{c['max_rel']:.3e}",
            "No" if not c["byte_identical"] else "Yes",
            c["tol_result"], scope_b])
    notes = [
        "Panel A (EvidenceTier = Independent CPU reference): the three small graphs are validated "
        "full-vector against an INDEPENDENT Sequential CPU reference (Reference = Sequential, "
        "Candidate = GPU_Opt; PBS job 2367583). Panel B (EvidenceTier = Cross-implementation "
        f"consistency): the corrected {CORRECTED_GRAPH} graph is checked for full-vector agreement "
        "across the six per-implementation BC vectors (job 2404743); this is an implementation-"
        "consistency check, NOT a comparison against an independent ground truth.",
        f"abs_tol = {CORRECTNESS_ABS_TOL}, rel_tol = {CORRECTNESS_REL_TOL}. PASS means zero "
        "mismatched elements under the predefined mixed tolerance; PASS does NOT imply byte-identical "
        "output (per-implementation SHA256 differ, so ByteIdentical = No). All 13 rows have "
        "MissingIndices = 0, MismatchedElements = 0, and ToleranceResult = PASS.",
        "PathMerge is an external comparator, not a ground-truth implementation; the PathMerge rows "
        "are a numerical-agreement check, not an exact-match claim. Maximum betweenness centrality "
        "agrees across all corrected-325557 implementations at vertex index 272816.",
        "Historical results obtained from the malformed-325557 input (former Core Fail, "
        "canonical_job_2368587) are EXCLUDED from this active correctness table and retained only as "
        f"provenance (superseded by job {CORRECTED_JOB_CORRECTNESS_MEM}).",
    ]
    return write_table("T5_correctness_summary", header, rows,
                       "T5  Correctness and Numerical Behavior", notes)


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


def write_manifests_and_readmes(fig_out, tab_out, mp, corr, small):
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
        f"raw_data/corrected_325557/job_{CORRECTED_JOB_ABLATION}/ablation_results.tsv",
        "result/ablation/corrected_325557/ablation_contributions.tsv",
        "result/ablation/corrected_325557/synthetic4_aggregate.tsv",
    ]
    memory_inputs = [
        "result/memory_scalability/corrected_325557/feasibility_boundary.tsv",
        f"raw_data/corrected_325557/job_{CORRECTED_JOB_CORRECTNESS_MEM}/feasibility_results.tsv",
        f"raw_data/corrected_325557/job_{CORRECTED_JOB_CORRECTNESS_MEM}/oom_evidence.tsv",
    ]
    # T4 additionally cites the post-hoc cgroup OOM record; F5 is unchanged by
    # this gate and keeps its original input set.
    memory_inputs_t4 = memory_inputs + [UM_B12288_EVIDENCE_PATH]
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
        "result/correctness/corrected_325557/comparison_summary.tsv",
        "result/correctness/corrected_325557/vector_summary.tsv",
    ] + [
        f"raw_data/corrected_325557/job_{CORRECTED_JOB_CORRECTNESS_MEM}/comparisons/"
        f"{c['ref_label']}__vs__{c['cand_label']}.json"
        for c in corr["comparisons"]
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
        ["F4", "Ablation Contributions (corrected 325557; mixed-checkpoint aggregate)",
         "Hybrid BFS and Dual Streams help; Warp-Cooperative Accumulation is graph-dependent "
         "(corrected 325557 H=1.4767x W=1.1012x A=1.5563x; synthetic-4 mixed aggregate "
         "H=1.679x W=1.066x A=1.391x)",
         rel_inputs(*ablation_inputs), gs, "Factor Main-Effect Speedup",
         "configuration medians; factorial and graph geometric means (mixed-checkpoint aggregate)",
         "5 per configuration (corrected 325557 job 2406254; other 3 job 2354994)",
         fig_out["F4"]["pdf"], fig_out["F4"]["png"], fig_out["F4"]["svg"],
         "4 synthetic graphs; mixed checkpoints (325557 corrected via job 2406254 / checkpoint "
         "45352a3; others job 2354994); per-graph and aggregate distinct; warm-up excluded from "
         "40 formal rows; not generalized to roadNet"],
        ["F5", "Memory Feasibility Boundary Validation (corrected 325557)",
         "Feasible batch within tested range: GPU_Opt_Pure 4096 < GPU_Opt 10240 < "
         "GPU_Opt_Pure_Chunked 16384 (Pure b8192 CUDA OOM; UM b12288 cgroup host-memory OOM kill)",
         rel_inputs(*memory_inputs), gs,
         "Single-run wall-clock time and failure class",
         "single run per configuration (no cross-trial aggregation)",
         "n=1 per configuration (targeted boundary validation)",
         fig_out["F5"]["pdf"], fig_out["F5"]["png"], fig_out["F5"]["svg"],
         "Corrected 325557 only (job 2404743 / checkpoint 45352a3); targeted boundary not a sweep; "
         "runtimes not a performance comparison; CUDA OOM (device) vs cgroup host-memory OOM kill "
         "(exit 137) distinct; failures shown as markers not 0 s; no unlimited-capacity claim"],
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
        ["T3", "Ablation Summary (corrected 325557; mixed-checkpoint aggregate)",
         "Corrected 325557 H=1.4767x W=1.1012x A=1.5563x; synthetic-4 mixed aggregate "
         "H=1.679x W=1.066x A=1.391x; Warp-Cooperative is graph-dependent",
         rel_inputs(*ablation_inputs), gs,
         "configuration medians; factorial and graph geometric means (mixed-checkpoint aggregate)",
         "5 per configuration (corrected 325557 job 2406254; other 3 job 2354994)",
         tab_out["T3"]["md"], tab_out["T3"]["tsv"],
         "4 synthetic graphs; mixed checkpoints; per-graph and aggregate distinct; warm-up "
         "excluded from 40 formal rows; old malformed-325557 values retained only as historical; "
         "not generalized to roadNet"],
        ["T4", "Memory Feasibility Boundary Validation (corrected 325557)",
         "Feasible batch within tested range: GPU_Opt_Pure 4096 < GPU_Opt 10240 < "
         "GPU_Opt_Pure_Chunked 16384 (Pure b8192 CUDA OOM; UM b12288 cgroup host-memory OOM kill)",
         rel_inputs(*memory_inputs_t4), gs,
         "single-run wall-clock time; evidence-backed failure class",
         "n=1 per configuration (targeted boundary validation)",
         tab_out["T4"]["md"], tab_out["T4"]["tsv"],
         "Corrected 325557 only (job 2404743 / checkpoint 45352a3); failures shown as N/A not 0 s; "
         "CUDA OOM (device, exit 1) vs cgroup host-memory OOM kill (exit 137) distinct; runtimes not "
         "a performance comparison; no unlimited-capacity claim; UM b12288 cgroup OOM is a post-hoc "
         "PBS-epilogue record, while the runtime per-config classifier recorded none"],
        ["T5", "Correctness and Numerical Behavior",
         "3 independent Sequential-CPU-reference small-graph checks (Tier A) + 10 cross-"
         "implementation comparisons on corrected 325557 (Tier B); all 13 have MissingIndices=0, "
         "MismatchedElements=0, PASS within mixed tolerance, ByteIdentical=No",
         rel_inputs(*correctness_inputs), gs,
         "single-run full-vector comparison (abs_tol 1e-3, rel_tol 1e-6)",
         "1 per comparison (3 independent CPU-reference + 10 cross-implementation = 13; "
         "jobs 2367583 / 2404743)",
         tab_out["T5"]["md"], tab_out["T5"]["tsv"],
         "Tier A is a comparison against an independent Sequential CPU reference; Tier B is a cross-"
         "implementation consistency check on the corrected 325557 graph (not an independent ground "
         "truth). Numerically consistent within mixed tolerance but not byte-identical (SHA256 "
         "differ); PathMerge is an external comparator, not ground truth; old malformed-input Core "
         "Fail retained only as historical provenance (superseded by job 2404743)"],
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
    lines.append("Regenerate the corrected-data figures (F4, F5) for Gate W7.4.1:\n")
    lines.append(f"```bash\n{GEN_COMMAND}\n```\n")
    lines.append("`THESIS_FIGS` selects which figures to (re)render. F1/F2/F3/F6/F7 are preserved "
                 "byte-for-byte from their original toolchain (matplotlib binary output is only "
                 "reproducible run-to-run within one toolchain), so this gate regenerates ONLY F4 "
                 "and F5 (corrected 325557). Each figure is exported as PDF (embedded fonts), "
                 "PNG (300 dpi), and SVG. See `FIGURE_MANIFEST.tsv` for per-figure provenance.\n")
    lines.append("## Policy\n")
    lines.append("- median = numpy.median; speedup = median(PathMerge tuned) / median(GPU_Opt).\n"
                 "- Failed configurations are drawn as explicit failure markers, never as 0 s. In F5, "
                 "a CUDA (GPU-device) out-of-memory and a cgroup host-memory OOM kill (exit 137) use "
                 "distinct markers and are never conflated.\n"
                 "- Missing / unmeasured points are not connected as if they existed.\n"
                 "- Colorblind-safe (Okabe-Ito) palette; series also distinguished by markers/hatching.\n"
                 "- Consistent graph order and implementation colors across figures.\n"
                 "- Deterministic output (fixed SOURCE_DATE_EPOCH, no embedded timestamps).\n")
    lines.append("## Figures\n")
    fig_desc = {
        "F1": ("main_runtime_comparison", "Grouped bars (log y): GPU_Opt b512 vs tuned PathMerge."),
        "F2": ("main_speedup_over_tuned_pathmerge", "Speedup bars with 1.0x parity line."),
        "F3": ("pathmerge_batch_sweep", "Per-graph sweep; screening/confirmation and clamping shown."),
        "F4": ("ablation_contributions", "Corrected 325557: synthetic-4 mixed-checkpoint aggregate "
               "main effects + per-graph main effects (Warp-Cooperative graph-dependence)."),
        "F5": ("memory_scalability_325557", "Corrected 325557 targeted feasibility boundary; failure "
               "band (not 0 s) distinguishes CUDA OOM (device) from cgroup host-memory OOM kill (exit 137)."),
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
    lines.append(f"Regenerate:\n```bash\n{GEN_COMMAND}\n```\n")
    lines.append("## Tables\n")
    td = {
        "T1": "Graph Metadata (nodes, edges, degrees, directedness).",
        "T2": "Main Performance (GPU_Opt vs tuned PathMerge, speedup, GTEPS, trials).",
        "T3": "Ablation Summary (corrected 325557 H/W/A + synthetic-4 mixed-checkpoint aggregate).",
        "T4": "Memory Feasibility Boundary Validation (corrected 325557; 5 boundary points, n=1).",
        "T5": "Correctness and Numerical Behavior (Tier A: independent Sequential-CPU-reference on "
              "3 small graphs; Tier B: cross-implementation consistency on corrected 325557; "
              "13 comparisons, mismatch 0).",
        "T6": "Experimental Environment (hardware, software, bandwidth).",
    }
    for tid, desc in td.items():
        lines.append(f"- **{tid}** `{tid}_*.md` / `.tsv` -- {desc}")
    lines.append("")
    lines.append("## Notes\n")
    lines.append("- Status vocabulary: Success, CUDA out-of-memory, Cgroup host-memory OOM kill, "
                 "PASS, No (byte-identical).\n"
                 "- T3/T4/T5 use the corrected 325557 official inputs (jobs 2404743 / 2406254, "
                 "checkpoint 45352a3). The old malformed-325557 Core Fail and legacy values are "
                 "retained only as historical and are not the current judgment.\n"
                 "- T4 failures are reported as `N/A (CUDA OOM)` / `N/A (cgroup host-memory OOM kill)`, "
                 "never as 0 seconds; a CUDA (device) OOM and a cgroup host-memory OOM kill (exit 137) "
                 "are kept distinct.\n"
                 "- T5 carries two evidence tiers: Tier A (Independent CPU reference) validates 3 "
                 "small graphs against an independent Sequential CPU reference; Tier B (Cross-"
                 "implementation consistency) checks the corrected 325557 graph across 6 per-"
                 "implementation vectors. All 13 comparisons are numerically consistent within the "
                 "mixed tolerance (abs_tol 1e-3, rel_tol 1e-6) with MissingIndices = 0 and "
                 "MismatchedElements = 0, but ByteIdentical = No; PathMerge is an external "
                 "comparator, not ground truth.\n")
    with open(TAB_DIR / "README.md", "w") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------- #
# Corrected-325557 artifact provenance manifest (Gate W7.4.1).
# --------------------------------------------------------------------------- #
import hashlib


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_corrected_provenance():
    """Record, for each artifact updated with corrected-325557 data, the full
    provenance the gate requires (canonical inputs, corrected graph SHA256,
    checkpoint, PBS job, generation command, artifact SHA256, mixed-checkpoint
    flag, memory failure class, correctness tolerance, limitation)."""
    abl_inputs = ("raw_data/corrected_325557/job_2406254/ablation_results.tsv;"
                  "raw_data/ablation/synthetic/job_2354994_20260710/ablation_results.tsv;"
                  "result/ablation/corrected_325557/ablation_contributions.tsv;"
                  "result/ablation/corrected_325557/synthetic4_aggregate.tsv")
    mem_inputs = ("result/memory_scalability/corrected_325557/feasibility_boundary.tsv;"
                  "raw_data/corrected_325557/job_2404743/feasibility_results.tsv;"
                  "raw_data/corrected_325557/job_2404743/oom_evidence.tsv")
    # T4 additionally cites the post-hoc cgroup OOM record (F5 is unchanged here).
    mem_inputs_t4 = mem_inputs + ";" + UM_B12288_EVIDENCE_PATH
    corr_inputs = ("result/correctness/small_full_vector/correctness_summary.tsv;"
                   "result/correctness/corrected_325557/comparison_summary.tsv;"
                   "result/correctness/corrected_325557/vector_summary.tsv;"
                   "raw_data/corrected_325557/job_2404743/comparisons/*.json")
    tol = f"abs_tol={CORRECTNESS_ABS_TOL}, rel_tol={CORRECTNESS_REL_TOL}"
    specs = [
        ("T3", ["result/tables/thesis/T3_ablation_summary.md",
                "result/tables/thesis/T3_ablation_summary.tsv"], abl_inputs,
         CORRECTED_JOB_ABLATION, "Yes", "not_applicable", "not_applicable",
         "4 synthetic graphs; per-graph and aggregate distinct; warm-up excluded from 40 formal rows"),
        ("F4", ["result/figures/thesis/ablation_contributions.pdf",
                "result/figures/thesis/ablation_contributions.png",
                "result/figures/thesis/ablation_contributions.svg"], abl_inputs,
         CORRECTED_JOB_ABLATION, "Yes", "not_applicable", "not_applicable",
         "4 synthetic graphs; mixed-checkpoint aggregate; not generalized to roadNet"),
        ("T4", ["result/tables/thesis/T4_memory_scalability.md",
                "result/tables/thesis/T4_memory_scalability.tsv"], mem_inputs_t4,
         CORRECTED_JOB_CORRECTNESS_MEM, "No",
         "CUDA OOM (device, exit 1) vs cgroup host-memory OOM kill (exit 137; runtime classifier "
         f"none, post-hoc PBS epilogue {UM_B12288_EVIDENCE_CLASS} at "
         f"{UM_B12288_EVIDENCE_PATH}:{UM_B12288_EVIDENCE_LINE})", "not_applicable",
         "targeted boundary n=1; runtimes not a performance comparison; no unlimited-capacity claim"),
        ("F5", ["result/figures/thesis/memory_scalability_325557.pdf",
                "result/figures/thesis/memory_scalability_325557.png",
                "result/figures/thesis/memory_scalability_325557.svg"], mem_inputs,
         CORRECTED_JOB_CORRECTNESS_MEM, "No",
         "CUDA OOM (device, exit 1) vs cgroup host-memory OOM kill (exit 137)", "not_applicable",
         "targeted boundary n=1; failures as markers not 0 s; points not connected"),
        ("T5", ["result/tables/thesis/T5_correctness_summary.md",
                "result/tables/thesis/T5_correctness_summary.tsv"], corr_inputs,
         CORRECTED_JOB_CORRECTNESS_MEM, "No", "not_applicable", tol,
         "13 comparisons (3 Tier A independent CPU reference + 10 Tier B cross-implementation) "
         "mismatch 0; ByteIdentical No; PathMerge is an external comparator, not ground truth"),
    ]
    header = ["ArtifactID", "ArtifactPath", "ArtifactSHA256", "CanonicalInputPaths",
              "CorrectedGraphSHA256", "CheckpointSHA", "PBSJobID", "GenerationCommand",
              "MixedCheckpoint", "MemoryFailureClass", "CorrectnessTolerance", "Limitation"]
    rows = []
    for aid, paths, inputs, job, mixed, failclass, ctol, lim in specs:
        for rel in paths:
            rows.append([aid, rel, _sha256_file(ROOT / rel), inputs,
                         CORRECTED_GRAPH_SHA256, CORRECTED_CHECKPOINT, job, GEN_COMMAND,
                         mixed, failclass, ctol, lim])
    out = RES / "CORRECTED_325557_ARTIFACT_PROVENANCE.tsv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return out


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
    corr = load_correctness()
    small = load_small_correctness()
    env = load_environment()
    thesis_values = {r["ValueID"]: r for r in read_tsv("docs/thesis/thesis_values.tsv")}
    evidence = {r["ClaimID"]: r for r in read_tsv("docs/thesis/evidence_matrix.tsv")}

    # ---- validation: UNCHANGED headline performance (T1/T2/F1/F2/F7) --------
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
    note("clamp[email-EuAll,b8192]",
         sweeps["email-EuAll"][8192]["effective"] == 7393,
         f"effective={sweeps['email-EuAll'][8192]['effective']}")
    note("clamp[325557_3216152,b8192]",
         sweeps["325557_3216152"][8192]["effective"] == 6018,
         f"effective={sweeps['325557_3216152'][8192]['effective']}")
    note("phase_other_nonnegative", all(d["other"] >= 0 for d in ph.values()),
         "per-trial Other medians are non-negative")

    # ---- validation: CORRECTED ablation (T3/F4) vs the audited index --------
    for factor, vid in {"H": "ABL-H-325557-CORR", "W": "ABL-W-325557-CORR",
                        "A": "ABL-A-325557-CORR"}.items():
        got = round(abl["corrected"][FACTOR_KEY[factor]], 4)
        exp = round(float(thesis_values[vid]["Value"]), 4)
        note(f"thesis_index_ablation_corrected[{factor}]", got == exp,
             f"recomputed={got:.4f} indexed={exp:.4f}")
    for factor, vid in {"H": "ABL-H-SYNTH-CORR", "W": "ABL-W-SYNTH-CORR",
                        "A": "ABL-A-SYNTH-CORR"}.items():
        got = round(abl["mixed_geo"][FACTOR_KEY[factor]], 3)
        exp = round(float(thesis_values[vid]["Value"]), 3)
        note(f"thesis_index_ablation_aggregate[{factor}]", got == exp,
             f"recomputed(3dp)={got:.3f} indexed={exp:.3f}")
    note("thesis_index_ablation_base_full",
         round(abl["base_full"]["H0W0A0"], 2) == round(float(thesis_values["ABL-BASE-325557-CORR"]["Value"]), 2)
         and round(abl["base_full"]["H1W1A1"], 2) == round(float(thesis_values["ABL-FULL-325557-CORR"]["Value"]), 2),
         f"H0W0A0={abl['base_full']['H0W0A0']:.2f} H1W1A1={abl['base_full']['H1W1A1']:.2f}")
    note("ablation_evidence_status",
         evidence["C-ABL-SYNTH-CORR"]["Status"] == "SUPPORTED_WITH_LIMITATIONS",
         f"C-ABL-SYNTH-CORR={evidence['C-ABL-SYNTH-CORR']['Status']}")

    # ---- validation: CORRECTED memory feasibility (T4/F5) -------------------
    note("failure_not_zero", all((d["runtime"] is None) == (not d["success"]) for d in mem),
         "every failed boundary point has runtime=None (drawn as failure marker, not 0 s)")
    fail_classes = {(d["impl"], d["batch"]): d["fail_class"] for d in mem if d["fail_class"]}
    expected_fail = {("GPU_Opt_Pure", 8192): "cuda_oom",
                     ("GPU_Opt", 12288): "cgroup_host_oom_kill"}
    note("memory_failure_classes", fail_classes == expected_fail,
         f"CUDA OOM vs cgroup host-memory OOM kill kept distinct: {sorted(fail_classes.items())}")
    note("memory_trials_n1", all(d["trials"] == 1 for d in mem),
         "each feasibility boundary configuration is n=1 (targeted validation, not a sweep)")
    thesis_mem_ok = (
        thesis_values["MEM-PURE-B8192-OOM-CORR"]["Metric"] == "cuda_oom"
        and thesis_values["MEM-UM-B12288-OOMKILL-CORR"]["Metric"] == "host_cgroup_oom_kill"
        and thesis_values["MEM-UM-B10240-OK-CORR"]["Metric"] == "success"
        and thesis_values["MEM-CHUNK-B16384-OK-CORR"]["Metric"] == "success")
    note("thesis_index_memory_classes", thesis_mem_ok,
         "MEM-*-CORR index agrees: Pure b8192 cuda_oom; UM b12288 host_cgroup_oom_kill; UM b10240 / Chunked b16384 success")
    note("memory_evidence_status",
         evidence["C-MEM-FEAS-CORR"]["Status"] == "SUPPORTED_WITH_LIMITATIONS",
         f"C-MEM-FEAS-CORR={evidence['C-MEM-FEAS-CORR']['Status']}")

    # ---- validation: CORRECTED correctness (T5) -----------------------------
    comps = corr["comparisons"]
    # Tier A -- independent CPU reference (3 small graphs).
    note("small_correctness_all_pass",
         len(small) == 3 and all(
             s["missing"] == 0 and s["mismatch"] == 0 and s["tol_result"] == "PASS"
             for s in small),
         "3 independent Sequential-CPU-reference small-graph comparisons: "
         "MissingIndices/MismatchedElements 0, ToleranceResult PASS")
    note("small_correctness_not_byte_identical",
         all(not s["byte_identical"] for s in small),
         "independent CPU-reference comparisons ByteIdentical = No (per-vector SHA256 differ)")
    note("t5_total_rows_13", len(small) + len(comps) == 13,
         f"T5 = {len(small)} Tier A (independent CPU reference) + {len(comps)} Tier B "
         f"(cross-implementation) = {len(small) + len(comps)} rows")
    # Tier B -- corrected-325557 cross-implementation consistency (10 comparisons).
    note("correctness_all_mismatch_zero", all(c["mismatch"] == 0 for c in comps),
         "all 10 corrected-325557 comparisons have MismatchedElements = 0")
    note("correctness_not_byte_identical", all(not c["byte_identical"] for c in comps),
         "all 10 comparisons ByteIdentical = No (per-implementation SHA256 differ)")
    note("correctness_tolerance_pass", all(c["tol_result"] == "PASS" for c in comps),
         "all 10 comparisons ToleranceResult = PASS")
    note("correctness_vectors_finite",
         all(c["nonfinite"] == 0 and c["missing"] == 0 for c in comps),
         "no missing or non-finite values in any compared vector")
    note("thesis_index_correctness_mismatch0",
         int(float(thesis_values["CORR-ALLPAIRS-MISMATCH0-CORR"]["Value"])) == 0,
         f"CORR-ALLPAIRS-MISMATCH0-CORR={thesis_values['CORR-ALLPAIRS-MISMATCH0-CORR']['Value']}")
    note("thesis_index_correctness_maxbc",
         thesis_values["CORR-MAXBC-325557-CORR"]["Value"] == "272816",
         f"max BC agreement index={thesis_values['CORR-MAXBC-325557-CORR']['Value']}")
    note("correctness_evidence_status",
         evidence["C-CORR-CORR-325557"]["Status"] == "SUPPORTED_WITH_LIMITATIONS",
         f"C-CORR-CORR-325557={evidence['C-CORR-CORR-325557']['Status']}")
    # Historical preservation: the OLD malformed-input Core Fail must still exist
    # (as historical), never relabeled as the current judgment.
    note("historical_core_fail_preserved",
         evidence["C-CORR-STRESS"]["Status"] == "NOT_YET_SUPPORTED",
         f"old malformed stress C-CORR-STRESS retained historically={evidence['C-CORR-STRESS']['Status']}")

    # ---- validation: canonical scope / no build_miyabi / tracked-or-pending -
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
    untracked_inputs, pending_commit_inputs = [], []
    for rel in sorted(INPUTS_USED):
        repo_rel = str(Path(ROOT.name) / rel)
        proc = subprocess.run(
            ["git", "-C", str(ROOT.parent), "ls-files", "--error-unmatch", "--", repo_rel],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if proc.returncode != 0:
            if rel.startswith(PENDING_COMMIT_INPUT_PREFIXES):
                pending_commit_inputs.append(rel)     # W7.4 corrected inputs, pending commit
            else:
                untracked_inputs.append(rel)
    note("all_inputs_git_tracked_or_pending", not untracked_inputs,
         "all inputs tracked or pending-commit corrected inputs"
         if not untracked_inputs else f"unexpected untracked={untracked_inputs}")
    note("corrected_inputs_pending_commit", True,
         f"{len(pending_commit_inputs)} corrected-325557 inputs are canonical but pending commit (Gate W7.4)")

    # ---- figures: regenerate ONLY the selected set (default all) ------------
    fig_funcs = {
        "F1": lambda: fig_F1(mp), "F2": lambda: fig_F2(mp),
        "F3": lambda: fig_F3(sweeps, tuned_batch), "F4": lambda: fig_F4(abl),
        "F5": lambda: fig_F5(mem), "F6": lambda: fig_F6(ks), "F7": lambda: fig_F7(ph),
    }
    fig_out, regenerated = {}, []
    for fid, stem in FIG_STEMS.items():
        if fid in FIGURES_TO_GENERATE:
            fig_out[fid] = fig_funcs[fid]()
            regenerated.append(fid)
        else:   # reference the existing (byte-preserved) figure files
            fig_out[fid] = {"pdf": stem + ".pdf", "png": stem + ".png", "svg": stem + ".svg"}
    # ---- tables (all; T1/T2/T6 are deterministic no-ops, T3/T4/T5 updated) --
    tab_out = {
        "T1": table_T1(stats, cat), "T2": table_T2(mp), "T3": table_T3(abl),
        "T4": table_T4(mem), "T5": table_T5(corr, small), "T6": table_T6(env),
    }
    write_manifests_and_readmes(fig_out, tab_out, mp, corr, small)
    prov = write_corrected_provenance()

    # ---- report ----
    print("=== Gate W7.4.1 generation: validation ===")
    all_ok = True
    for name, ok, detail in VALIDATION:
        flag = "OK " if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{flag}] {name}: {detail}")
    print(f"\nFigures regenerated this run: {sorted(regenerated)} "
          f"(preserved: {sorted(set(FIG_STEMS) - set(regenerated))})")
    print(f"Tables written: {sorted(tab_out)}")
    print(f"Corrected-artifact provenance: {prov.relative_to(ROOT)}")
    print(f"Distinct canonical inputs used: {len(INPUTS_USED)} "
          f"({len(pending_commit_inputs)} pending-commit corrected inputs)")
    print("ALL_VALIDATION_OK" if all_ok else "VALIDATION_FAILURES_PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
