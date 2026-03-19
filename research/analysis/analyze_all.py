#!/usr/bin/env python3
"""
Miyabi BC Experiment -- Comprehensive Analysis & Visualization Script
=====================================================================
Master's thesis data analysis tool

Usage:
    python3 analyze_all.py [--result-dir <path>] [--output-dir <path>]

Default paths:
    --result-dir: ../build_miyabi  (directory containing TSV result files)
    --output-dir: ./figures        (output directory for figures)
"""

import argparse
import os
import sys
import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless environment (Miyabi login node)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
# Constants / Configuration
# ============================================================

# Canonical display label for each internal implementation key.
# All 7 implementations are listed with verbose, paper-ready English labels.
IMPL_LABELS = {
    # Uppercase variants (from TSV column names)
    'Sequential':      'Sequential (Single-thread CPU)',
    'OpenMP':          'OpenMP (72-core CPU Parallel)',
    'GPU':             'GPU-HBM3 (Single-stream)',
    'GPU_Stream':      'GPU-HBM3+Stream (Double-buffer)',
    'GPU_Managed':     'GPU-UVM (Naive)',
    'GPU_ReadMostly':  'GPU-UVM+ReadMostly (Method 1)',
    'GPU_Opt':         'GPU-UVM+Opt (Method 1+2)',
    # Lowercase aliases (from file-name based TSV keys)
    'sequential':      'Sequential (Single-thread CPU)',
    'openmp':          'OpenMP (72-core CPU Parallel)',
    'omp':             'OpenMP (72-core CPU Parallel)',
    'gpu':             'GPU-HBM3 (Single-stream)',
    'gpu_stream':      'GPU-HBM3+Stream (Double-buffer)',
    'gpu_managed':     'GPU-UVM (Naive)',
    'gpu_readmostly':  'GPU-UVM+ReadMostly (Method 1)',
    'gpu_opt':         'GPU-UVM+Opt (Method 1+2)',
}

# Ordered list of implementations (display preference: uppercase first).
IMPL_ORDER     = ['Sequential', 'OpenMP', 'GPU', 'GPU_Stream',
                  'GPU_Managed', 'GPU_ReadMostly', 'GPU_Opt']
IMPL_ORDER_ALT = ['sequential', 'openmp', 'omp', 'gpu', 'gpu_stream',
                  'gpu_managed', 'gpu_readmostly', 'gpu_opt']

# Colorblind-friendly palette (Wong, 2011).
_DISPLAY_COLORS = {
    'Sequential (Single-thread CPU)':  '#999999',
    'OpenMP (72-core CPU Parallel)':   '#E69F00',
    'GPU-HBM3 (Single-stream)':        '#0072B2',
    'GPU-HBM3+Stream (Double-buffer)': '#56B4E9',
    'GPU-UVM (Naive)':                 '#D55E00',
    'GPU-UVM+ReadMostly (Method 1)':   '#009E73',
    'GPU-UVM+Opt (Method 1+2)':        '#CC79A7',
}

def _color(key: str) -> str:
    """Return color for an implementation key or its display label."""
    label = IMPL_LABELS.get(key, key)
    return _DISPLAY_COLORS.get(label, _DISPLAY_COLORS.get(key, '#888888'))

# Keep a flat IMPL_COLORS dict for backward-compatible direct access.
IMPL_COLORS = {k: _color(k) for k in list(IMPL_LABELS.keys()) + list(_DISPLAY_COLORS.keys())}

_DISPLAY_MARKERS = {
    'Sequential (Single-thread CPU)':  'o',
    'OpenMP (72-core CPU Parallel)':   's',
    'GPU-HBM3 (Single-stream)':        '^',
    'GPU-HBM3+Stream (Double-buffer)': 'v',
    'GPU-UVM (Naive)':                 'D',
    'GPU-UVM+ReadMostly (Method 1)':   'P',
    'GPU-UVM+Opt (Method 1+2)':        '*',
}

def _marker(key: str) -> str:
    label = IMPL_LABELS.get(key, key)
    return _DISPLAY_MARKERS.get(label, _DISPLAY_MARKERS.get(key, 'o'))

IMPL_MARKERS = {k: _marker(k) for k in list(IMPL_LABELS.keys()) + list(_DISPLAY_MARKERS.keys())}

# Graph name -> node count (fallback when not extractable from TSV Graph column)
GRAPH_NODES = {
    'benchmark_7000':          7_000,
    'benchmark_7000_41459':    7_000,
    'benchmark_11023':        11_023,
    'benchmark_11023_62184':  11_023,
    'benchmark_85830':        85_830,
    'benchmark_85830.data':   85_830,
    'random':                 32_768,   # gen_graph default
    'random_32K':             32_768,
    '56438':                  56_438,
    '56438_300801':           56_438,
    '325557':                325_557,
    '325557_3216152':        325_557,
    'email-EuAll':           265_214,
    'amazon0302':            262_111,
    'web-Stanford':          281_903,
    'web-NotreDame':         325_729,
    'amazon0505':            410_236,
    'web-Google':            875_713,
    'roadNet-PA':          1_090_920,
    'roadNet-TX':          1_379_917,
    'roadNet-CA':          1_971_281,
}

# Human-readable display names for each graph dataset (paper-ready labels).
GRAPH_DISPLAY_NAMES = {
    'benchmark_7000':          'Synth-7K',
    'benchmark_7000_41459':    'Synth-7K',
    'benchmark_11023':         'Synth-11K',
    'benchmark_11023_62184':   'Synth-11K',
    'benchmark_85830':         'Synth-85K',
    'benchmark_85830.data':    'Synth-85K',
    'random':                  'Random-32K',
    'random_32K':              'Random-32K',
    '56438':                   'RealWorld-56K',
    '56438_300801':            'RealWorld-56K',
    '325557':                  'RealWorld-325K',
    '325557_3216152':          'RealWorld-325K',
    'email-EuAll':             'Email-EuAll-265K',
    'amazon0302':              'Amazon-262K',
    'web-Stanford':            'Web-Stanford-281K',
    'web-NotreDame':           'Web-NotreDame-325K',
    'amazon0505':              'Amazon-410K',
    'web-Google':              'Web-Google-875K',
    'roadNet-PA':              'Road-PA-1.09M',
    'roadNet-TX':              'Road-TX-1.38M',
    'roadNet-CA':              'Road-CA-1.97M',
}


def graph_display_name(graph_base: str) -> str:
    """Return paper-ready display name for a graph base name."""
    return GRAPH_DISPLAY_NAMES.get(graph_base, graph_base)

# Graph name -> edge count
GRAPH_EDGES = {
    'benchmark_7000_41459':      41_459,
    'benchmark_11023_62184':     62_184,
    'benchmark_85830.data':     241_000,
    'random':                   101_000,
    '56438_300801':             300_801,
    '325557_3216152':         3_216_152,
    'email-EuAll':              420_045,
    'amazon0302':               900_000,
    'web-Stanford':           2_312_497,
    'web-NotreDame':          1_497_134,
    'amazon0505':             2_439_437,
    'web-Google':             5_105_039,
    'roadNet-PA':             1_541_514,
    'roadNet-TX':             1_921_660,
    'roadNet-CA':             2_766_607,
}

BW_THEORY = {
    'HBM3_DtoD':         4020,
    'Pinned_HtoD':        900,
    'Pinned_DtoH':        900,
    'NVLink_C2C_Prefetch':900,
}

BW_LABELS = {
    'HBM3_DtoD':          'HBM3 D->D',
    'Pinned_HtoD':        'Pinned H->D',
    'Pinned_DtoH':        'Pinned D->H',
    'NVLink_C2C_Prefetch':'NVLink-C2C',
}

FIGURE_DPI = 300
FIGURE_SIZE_WIDE = (14, 6)
FIGURE_SIZE_SQUARE = (7, 5)

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'legend.fontsize':   9,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'figure.dpi':       FIGURE_DPI,
    'savefig.bbox':    'tight',
})


# ============================================================
# Utility functions
# ============================================================

def node_count_from_name(graph_name: str) -> int:
    """Return node count for a graph name (lookup in GRAPH_NODES table)."""
    base = Path(graph_name).name
    return GRAPH_NODES.get(base, 0)


def format_nodes(n: int) -> str:
    """Convert node count to a human-readable string (e.g. 325K, 1.97M)."""
    if n >= 1_000_000:
        return f'{n/1_000_000:.2f}M'
    if n >= 1_000:
        return f'{n//1_000}K'
    return str(n)


KNOWN_IMPLS = {'Sequential', 'OpenMP', 'GPU', 'GPU_Stream', 'GPU_Managed', 'GPU_ReadMostly',
               'GPU_Opt', 'sequential', 'openmp', 'omp', 'gpu', 'gpu_stream', 'gpu_managed',
               'gpu_readmostly', 'gpu_opt', 'mpi_gpu', 'brandes_mpi_runner'}


def load_tsv_skip_duplicate_headers(path: Path) -> pd.DataFrame:
    """Load a TSV that may contain duplicate header rows, stdout noise, or error messages.

    Missing-value tokens ('—', '-', '') are treated as NaN so that numeric
    conversion does not crash on them.
    """
    if not path.exists():
        return pd.DataFrame()
    rows = []
    header = None
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t')
            if header is None:
                # Detect header row
                if parts[0] in ('Implementation', 'BatchSize', 'Transfer_Type'):
                    header = parts
                continue
            # Skip duplicate header rows
            if parts == header:
                continue
            # Skip lines whose first field is not a known implementation (stdout noise guard)
            # For BatchSize/Transfer_Type columns, use numeric check instead
            first = parts[0] if parts else ''
            if header[0] == 'Implementation':
                if first not in KNOWN_IMPLS:
                    continue
            # Skip rows with fewer fields than header
            if len(parts) < len(header):
                continue
            rows.append(parts[:len(header)])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=header)
    df.columns = df.columns.str.strip()
    # Treat em-dash and standalone dash as missing values, but only in
    # string-type columns to avoid corrupting numeric text like '-5'.
    _missing = {'—', '-', ''}
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: np.nan if v in _missing else v)
    # Convert to numeric types
    for col in ['Time_sec', 'GTEPS', 'BatchSize',
                'BFS_sec', 'Back_sec', 'Size_GB',
                'Time_ms', 'Bandwidth_GBs', 'Theoretical_GBs', 'Ratio_pct']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Drop all-NaN rows
    df = df.dropna(how='all')
    return df


def add_graph_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add node count and display-name columns derived from the Graph column."""
    if df.empty or 'Graph' not in df.columns:
        return df
    df = df.copy()
    df['graph_base'] = df['Graph'].apply(lambda x: Path(x).name)
    df['nodes'] = df['graph_base'].map(lambda x: GRAPH_NODES.get(x, 0))
    df['graph_label'] = df['graph_base'].map(graph_display_name)
    return df.sort_values('nodes')


def get_impl_order(df: pd.DataFrame) -> list:
    """Return the canonical ordering of implementation names present in the data."""
    present = set(df['Implementation'].unique()) if not df.empty else set()
    order = []
    # Prefer uppercase variant first
    for impl in IMPL_ORDER:
        if impl in present:
            order.append(impl)
    # Fallback to lowercase
    if not order:
        for impl in IMPL_ORDER_ALT:
            if impl in present:
                order.append(impl)
    # Append any remaining unknown implementations
    for impl in present:
        if impl not in order:
            order.append(impl)
    return order


def _legend_outside(ax, n_series: int):
    """Place legend outside the plot area when there are 4 or more series."""
    if n_series >= 4:
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    else:
        ax.legend(loc='best')


def _annotate_bars(ax, bars, fmt='{:.2f}'):
    """Add value annotations on top of each bar."""
    for bar in bars:
        height = bar.get_height()
        if np.isfinite(height) and height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                    fmt.format(height), ha='center', va='bottom',
                    fontsize=7, clip_on=True)


def _savefig(fig, out_path: Path):
    """Save figure as both PDF and PNG with tight bounding box."""
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"[OK] {out_path}")


# ============================================================
# Plot functions
# ============================================================

def plot_execution_time(df: pd.DataFrame, out_dir: Path):
    """Execution Time Comparison Across All Datasets (line chart, log-scale Y)."""
    if df.empty:
        print("[SKIP] Execution time graph: no data")
        return

    df = add_graph_metadata(df)

    # Exclude non-positive values before log scale
    df = df[df['Time_sec'] > 0].copy()
    if df.empty:
        print("[SKIP] Execution time graph: all values <= 0")
        return

    graphs_sorted = sorted(df['graph_base'].unique(), key=lambda g: GRAPH_NODES.get(g, 0))
    impls = get_impl_order(df)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    for impl in impls:
        sub = df[df['Implementation'] == impl].sort_values('nodes')
        if sub.empty:
            continue
        xs = list(range(len(sub)))
        ys = sub['Time_sec'].values
        ax.plot(xs, ys,
                label=IMPL_LABELS.get(impl, impl),
                color=IMPL_COLORS.get(impl, '#888888'),
                marker=IMPL_MARKERS.get(impl, 'o'),
                linewidth=1.8, markersize=6)

    x_labels = [graph_display_name(g) for g in graphs_sorted]
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=40, ha='right')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Vertices')
    ax.set_ylabel('Execution Time (seconds) [log scale]')
    ax.set_title('Execution Time Comparison Across All Datasets')
    _legend_outside(ax, len(impls))
    ax.grid(True, which='both', alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()
    _savefig(fig, out_dir / 'execution_time_comparison.pdf')
    plt.close(fig)


def plot_speedup(df: pd.DataFrame, out_dir: Path, baseline_impl: str = 'sequential'):
    """Speedup over a baseline implementation vs. graph size."""
    if df.empty:
        print("[SKIP] Speedup graph: no data")
        return

    df = add_graph_metadata(df)
    base_df = df[df['Implementation'] == baseline_impl][['graph_base', 'Time_sec']].rename(
        columns={'Time_sec': 'base_time'}
    )
    if base_df.empty:
        print(f"[SKIP] Speedup graph: no data for baseline '{baseline_impl}'")
        return

    merged = df.merge(base_df, on='graph_base')
    # Guard against division-by-zero
    merged['base_time'] = merged['base_time'].replace(0, np.nan)
    merged['speedup'] = merged['base_time'] / merged['Time_sec']
    merged = merged.sort_values('nodes')

    graphs_sorted = sorted(merged['graph_base'].unique(), key=lambda g: GRAPH_NODES.get(g, 0))
    impls = get_impl_order(df)

    baseline_label = IMPL_LABELS.get(baseline_impl, baseline_impl)
    # Choose file-name suffix based on the display label of the baseline
    if 'Sequential' in baseline_label:
        suffix = 'Sequential'
        title = 'Speedup over Sequential (Single-thread CPU)'
        ylabel = 'Speedup (x)'
    elif 'OpenMP' in baseline_label:
        suffix = 'OpenMP'
        title = 'Speedup over OpenMP (72-core Parallel)'
        ylabel = 'Speedup (x)'
    else:
        suffix = baseline_impl
        title = f'Speedup over {baseline_label}'
        ylabel = 'Speedup (x)'

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    plotted = 0
    for impl in impls:
        if impl == baseline_impl:
            continue
        sub = merged[merged['Implementation'] == impl].sort_values('nodes')
        if sub.empty:
            continue
        xs = list(range(len(sub)))
        ys = sub['speedup'].values

        ax.plot(xs, ys,
                label=IMPL_LABELS.get(impl, impl),
                color=IMPL_COLORS.get(impl, '#888888'),
                marker=IMPL_MARKERS.get(impl, 'o'),
                linewidth=1.8, markersize=6)
        plotted += 1

    x_labels = [graph_display_name(g) for g in graphs_sorted]
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=40, ha='right')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1,
               label=f'{baseline_label} (baseline = 1×)')
    ax.set_xlabel('Number of Vertices')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _legend_outside(ax, plotted + 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _savefig(fig, out_dir / f'speedup_vs_{suffix}.pdf')
    plt.close(fig)


def plot_gteps(df: pd.DataFrame, out_dir: Path):
    """Throughput Comparison (GTEPS) Across All Datasets (line chart)."""
    if df.empty:
        print("[SKIP] GTEPS graph: no data")
        return

    df = add_graph_metadata(df)
    # Exclude non-positive values before plotting
    df = df[df['GTEPS'] > 0].copy()
    if df.empty:
        print("[SKIP] GTEPS graph: all values <= 0")
        return

    graphs_sorted = sorted(df['graph_base'].unique(), key=lambda g: GRAPH_NODES.get(g, 0))
    impls = get_impl_order(df)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    for impl in impls:
        sub = df[df['Implementation'] == impl].sort_values('nodes')
        if sub.empty:
            continue
        xs = list(range(len(sub)))
        ys = sub['GTEPS'].values

        ax.plot(xs, ys,
                label=IMPL_LABELS.get(impl, impl),
                color=IMPL_COLORS.get(impl, '#888888'),
                marker=IMPL_MARKERS.get(impl, 'o'),
                linewidth=1.8, markersize=6)

    x_labels = [graph_display_name(g) for g in graphs_sorted]
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=40, ha='right')
    ax.set_xlabel('Number of Vertices')
    ax.set_ylabel('Throughput (GTEPS)')
    ax.set_title('Throughput Comparison (GTEPS) Across All Datasets')
    _legend_outside(ax, len(impls))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _savefig(fig, out_dir / 'gteps_comparison.pdf')
    plt.close(fig)


def plot_phase2_comparison(df: pd.DataFrame, out_dir: Path):
    """Execution time bar-chart comparison for GPU-family implementations."""
    if df.empty:
        print("[SKIP] Phase2 graph: no data")
        return

    df = add_graph_metadata(df)
    # All 7 implementation keys (both case variants)
    all_impls = ['GPU', 'GPU_Stream', 'GPU_Managed', 'GPU_ReadMostly', 'GPU_Opt',
                 'gpu', 'gpu_stream', 'gpu_managed', 'gpu_readmostly', 'gpu_opt']
    impls_present = [i for i in all_impls if i in df['Implementation'].values]
    if not impls_present:
        print("[SKIP] Phase2 graph: no GPU implementation data")
        return

    graphs_sorted = sorted(df['graph_base'].unique(), key=lambda g: GRAPH_NODES.get(g, 0))
    n_graphs = len(graphs_sorted)
    n_impls  = len(impls_present)
    x = np.arange(n_graphs)
    width = 0.8 / max(n_impls, 1)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    for i, impl in enumerate(impls_present):
        sub = df[df['Implementation'] == impl]
        times = []
        for g in graphs_sorted:
            vals = sub[sub['graph_base'] == g]['Time_sec'].values
            times.append(vals[0] if len(vals) > 0 and vals[0] > 0 else np.nan)
        offset = (i - n_impls / 2 + 0.5) * width
        bars = ax.bar(x + offset, times, width,
                      label=IMPL_LABELS.get(impl, impl),
                      color=IMPL_COLORS.get(impl, '#888888'), alpha=0.85)
        _annotate_bars(ax, bars, fmt='{:.2f}')

    x_labels = [graph_display_name(g) for g in graphs_sorted]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=40, ha='right')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Vertices')
    ax.set_ylabel('Execution Time (seconds) [log scale]')
    ax.set_title('Execution Time Comparison Across All Datasets')
    _legend_outside(ax, n_impls)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    _savefig(fig, out_dir / 'phase2_memory_comparison.pdf')
    plt.close(fig)


def plot_bandwidth(df: pd.DataFrame, out_dir: Path):
    """GH200 Memory Bandwidth: HBM3 vs. NVLink-C2C (bar chart)."""
    if df.empty:
        print("[SKIP] Bandwidth graph: no data")
        return

    # Use data at the largest buffer size (most stable measurement)
    max_size = df['Size_GB'].max()
    df_max = df[df['Size_GB'] == max_size].copy()

    types = [t for t in ['HBM3_DtoD', 'Pinned_HtoD', 'Pinned_DtoH', 'NVLink_C2C_Prefetch']
             if t in df_max['Transfer_Type'].values]
    if not types:
        print("[SKIP] Bandwidth graph: no transfer type data")
        return

    measured    = [df_max[df_max['Transfer_Type'] == t]['Bandwidth_GBs'].values[0] for t in types]
    theoretical = [BW_THEORY.get(t, 0) for t in types]
    labels      = [BW_LABELS.get(t, t) for t in types]

    x = np.arange(len(types))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)
    bars1 = ax.bar(x - width/2, measured,    width, label='Measured',
                   color='#1976D2', alpha=0.9)
    bars2 = ax.bar(x + width/2, theoretical, width, label='Theoretical Peak',
                   color='#90CAF9', alpha=0.7, hatch='///')

    # Utilization percentage above measured bars
    for rect, meas, theo in zip(bars1, measured, theoretical):
        if theo > 0:
            ratio = meas / theo * 100
            ax.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 10,
                    f'{ratio:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Memory Bandwidth (GB/s)')
    ax.set_title(f'GH200 Memory Bandwidth: HBM3 vs. NVLink-C2C\n(buffer = {max_size:.1f} GB)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    _savefig(fig, out_dir / 'bandwidth_comparison.pdf')
    plt.close(fig)


def plot_bandwidth_by_size(df: pd.DataFrame, out_dir: Path):
    """Bandwidth vs. buffer size (line chart)."""
    if df.empty:
        print("[SKIP] Bandwidth-size graph: no data")
        return

    types = [t for t in ['HBM3_DtoD', 'Pinned_HtoD', 'Pinned_DtoH', 'NVLink_C2C_Prefetch']
             if t in df['Transfer_Type'].values]
    if not types:
        return

    colors = {'HBM3_DtoD': '#D32F2F', 'Pinned_HtoD': '#1976D2',
              'Pinned_DtoH': '#388E3C', 'NVLink_C2C_Prefetch': '#F57C00'}

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)

    for t in types:
        sub = df[df['Transfer_Type'] == t].sort_values('Size_GB')
        sub = sub[sub['Bandwidth_GBs'] > 0]
        ax.plot(sub['Size_GB'], sub['Bandwidth_GBs'],
                label=BW_LABELS.get(t, t),
                color=colors.get(t, '#888888'),
                marker='o', linewidth=1.8, markersize=6)
        theo = BW_THEORY.get(t, 0)
        if theo > 0:
            ax.axhline(theo, color=colors.get(t, '#888888'),
                       linestyle=':', alpha=0.4, linewidth=1)

    ax.set_xscale('log')
    ax.set_xlabel('Buffer Size (GB) [log scale]')
    ax.set_ylabel('Memory Bandwidth (GB/s)')
    ax.set_title('Memory Bandwidth vs. Buffer Size\n(dotted lines = theoretical peak)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    _savefig(fig, out_dir / 'bandwidth_vs_bufsize.pdf')
    plt.close(fig)


def plot_batchsize_sweep(df: pd.DataFrame, out_dir: Path):
    """Batch Size Sensitivity (line chart for each graph)."""
    if df.empty:
        print("[SKIP] Batch size graph: no data")
        return

    graphs = df['Graph'].apply(lambda x: Path(x).name).unique() if 'Graph' in df.columns else []

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel in zip(
        axes,
        ['Time_sec', 'GTEPS'],
        ['Execution Time (seconds)', 'Throughput (GTEPS)']
    ):
        for g in graphs:
            sub = df[df['Graph'].apply(lambda x: Path(x).name) == g]
            if sub.empty or 'BatchSize' not in sub.columns:
                continue
            sub = sub.sort_values('BatchSize')
            sub = sub[sub[metric] > 0] if metric == 'Time_sec' else sub
            label = graph_display_name(g)
            ax.plot(sub['BatchSize'], sub[metric], marker='o', label=label)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs. Batch Size')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Batch Size Sensitivity', fontsize=13)
    fig.tight_layout()
    _savefig(fig, out_dir / 'batchsize_sensitivity.pdf')
    plt.close(fig)


# ============================================================
# ============================================================
# Table generation
# ============================================================
# ============================================================

def make_exec_time_table(df: pd.DataFrame, tables_dir: Path):
    """Execution time comparison table (TSV + LaTeX)."""
    if df.empty:
        print("[SKIP] Execution time table: no data")
        return

    df = add_graph_metadata(df)
    pivot = df.pivot_table(index='graph_base', columns='Implementation',
                           values='Time_sec', aggfunc='mean')
    pivot = pivot.reindex(columns=[c for c in IMPL_ORDER if c in pivot.columns])
    pivot.index = [f"{g}\n({format_nodes(GRAPH_NODES.get(g,0))} nodes)" for g in pivot.index]

    # TSV
    tsv_path = tables_dir / 'exec_time_table.tsv'
    pivot.to_csv(tsv_path, sep='\t', float_format='%.4f')
    print(f"[OK] {tsv_path}")

    # LaTeX
    latex_path = tables_dir / 'exec_time_table.tex'
    col_labels = [IMPL_LABELS.get(c, c) for c in pivot.columns]
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Execution Time (sec) by Implementation and Graph Size}\n")
        f.write("\\label{tab:exec_time}\n")
        ncols = len(pivot.columns) + 1
        f.write(f"\\begin{{tabular}}{{l{'r' * len(pivot.columns)}}}\n")
        f.write("\\hline\n")
        f.write("Graph & " + " & ".join(col_labels) + " \\\\\n")
        f.write("\\hline\n")
        for idx, row in pivot.iterrows():
            cells = [f"{v:.4f}" if not pd.isna(v) else "--" for v in row.values]
            f.write(f"{idx.replace(chr(10), ' ')} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"[OK] {latex_path}")


def make_speedup_table(df: pd.DataFrame, tables_dir: Path,
                       baseline_impl: str = 'sequential'):
    """Speedup ratio table."""
    if df.empty:
        print("[SKIP] Speedup table: no data")
        return

    df = add_graph_metadata(df)
    base_df = df[df['Implementation'] == baseline_impl][['graph_base', 'Time_sec']].rename(
        columns={'Time_sec': 'base_time'}
    )
    if base_df.empty:
        print(f"[SKIP] Speedup table: no baseline data for '{baseline_impl}'")
        return

    merged = df.merge(base_df, on='graph_base')
    # Guard against division-by-zero
    merged['base_time'] = merged['base_time'].replace(0, np.nan)
    merged['speedup'] = merged['base_time'] / merged['Time_sec']
    pivot = merged.pivot_table(index='graph_base', columns='Implementation',
                               values='speedup', aggfunc='mean')
    pivot = pivot.reindex(columns=[c for c in IMPL_ORDER if c in pivot.columns])
    pivot.index = [f"{g} ({format_nodes(GRAPH_NODES.get(g,0))})" for g in pivot.index]

    tsv_path = tables_dir / f'speedup_table_vs_{baseline_impl}.tsv'
    pivot.to_csv(tsv_path, sep='\t', float_format='%.2f')
    print(f"[OK] {tsv_path}")

    latex_path = tables_dir / f'speedup_table_vs_{baseline_impl}.tex'
    col_labels = [IMPL_LABELS.get(c, c) for c in pivot.columns]
    baseline_label = IMPL_LABELS.get(baseline_impl, baseline_impl)
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Speedup over {baseline_label}}}\n")
        f.write(f"\\label{{tab:speedup_vs_{baseline_impl}}}\n")
        f.write(f"\\begin{{tabular}}{{l{'r' * len(pivot.columns)}}}\n")
        f.write("\\hline\n")
        f.write("Graph & " + " & ".join(col_labels) + " \\\\\n")
        f.write("\\hline\n")
        for idx, row in pivot.iterrows():
            cells = [f"{v:.2f}$\\times$" if not pd.isna(v) else "--" for v in row.values]
            f.write(f"{idx} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"[OK] {latex_path}")


def make_gteps_table(df: pd.DataFrame, tables_dir: Path):
    """GTEPS table."""
    if df.empty:
        print("[SKIP] GTEPS table: no data")
        return

    df = add_graph_metadata(df)
    pivot = df.pivot_table(index='graph_base', columns='Implementation',
                           values='GTEPS', aggfunc='mean')
    pivot = pivot.reindex(columns=[c for c in IMPL_ORDER if c in pivot.columns])
    pivot.index = [f"{g} ({format_nodes(GRAPH_NODES.get(g,0))})" for g in pivot.index]

    tsv_path = tables_dir / 'gteps_table.tsv'
    pivot.to_csv(tsv_path, sep='\t', float_format='%.4f')
    print(f"[OK] {tsv_path}")

    latex_path = tables_dir / 'gteps_table.tex'
    col_labels = [IMPL_LABELS.get(c, c) for c in pivot.columns]
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Throughput (GTEPS) by Implementation and Graph Size}\n")
        f.write("\\label{tab:gteps}\n")
        f.write(f"\\begin{{tabular}}{{l{'r' * len(pivot.columns)}}}\n")
        f.write("\\hline\n")
        f.write("Graph & " + " & ".join(col_labels) + " \\\\\n")
        f.write("\\hline\n")
        for idx, row in pivot.iterrows():
            cells = [f"{v:.4f}" if not pd.isna(v) else "--" for v in row.values]
            f.write(f"{idx} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"[OK] {latex_path}")


def make_bandwidth_table(df: pd.DataFrame, tables_dir: Path):
    """Bandwidth measurement table (all buffer sizes)."""
    if df.empty:
        print("[SKIP] Bandwidth table: no data")
        return

    pivot = df.pivot_table(index='Transfer_Type', columns='Size_GB',
                           values='Bandwidth_GBs', aggfunc='mean')
    pivot.index = [BW_LABELS.get(t, t) for t in pivot.index]

    tsv_path = tables_dir / 'bandwidth_table.tsv'
    pivot.to_csv(tsv_path, sep='\t', float_format='%.1f')
    print(f"[OK] {tsv_path}")

    # Theoretical peak comparison table (largest buffer size)
    max_size = df['Size_GB'].max()
    df_max = df[df['Size_GB'] == max_size].copy()
    df_max['label'] = df_max['Transfer_Type'].map(BW_LABELS)
    df_max = df_max.set_index('label')

    latex_path = tables_dir / 'bandwidth_table.tex'
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{GH200 Memory Bandwidth ({max_size:.1f} GB buffer)}}\n")
        f.write("\\label{tab:bandwidth}\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\hline\n")
        f.write("Transfer Type & Measured (GB/s) & Theoretical (GB/s) & Utilization (\\%) \\\\\n")
        f.write("\\hline\n")
        for idx, row in df_max.iterrows():
            f.write(f"{idx} & {row['Bandwidth_GBs']:.1f} & "
                    f"{row['Theoretical_GBs']:.0f} & {row['Ratio_pct']:.1f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"[OK] {latex_path}")


# ============================================================
# Thesis adequacy assessment
# ============================================================

def assess_thesis_adequacy(baseline_df: pd.DataFrame,
                            phase2_df: pd.DataFrame,
                            bw_df: pd.DataFrame):
    """Evaluate whether data is sufficient for a master's thesis and print key discussion points."""
    print("\n" + "=" * 60)
    print("  Thesis Adequacy Assessment Report")
    print("=" * 60)

    score = 0
    max_score = 10
    issues = []
    strengths = []

    # --- 1. Basic data volume ---
    if not baseline_df.empty:
        n_graphs = baseline_df['Graph'].nunique() if 'Graph' in baseline_df.columns else 0
        n_impls = baseline_df['Implementation'].nunique() if 'Implementation' in baseline_df.columns else 0
        if n_graphs >= 5:
            score += 2
            strengths.append(f"+ {n_graphs} graphs x {n_impls} implementations -- sufficient data points")
        elif n_graphs >= 3:
            score += 1
            issues.append(f"~ Few graphs ({n_graphs}). 5 or more recommended")
        else:
            issues.append(f"x Insufficient graph count ({n_graphs} graphs)")
    else:
        issues.append("x No baseline data -- please re-run experiments")

    # --- 2. Speedup significance ---
    if not baseline_df.empty and 'Implementation' in baseline_df.columns:
        df_meta = add_graph_metadata(baseline_df)
        seq_df = df_meta[df_meta['Implementation'].isin(['sequential', 'Sequential'])]
        gpu_opt_df = df_meta[df_meta['Implementation'].isin(['gpu_opt', 'GPU_Opt'])]
        if not seq_df.empty and not gpu_opt_df.empty:
            merged = seq_df[['graph_base', 'Time_sec']].merge(
                gpu_opt_df[['graph_base', 'Time_sec']], on='graph_base', suffixes=('_seq', '_gpu')
            )
            if not merged.empty:
                max_speedup = (merged['Time_sec_seq'] / merged['Time_sec_gpu']).max()
                if max_speedup > 100:
                    score += 3
                    strengths.append(f"+ Max speedup {max_speedup:.0f}x -- sufficient for thesis claim")
                elif max_speedup > 10:
                    score += 2
                    strengths.append(f"+ Max speedup {max_speedup:.1f}x -- significant performance gain")
                else:
                    score += 1
                    issues.append(f"~ Speedup only {max_speedup:.1f}x -- discussion required")
        else:
            issues.append("~ Missing sequential or gpu_opt data (cannot compute speedup)")

    # --- 3. NVLink-C2C effectiveness evaluation ---
    if not baseline_df.empty and not phase2_df.empty:
        score += 1
        strengths.append("+ gpu vs gpu_managed comparison available (NVLink-C2C evaluable)")
    elif not phase2_df.empty:
        score += 1
        strengths.append("+ Phase2 data available (Unified Memory evaluable)")
    else:
        issues.append("~ No Phase2 data -- cannot demonstrate gpu_managed effect")

    # --- 4. Bandwidth data ---
    if not bw_df.empty:
        score += 1
        max_hbm3 = bw_df[bw_df['Transfer_Type'] == 'HBM3_DtoD']['Bandwidth_GBs'].max() if 'Transfer_Type' in bw_df.columns else 0
        nvlink = bw_df[bw_df['Transfer_Type'] == 'NVLink_C2C_Prefetch']['Bandwidth_GBs'].max() if 'Transfer_Type' in bw_df.columns else 0
        strengths.append(f"+ Bandwidth data available (HBM3: {max_hbm3:.0f} GB/s, NVLink-C2C: {nvlink:.0f} GB/s)")
    else:
        issues.append("~ No bandwidth data")

    # --- 5. Graph size scalability ---
    if not baseline_df.empty:
        df_meta = add_graph_metadata(baseline_df)
        sizes = df_meta['nodes'].unique()
        sizes = [s for s in sizes if s > 0]
        if len(sizes) >= 4:
            min_n, max_n = min(sizes), max(sizes)
            ratio = max_n / min_n if min_n > 0 else 0
            if ratio > 100:
                score += 2
                strengths.append(f"+ Scale range {ratio:.0f}x ({format_nodes(int(min_n))} to {format_nodes(int(max_n))}) -- scalability evaluable")
            else:
                score += 1
                issues.append(f"~ Scale range {ratio:.0f}x -- adding larger graphs recommended")
        else:
            issues.append(f"~ Too few graph sizes ({len(sizes)} variants)")

    # --- Assessment result ---
    print(f"\nTotal score: {score}/{max_score}")
    print()

    if score >= 8:
        verdict = "Excellent: Data is sufficient for a master's thesis"
    elif score >= 5:
        verdict = "Good: Largely sufficient, but some reinforcement is desirable"
    else:
        verdict = "Insufficient: Additional experiments are needed"

    print(f"Verdict: {verdict}\n")

    if strengths:
        print("[Strengths]")
        for s in strengths:
            print(f"  {s}")
    if issues:
        print("\n[Issues / Recommendations]")
        for i in issues:
            print(f"  {i}")

    print("\n[Key discussion points for the thesis]")
    print("  1. GH200 NVLink-C2C (900 GB/s): ~19% utilization for LPDDR5X->GPU transfers")
    print("     -> Discuss performance gap between small graphs (fit in HBM3) and large graphs (LPDDR5X overflow)")
    print("  2. gpu_opt = 2-stream DP + cudaMemsetAsync + ReadMostly combined")
    print("     -> Analyze which optimization contributed most using phase_timing.log")
    print("  3. Compare trends: Road network (sparse, large) vs Web graph (dense, medium)")
    print("  4. CPU parallelization limit: compare sequential vs OpenMP")
    print("  5. Compare GTEPS with other BC implementations in the literature (Graph500 TEPS etc.)")

    print("\n" + "=" * 60)
    return score


# ============================================================
# ============================================================
# MPI scaling analysis
# ============================================================
# ============================================================

def load_scaling_tsv(path: Path) -> pd.DataFrame:
    """Load an MPI scaling TSV.
    Format: Impl  Graph  Ranks  SHARP  Time_sec  GTEPS
    """
    if not path.exists():
        print(f"  [SKIP] {path} does not exist")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, sep='\t', comment='#')
        df.columns = df.columns.str.strip()
        if 'Ranks' in df.columns:
            df['Ranks'] = pd.to_numeric(df['Ranks'], errors='coerce')
        if 'Time_sec' in df.columns:
            df['Time_sec'] = pd.to_numeric(df['Time_sec'], errors='coerce')
        if 'GTEPS' in df.columns:
            df['GTEPS'] = pd.to_numeric(df['GTEPS'], errors='coerce')
        if 'SHARP' in df.columns:
            df['SHARP'] = pd.to_numeric(df['SHARP'], errors='coerce').astype(int)
        df = df.dropna(subset=['Ranks', 'Time_sec'])
        print(f"  Loaded: {path.name} -> {len(df)} rows")
        return df
    except Exception as e:
        print(f"  [ERROR] {path}: {e}")
        return pd.DataFrame()


def plot_strong_scaling(df: pd.DataFrame, out_dir: Path):
    """Strong scaling: number of ranks vs. execution time (SHARP=0/1 comparison)."""
    if df.empty:
        print("  [SKIP] No strong scaling data")
        return

    impls = df['Impl'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Strong Scaling -- Betweenness Centrality (MPI)', fontsize=14)

    # Left: execution time vs. number of ranks
    ax = axes[0]
    colors_by_sharp = {0: '#2196F3', 1: '#F44336'}
    markers_by_sharp = {0: 'o', 1: 's'}
    for impl in impls:
        for sharp in sorted(df['SHARP'].unique()):
            sub = df[(df['Impl'] == impl) & (df['SHARP'] == sharp)].sort_values('Ranks')
            if sub.empty:
                continue
            label = f"{impl} (SHARP={'ON' if sharp else 'OFF'})"
            ax.plot(sub['Ranks'], sub['Time_sec'],
                    marker=markers_by_sharp.get(sharp, 'o'),
                    color=colors_by_sharp.get(sharp, '#888888'),
                    linestyle='-' if sharp == 1 else '--',
                    label=label)
    # Ideal scaling line
    ranks_1 = df[df['Ranks'] == 1]['Time_sec']
    if not ranks_1.empty:
        t1 = ranks_1.mean()
        rank_range = sorted(df['Ranks'].unique())
        ax.plot(rank_range, [t1 / r for r in rank_range],
                'k:', linewidth=1.0, label='Ideal (T1/P)')
    ax.set_xlabel('Number of MPI Ranks (Nodes)', fontsize=11)
    ax.set_ylabel('Execution Time [s]', fontsize=11)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_title('Execution Time vs. Number of Ranks')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: parallel efficiency vs. number of ranks
    ax2 = axes[1]
    for impl in impls:
        for sharp in sorted(df['SHARP'].unique()):
            sub = df[(df['Impl'] == impl) & (df['SHARP'] == sharp)].sort_values('Ranks')
            if sub.empty or 1 not in sub['Ranks'].values:
                continue
            t1 = sub[sub['Ranks'] == 1]['Time_sec'].values[0]
            eff = [t1 / (row['Ranks'] * row['Time_sec']) for _, row in sub.iterrows()]
            label = f"{impl} (SHARP={'ON' if sharp else 'OFF'})"
            ax2.plot(sub['Ranks'], eff,
                     marker=markers_by_sharp.get(sharp, 'o'),
                     color=colors_by_sharp.get(sharp, '#888888'),
                     linestyle='-' if sharp == 1 else '--',
                     label=label)
    ax2.axhline(y=1.0, color='k', linestyle=':', linewidth=1.0, label='Ideal Efficiency')
    ax2.set_xlabel('Number of MPI Ranks (Nodes)', fontsize=11)
    ax2.set_ylabel('Parallel Efficiency E = T1 / (P x Tp)', fontsize=11)
    ax2.set_xscale('log', base=2)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.set_ylim(0, 1.3)
    ax2.set_title('Parallel Efficiency vs. Number of Ranks')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        out_path = out_dir / f'scaling_strong.{ext}'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"  -> {out_path}")
    plt.close()


def plot_weak_scaling(df: pd.DataFrame, out_dir: Path):
    """Weak scaling: number of ranks vs. execution time (constant graph size per rank)."""
    if df.empty:
        print("  [SKIP] No weak scaling data")
        return

    impls = df['Impl'].unique()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors_by_sharp = {0: '#2196F3', 1: '#F44336'}
    markers_by_sharp = {0: 'o', 1: 's'}

    for impl in impls:
        for sharp in sorted(df['SHARP'].unique()):
            sub = df[(df['Impl'] == impl) & (df['SHARP'] == sharp)].sort_values('Ranks')
            if sub.empty:
                continue
            label = f"{impl} (SHARP={'ON' if sharp else 'OFF'})"
            ax.plot(sub['Ranks'], sub['Time_sec'],
                    marker=markers_by_sharp.get(sharp, 'o'),
                    color=colors_by_sharp.get(sharp, '#888888'),
                    linestyle='-' if sharp == 1 else '--',
                    label=label)

    # Ideal weak scaling: constant time
    ranks_1 = df[df['Ranks'] == 1]['Time_sec']
    if not ranks_1.empty:
        t1 = ranks_1.mean()
        ax.axhline(y=t1, color='k', linestyle=':', linewidth=1.0, label='Ideal (constant)')

    ax.set_xlabel('Number of MPI Ranks (Nodes)', fontsize=12)
    ax.set_ylabel('Execution Time [s]', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_title('Weak Scaling -- Betweenness Centrality (MPI)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        out_path = out_dir / f'scaling_weak.{ext}'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"  -> {out_path}")
    plt.close()


def plot_sharp_comparison(strong_df: pd.DataFrame, out_dir: Path):
    """Bar chart showing the effect of SHARP AllReduce (SHARP=0 vs SHARP=1)."""
    if strong_df.empty:
        print("  [SKIP] No SHARP comparison data")
        return
    if 'SHARP' not in strong_df.columns or strong_df['SHARP'].nunique() < 2:
        print("  [SKIP] Both SHARP=0 and SHARP=1 data are required")
        return

    impls = strong_df['Impl'].unique()
    ranks = sorted(strong_df['Ranks'].unique())
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(ranks))
    bar_width = 0.35
    colors = {'off': '#2196F3', 'on': '#F44336'}

    for i, sharp in enumerate([0, 1]):
        times = []
        for r in ranks:
            sub = strong_df[(strong_df['SHARP'] == sharp) & (strong_df['Ranks'] == r)]
            times.append(sub['Time_sec'].mean() if not sub.empty else float('nan'))
        offset = (i - 0.5) * bar_width
        bars = ax.bar(x + offset, times, bar_width,
                      label=f"SHARP={'ON' if sharp else 'OFF'}",
                      color=colors['on' if sharp else 'off'],
                      alpha=0.8)
        for bar, t in zip(bars, times):
            if not np.isnan(t):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{t:.2f}s', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Number of MPI Ranks', fontsize=12)
    ax.set_ylabel('Execution Time [s]', fontsize=12)
    ax.set_title('SHARP AllReduce Effect -- Strong Scaling Comparison', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        out_path = out_dir / f'sharp_comparison.{ext}'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"  -> {out_path}")
    plt.close()


def make_scaling_table(strong_df: pd.DataFrame, tables_dir: Path):
    """Strong scaling: parallel efficiency TSV + LaTeX table."""
    if strong_df.empty:
        print("  [SKIP] Scaling table: no data")
        return

    rows = []
    for impl in strong_df['Impl'].unique():
        for sharp in sorted(strong_df['SHARP'].unique()):
            sub = strong_df[(strong_df['Impl'] == impl) & (strong_df['SHARP'] == sharp)].sort_values('Ranks')
            if sub.empty:
                continue
            t1_rows = sub[sub['Ranks'] == 1]
            if t1_rows.empty:
                continue
            t1 = t1_rows['Time_sec'].values[0]
            for _, row in sub.iterrows():
                speedup = t1 / row['Time_sec']
                efficiency = speedup / row['Ranks']
                rows.append({
                    'Impl': impl,
                    'SHARP': int(row['SHARP']),
                    'Ranks': int(row['Ranks']),
                    'Time_sec': row['Time_sec'],
                    'Speedup': speedup,
                    'Efficiency': efficiency,
                    'GTEPS': row.get('GTEPS', float('nan')),
                })

    if not rows:
        print("  [SKIP] Scaling table: no single-rank data")
        return

    result_df = pd.DataFrame(rows)

    tsv_path = tables_dir / 'scaling_table.tsv'
    result_df.to_csv(tsv_path, sep='\t', index=False, float_format='%.4f')
    print(f"  -> {tsv_path}")

    # LaTeX table
    tex_lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{MPI Strong Scaling: Parallel Efficiency}',
        r'\label{tab:scaling_efficiency}',
        r'\begin{tabular}{llrrrr}',
        r'\hline',
        r'Implementation & SHARP & Ranks & Time [s] & Speedup & Parallel Efficiency \\',
        r'\hline',
    ]
    for _, row in result_df.iterrows():
        sharp_str = 'ON' if row['SHARP'] else 'OFF'
        tex_lines.append(
            f"{row['Impl']} & {sharp_str} & {int(row['Ranks'])} & "
            f"{row['Time_sec']:.3f} & {row['Speedup']:.2f} & {row['Efficiency']:.2f} \\\\"
        )
    tex_lines += [r'\hline', r'\end{tabular}', r'\end{table}']

    tex_path = tables_dir / 'scaling_table.tex'
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')
    print(f"  -> {tex_path}")


# ============================================================
# Ablation study visualization
# ============================================================

def plot_ablation(df: pd.DataFrame, out_dir: Path):
    """Ablation Study: Relative Performance vs. GPU-HBM3 Baseline.

    Shows all 7 implementations normalized to GPU-HBM3 (Single-stream).
    gpu_stream (Stage 1) is fully included to demonstrate the effect of
    double-buffering independently of Unified Memory.
    """
    if df.empty:
        print("[SKIP] Ablation graph: no data")
        return

    df = add_graph_metadata(df)

    # Normalize implementation names (unify lowercase/mixed case)
    name_map = {
        'sequential':     'Sequential',
        'openmp':         'OpenMP',
        'omp':            'OpenMP',
        'gpu':            'GPU',
        'gpu_stream':     'GPU_Stream',
        'gpu_managed':    'GPU_Managed',
        'gpu_readmostly': 'GPU_ReadMostly',
        'gpu_opt':        'GPU_Opt',
    }
    df = df.copy()
    df['Implementation'] = df['Implementation'].replace(name_map)

    ablation_order = ['Sequential', 'OpenMP', 'GPU', 'GPU_Stream',
                      'GPU_Managed', 'GPU_ReadMostly', 'GPU_Opt']
    impls_present  = [i for i in ablation_order if i in df['Implementation'].values]
    if len(impls_present) < 2:
        print(f"[SKIP] Ablation graph: insufficient implementations ({impls_present})")
        return

    graphs_sorted = sorted(df['graph_base'].unique(), key=lambda g: GRAPH_NODES.get(g, 0))
    n_graphs = len(graphs_sorted)
    n_impls  = len(impls_present)
    x = np.arange(n_graphs)
    width = 0.8 / max(n_impls, 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Left: absolute GTEPS bar chart ----
    ax = axes[0]
    for i, impl in enumerate(impls_present):
        sub = df[df['Implementation'] == impl]
        gteps = []
        for g in graphs_sorted:
            vals = sub[sub['graph_base'] == g]['GTEPS'].values
            gteps.append(vals[0] if len(vals) > 0 and vals[0] > 0 else np.nan)
        offset = (i - n_impls / 2 + 0.5) * width
        bars = ax.bar(x + offset, gteps, width,
                      label=IMPL_LABELS.get(impl, impl),
                      color=IMPL_COLORS.get(impl, '#888888'), alpha=0.85)
        _annotate_bars(ax, bars, fmt='{:.2f}')

    x_labels = [graph_display_name(g) for g in graphs_sorted]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=40, ha='right')
    ax.set_xlabel('Number of Vertices')
    ax.set_ylabel('Throughput (GTEPS)')
    ax.set_title('Throughput (GTEPS) by Implementation')
    _legend_outside(ax, n_impls)
    ax.grid(True, axis='y', alpha=0.3)

    # ---- Right: relative performance vs. GPU-HBM3 baseline ----
    ax2 = axes[1]
    ref_impl = 'GPU'
    if ref_impl in impls_present:
        ref_sub = df[df['Implementation'] == ref_impl]
        for i, impl in enumerate(impls_present):
            if impl == ref_impl:
                continue
            sub = df[df['Implementation'] == impl]
            ratios = []
            for g in graphs_sorted:
                ref_vals = ref_sub[ref_sub['graph_base'] == g]['GTEPS'].values
                tgt_vals = sub[sub['graph_base'] == g]['GTEPS'].values
                if (len(ref_vals) > 0 and len(tgt_vals) > 0
                        and ref_vals[0] > 0 and tgt_vals[0] > 0):
                    ratios.append(tgt_vals[0] / ref_vals[0])
                else:
                    ratios.append(np.nan)
            j = impls_present.index(impl)
            offset = (j - n_impls / 2 + 0.5) * width
            bars2 = ax2.bar(x + offset, ratios, width,
                            label=IMPL_LABELS.get(impl, impl),
                            color=IMPL_COLORS.get(impl, '#888888'), alpha=0.85)
            _annotate_bars(ax2, bars2, fmt='{:.2f}')

        ax2.axhline(1.0, color='black', linestyle='--', linewidth=0.8,
                    label='GPU-HBM3 (Single-stream) baseline = 1.0')
        ax2.set_xticks(x)
        ax2.set_xticklabels(x_labels, rotation=40, ha='right')
        ax2.set_xlabel('Number of Vertices')
        ax2.set_ylabel('Relative Performance (GPU-HBM3 = 1.0)')
        ax2.set_title('Ablation Study: Relative Performance vs. GPU-HBM3 Baseline')
        _legend_outside(ax2, n_impls)
        ax2.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    _savefig(fig, out_dir / 'ablation_study.pdf')
    plt.close(fig)


def make_ablation_table(df: pd.DataFrame, tables_dir: Path):
    """Generate TSV/LaTeX tables for ablation study results (all 7 implementations)."""
    if df.empty:
        print("[SKIP] Ablation table: no data")
        return

    df = add_graph_metadata(df)
    name_map = {
        'sequential':     'Sequential',
        'openmp':         'OpenMP',
        'omp':            'OpenMP',
        'gpu':            'GPU',
        'gpu_stream':     'GPU_Stream',
        'gpu_managed':    'GPU_Managed',
        'gpu_readmostly': 'GPU_ReadMostly',
        'gpu_opt':        'GPU_Opt',
    }
    df = df.copy()
    df['Implementation'] = df['Implementation'].replace(name_map)

    ablation_order = ['Sequential', 'OpenMP', 'GPU', 'GPU_Stream',
                      'GPU_Managed', 'GPU_ReadMostly', 'GPU_Opt']
    impls_present  = [i for i in ablation_order if i in df['Implementation'].values]

    # GTEPS pivot
    pivot = df.pivot_table(index='graph_base', columns='Implementation',
                           values='GTEPS', aggfunc='first')
    pivot = pivot.reindex(columns=[i for i in ablation_order if i in pivot.columns])
    graphs_sorted = sorted(pivot.index, key=lambda g: GRAPH_NODES.get(g, 0))
    pivot = pivot.reindex(graphs_sorted)

    tsv_path = tables_dir / 'ablation_gteps_table.tsv'
    pivot.to_csv(tsv_path, sep='\t', float_format='%.3f')
    print(f"  -> {tsv_path}")

    # Relative speedup vs. GPU baseline
    if 'GPU' in pivot.columns:
        rel = pivot.div(pivot['GPU'], axis=0)
        rel_path = tables_dir / 'ablation_speedup_vs_gpu.tsv'
        rel.to_csv(rel_path, sep='\t', float_format='%.3f')
        print(f"  -> {rel_path}")

    # LaTeX table
    col_labels = [IMPL_LABELS.get(c, c) for c in pivot.columns]
    tex_lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Ablation study: GTEPS (relative performance vs. GPU)}',
        r'\label{tab:ablation}',
        r'\begin{tabular}{l' + 'r' * len(pivot.columns) + '}',
        r'\hline',
        'Graph & ' + ' & '.join(col_labels) + r' \\',
        r'\hline',
    ]
    for gname in pivot.index:
        row = pivot.loc[gname]
        n = GRAPH_NODES.get(gname, 0)
        cells = []
        for impl in pivot.columns:
            v = row[impl]
            cells.append('--' if pd.isna(v) else f'{v:.2f}')
        tex_lines.append(
            f'{format_nodes(n)} ({gname}) & ' + ' & '.join(cells) + r' \\'
        )
    tex_lines += [r'\hline', r'\end{tabular}', r'\end{table}']
    tex_path = tables_dir / 'ablation_table.tex'
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')
    print(f"  -> {tex_path}")



# ============================================================
# Additional single-GPU analysis figures
# ============================================================

def plot_scaling_analysis(df: pd.DataFrame, out_dir: Path):
    """Execution Time Scaling with Graph Size (line chart, all implementations).

    Identical to plot_execution_time but saved with the canonical name
    'scaling_analysis' and focused on scaling behavior.
    """
    if df.empty:
        print("[SKIP] Scaling analysis graph: no data")
        return

    df = add_graph_metadata(df)
    df = df[df['Time_sec'] > 0].copy()
    if df.empty:
        print("[SKIP] Scaling analysis graph: all values <= 0")
        return

    graphs_sorted = sorted(df['graph_base'].unique(), key=lambda g: GRAPH_NODES.get(g, 0))
    impls = get_impl_order(df)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    for impl in impls:
        sub = df[df['Implementation'] == impl].sort_values('nodes')
        if sub.empty:
            continue
        xs = list(range(len(sub)))
        ys = sub['Time_sec'].values
        ax.plot(xs, ys,
                label=IMPL_LABELS.get(impl, impl),
                color=IMPL_COLORS.get(impl, '#888888'),
                marker=IMPL_MARKERS.get(impl, 'o'),
                linewidth=1.8, markersize=6)

    x_labels = [graph_display_name(g) for g in graphs_sorted]
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=40, ha='right')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Vertices')
    ax.set_ylabel('Execution Time (seconds) [log scale]')
    ax.set_title('Execution Time Scaling with Graph Size')
    _legend_outside(ax, len(impls))
    ax.grid(True, which='both', alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()
    _savefig(fig, out_dir / 'scaling_analysis.pdf')
    plt.close(fig)


def plot_threshold_sensitivity(tsv_path: Path, out_dir: Path):
    """Topology Placement Threshold Sensitivity (line chart).

    X-axis: threshold values (0.001, 0.01, 0.1, 0.35)
    Y-axis: execution time (seconds)
    One line per graph dataset tested.
    Data source: data/threshold_sensitivity.tsv (skipped gracefully if missing).
    """
    if not tsv_path.exists():
        print(f"[SKIP] Threshold sensitivity: {tsv_path} not found")
        return

    try:
        df = pd.read_csv(tsv_path, sep='\t', comment='#',
                         na_values=['—', '-', ''])
    except Exception as e:
        print(f"[SKIP] Threshold sensitivity: failed to load {tsv_path}: {e}")
        return

    df.columns = df.columns.str.strip()

    # Expected columns: Threshold  Graph  Time_sec  (optional: GTEPS)
    required = {'Threshold', 'Graph', 'Time_sec'}
    if not required.issubset(df.columns):
        print(f"[SKIP] Threshold sensitivity: missing columns in {tsv_path} "
              f"(need {required}, got {set(df.columns)})")
        return

    df['Threshold'] = pd.to_numeric(df['Threshold'], errors='coerce')
    df['Time_sec']  = pd.to_numeric(df['Time_sec'],  errors='coerce')
    df = df.dropna(subset=['Threshold', 'Time_sec'])
    df = df[df['Time_sec'] > 0]

    if df.empty:
        print("[SKIP] Threshold sensitivity: no valid data rows")
        return

    df['graph_base'] = df['Graph'].apply(lambda x: Path(x).name)
    graphs = sorted(df['graph_base'].unique(),
                    key=lambda g: GRAPH_NODES.get(g, 0))

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    for g in graphs:
        sub = df[df['graph_base'] == g].sort_values('Threshold')
        if sub.empty:
            continue
        label = graph_display_name(g)
        ax.plot(sub['Threshold'], sub['Time_sec'],
                marker='o', linewidth=1.8, markersize=6, label=label)

    # Vertical dashed line at the default threshold (0.35)
    ax.axvline(x=0.35, color='black', linestyle='--', linewidth=1.2,
               label='Default (0.35)')
    ax.text(0.35, 0.98, '  Default\n  (0.35)',
            transform=ax.get_xaxis_transform(),
            va='top', fontsize=9, color='black')

    ax.set_xscale('log')
    ax.set_xlabel('Topology Placement Threshold (log scale)')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Topology Placement Threshold Sensitivity')
    _legend_outside(ax, len(graphs))
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    _savefig(fig, out_dir / 'threshold_sensitivity.pdf')
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Miyabi BC Experiment: Comprehensive Analysis Script'
    )
    parser.add_argument(
        '--result-dir',
        default=str(Path(__file__).parent.parent / 'build_miyabi'),
        help='Directory containing TSV result files'
    )
    parser.add_argument(
        '--output-dir',
        default=str(Path(__file__).parent / 'figures'),
        help='Output directory for figures'
    )
    parser.add_argument(
        '--tables-dir',
        default=str(Path(__file__).parent / 'tables'),
        help='Output directory for tables'
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.output_dir)
    tables_dir = Path(args.tables_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print("=" * 60)
    print("  Miyabi BC Experiment -- Analysis & Visualization")
    print(f"  Result directory: {result_dir}")
    print(f"  Output directory: {out_dir}")
    print("=" * 60)

    # ----- Load data -----
    baseline_path = result_dir / 'result_baseline' / 'summary.tsv'
    phase2_path   = result_dir / 'result_phase2'   / 'summary.tsv'
    bw_path       = result_dir / 'result_bandwidth' / 'bandwidth.tsv'

    print("\n--- Loading data ---")
    baseline_df = load_tsv_skip_duplicate_headers(baseline_path)
    phase2_df   = load_tsv_skip_duplicate_headers(phase2_path)
    bw_df       = load_tsv_skip_duplicate_headers(bw_path)

    print(f"baseline: {len(baseline_df)} rows ({'valid' if not baseline_df.empty else 'empty/missing'})")
    print(f"phase2:   {len(phase2_df)} rows ({'valid' if not phase2_df.empty else 'empty/missing'})")
    print(f"bandwidth:{len(bw_df)} rows ({'valid' if not bw_df.empty else 'empty/missing'})")

    # batchsize sweep: multiple files
    batchsize_dir = result_dir / 'result_batchsize_sweep'
    batchsize_frames = []
    if batchsize_dir.exists():
        for f in batchsize_dir.glob('batchsize_*.tsv'):
            sub = load_tsv_skip_duplicate_headers(f)
            if not sub.empty:
                batchsize_frames.append(sub)
    batchsize_df = pd.concat(batchsize_frames) if batchsize_frames else pd.DataFrame()

    # MPI scaling: result_scaling/ directory
    scaling_dir = result_dir / 'result_scaling'
    strong_dfs = []
    if scaling_dir.exists():
        for f in scaling_dir.glob('strong_*.tsv'):
            sub = load_scaling_tsv(f)
            if not sub.empty:
                strong_dfs.append(sub)
    strong_df = pd.concat(strong_dfs, ignore_index=True) if strong_dfs else pd.DataFrame()

    weak_path = scaling_dir / 'weak_scaling.tsv'
    weak_df = load_scaling_tsv(weak_path) if scaling_dir.exists() else pd.DataFrame()

    # Ablation study: result_ablation/ablation_summary.tsv
    ablation_path = result_dir / 'result_ablation' / 'ablation_summary.tsv'
    ablation_df = load_tsv_skip_duplicate_headers(ablation_path)

    # Threshold sensitivity: data/threshold_sensitivity.tsv
    threshold_path = result_dir.parent / 'data' / 'threshold_sensitivity.tsv'
    if not threshold_path.exists():
        # Also try relative to the script
        threshold_path = Path(__file__).parent.parent / 'data' / 'threshold_sensitivity.tsv'

    print(f"strong scaling: {len(strong_df)} rows ({'valid' if not strong_df.empty else 'empty/missing'})")
    print(f"weak scaling:   {len(weak_df)} rows ({'valid' if not weak_df.empty else 'empty/missing'})")
    print(f"ablation:       {len(ablation_df)} rows ({'valid' if not ablation_df.empty else 'empty/missing'})")
    print(f"threshold sens: {'found' if threshold_path.exists() else 'not found'} ({threshold_path})")

    # ----- Figure generation -----
    print("\n--- Generating figures ---")

    # Merge baseline + phase2 data for execution time graph
    non_empty = [df for df in [baseline_df, phase2_df] if not df.empty]
    combined = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    if not combined.empty:
        combined = combined.drop_duplicates(subset=['Implementation', 'Graph'])

    plot_execution_time(combined, out_dir)

    # Determine baseline implementation name from data
    if not combined.empty:
        impls_in_data = set(combined['Implementation'].unique())
        seq_impl = 'Sequential' if 'Sequential' in impls_in_data else 'sequential'
        if 'OpenMP' in impls_in_data:
            omp_impl = 'OpenMP'
        elif 'openmp' in impls_in_data:
            omp_impl = 'openmp'
        else:
            omp_impl = 'omp'
    else:
        seq_impl, omp_impl = 'Sequential', 'OpenMP'

    plot_speedup(combined, out_dir, baseline_impl=seq_impl)
    plot_speedup(combined, out_dir, baseline_impl=omp_impl)
    plot_gteps(combined, out_dir)
    plot_phase2_comparison(phase2_df if not phase2_df.empty else combined, out_dir)
    plot_scaling_analysis(combined, out_dir)
    plot_bandwidth(bw_df, out_dir)
    plot_bandwidth_by_size(bw_df, out_dir)
    plot_batchsize_sweep(batchsize_df, out_dir)

    # Ablation graph (all 7 implementations including gpu_stream).
    # Fall back to combined data if dedicated ablation TSV is absent.
    ablation_data = ablation_df if not ablation_df.empty else combined
    plot_ablation(ablation_data, out_dir)

    # Threshold sensitivity (skips gracefully if TSV is missing)
    plot_threshold_sensitivity(threshold_path, out_dir)

    # MPI scaling graphs
    if not strong_df.empty:
        plot_strong_scaling(strong_df, out_dir)
        plot_sharp_comparison(strong_df, out_dir)
    if not weak_df.empty:
        plot_weak_scaling(weak_df, out_dir)

    # ----- Table generation -----
    print("\n--- Generating tables ---")
    make_exec_time_table(combined, tables_dir)
    make_speedup_table(combined, tables_dir, baseline_impl=seq_impl)
    make_speedup_table(combined, tables_dir, baseline_impl=omp_impl)
    make_gteps_table(combined, tables_dir)
    make_bandwidth_table(bw_df, tables_dir)
    make_scaling_table(strong_df, tables_dir)
    make_ablation_table(ablation_data, tables_dir)

    # ----- Thesis adequacy assessment -----
    assess_thesis_adequacy(baseline_df, phase2_df, bw_df)

    print("\nAnalysis complete.")
    print(f"Figures: {out_dir}/")
    print(f"Tables:  {tables_dir}/")


if __name__ == '__main__':
    main()
