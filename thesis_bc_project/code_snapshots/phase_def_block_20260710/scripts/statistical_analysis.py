#!/usr/bin/env python3
import sys
import os
import re
import csv
import math
from collections import defaultdict

try:
    from scipy.stats import wilcoxon
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

def parse_results(tsv_paths):
    # Returns raw_trials: {(method, graph): [time1, time2, ...]}
    raw_trials = defaultdict(list)
    for tsv_path in tsv_paths:
        if not os.path.isfile(tsv_path): continue
        
        with open(tsv_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 7 or row[0] == "Implementation": continue
                if row[5] in ("FAIL", "TIMEOUT"): continue
                impl, graph = row[0], row[1]
                try:
                    t = float(row[5])
                    raw_trials[(impl, graph)].append(t)
                except ValueError:
                    pass
    return raw_trials

def parse_phases(log_paths):
    # Returns phases: {(method, graph): {'h2d': x, 'bfs': y, 'bwd': z, 'd2h': w}}
    phases = defaultdict(lambda: {'h2d': 0.0, 'bfs': 0.0, 'bwd': 0.0, 'd2h': 0.0, 'count': 0})
    
    current_impl = None
    current_graph = None
    
    for log_path in log_paths:
        if not os.path.isfile(log_path): continue
        with open(log_path, "r") as f:
            for line in f:
                m1 = re.match(r"Running:\s+(\S+)\s+on\s+(\S+)\.\.\.", line)
                if m1:
                    current_impl = m1.group(1)
                    current_graph = m1.group(2).rstrip(".")
                    continue
                
                if "[GPU Phase]" in line and current_impl and current_graph:
                    h2d = re.search(r"H2D.*?wall=([\d.]+)\s*s", line)
                    bfs = re.search(r"BFS.*?wall=([\d.]+)\s*s", line)
                    bwd = re.search(r"Backward.*?wall=([\d.]+)\s*s", line)
                    d2h = re.search(r"D2H.*?wall=([\d.]+)\s*s", line)
                    # 現行ログは H2D/D2H wall を持たず Prefetch cum を持つ (UM 版)。
                    # 転送成分としては Prefetch を h2d 相当に用いる。
                    pf = re.search(r"Prefetch\s*(?:cum|wall)=([\d.]+)\s*s", line)

                    key = (current_impl, current_graph)
                    if bfs and bwd:
                        phases[key]['h2d'] += float(h2d.group(1)) if h2d else (float(pf.group(1)) if pf else 0.0)
                        phases[key]['bfs'] += float(bfs.group(1))
                        phases[key]['bwd'] += float(bwd.group(1))
                        phases[key]['d2h'] += float(d2h.group(1)) if d2h else 0.0
                        phases[key]['count'] += 1
                        
    # Average them
    avg_phases = {}
    for k, v in phases.items():
        if v['count'] > 0:
            avg_phases[k] = {
                'h2d': v['h2d'] / v['count'],
                'bfs': v['bfs'] / v['count'],
                'bwd': v['bwd'] / v['count'],
                'd2h': v['d2h'] / v['count'],
            }
    return avg_phases

def run_wilcoxon(trials_opt, trials_pure):
    if not HAS_LIBS:
        return "N/A (scipy not installed)"
    
    n = min(len(trials_opt), len(trials_pure))
    if n < 5:
        return f"N/A (needs >=5 trials, got {n})"
        
    try:
        stat, p = wilcoxon(trials_opt[:n], trials_pure[:n])
        if p > 0.05:
            return f"p={p:.3f} (No sign. diff, ≈)"
        else:
            return f"p={p:.3f} (Sign. diff)"
    except Exception as e:
        return f"Error: {str(e)}"

def generate_stats_table(raw_trials, out_md):
    graphs = sorted(list(set(g for (m, g) in raw_trials.keys())))
    methods = ["GPU_Opt", "GPU_Opt_Pure"]
    
    with open(out_md, "w") as f:
        f.write("# Statistical Analysis (GPU_Opt vs GPU_Opt_Pure)\n\n")
        f.write("| Graph | GPU_Opt (Mean ± Std) | GPU_Opt_Pure (Mean ± Std) | Wilcoxon Test (p-value) |\n")
        f.write("|:---|---:|---:|:---|\n")
        
        for g in graphs:
            t_opt = raw_trials.get(("GPU_Opt", g), [])
            t_pure = raw_trials.get(("GPU_Opt_Pure", g), [])
            
            str_opt = "—"
            if t_opt:
                mean_o = sum(t_opt)/len(t_opt)
                std_o = math.sqrt(sum((x-mean_o)**2 for x in t_opt)/(len(t_opt)-1)) if len(t_opt)>1 else 0
                str_opt = f"{mean_o:.4f} ± {std_o:.4f} (n={len(t_opt)})"
                
            str_pure = "—"
            if t_pure:
                mean_p = sum(t_pure)/len(t_pure)
                std_p = math.sqrt(sum((x-mean_p)**2 for x in t_pure)/(len(t_pure)-1)) if len(t_pure)>1 else 0
                str_pure = f"{mean_p:.4f} ± {std_p:.4f} (n={len(t_pure)})"
                
            test_res = run_wilcoxon(t_opt, t_pure) if (t_opt and t_pure) else "—"
            
            f.write(f"| {g} | {str_opt} | {str_pure} | {test_res} |\n")

def plot_phase_breakdown(avg_phases, out_pdf):
    if not HAS_LIBS:
        print("Skipping plots (matplotlib not installed)")
        return
        
    graphs = sorted(list(set(g for (m, g) in avg_phases.keys())))
    if not graphs:
        print("Skipping phase breakdown plot (no phase data parsed)")
        return
    methods = ["GPU_Opt", "GPU_Opt_Pure"]
    
    fig, axes = plt.subplots(len(graphs), 1, figsize=(10, 3*len(graphs)))
    if len(graphs) == 1: axes = [axes]
    
    for ax, g in zip(axes, graphs):
        data = []
        labels = []
        for m in methods:
            if (m, g) in avg_phases:
                p = avg_phases[(m, g)]
                data.append([p['h2d'], p['bfs'], p['bwd'], p['d2h']])
                labels.append(m)
        
        if not data: continue
        
        data = np.array(data)
        bottoms = np.zeros(len(labels))
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        names = ['H2D (Setup)', 'BFS (Kernel)', 'Backward (Kernel)', 'D2H (Result)']
        
        for i in range(4):
            ax.barh(labels, data[:, i], left=bottoms, color=colors[i], label=names[i])
            bottoms += data[:, i]
            
        ax.set_title(f"Phase Breakdown: {g}")
        ax.set_xlabel("Time (seconds)")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(out_pdf)

def plot_batch_scalability(tsv_path, out_pdf):
    if not HAS_LIBS: return
    if not os.path.isfile(tsv_path): return
    
    data = defaultdict(lambda: {'batch': [], 'time': []})
    with open(tsv_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            # Format: Impl BatchSize Trial Time_sec GTEPS Status
            if len(row) < 6 or row[0] == "Implementation": continue
            m, b, trial, t, g, s = row[0], row[1], row[2], row[3], row[4], row[5]
            if s == "SUCCESS":
                data[m]['batch'].append(int(b))
                data[m]['time'].append(float(t))
                
    if not data: return
    
    plt.figure(figsize=(8, 5))
    for m in ["gpu_opt", "gpu_opt_pure"]:
        if m in data:
            batch_times = defaultdict(list)
            for b, t in zip(data[m]['batch'], data[m]['time']):
                batch_times[b].append(t)
                
            x_vals = sorted(list(batch_times.keys()))
            y_mean = [sum(batch_times[b])/len(batch_times[b]) for b in x_vals]
            y_std = [np.std(batch_times[b]) if len(batch_times[b]) > 1 else 0 for b in x_vals]
            
            plt.errorbar(x_vals, y_mean, yerr=y_std, marker='o', capsize=4, label=m)
            
    plt.axvline(x=8192, color='r', linestyle='--', label='HBM3 Limit (approx)')
    plt.title("Capacity vs Performance (Batch Scalability)")
    plt.xlabel("Batch Size (BC_BATCH_OVERRIDE)")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_pdf)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs='+', help="results.tsv files from small/medium/large scripts", default=[])
    parser.add_argument("--phases", nargs='+', help="phase_timing.log files from small/medium/large scripts", default=[])
    parser.add_argument("--oversubscribe", nargs='+', help="oversubscribe_results.tsv", default=[])
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    if args.results:
        raw_trials = parse_results(args.results)
        generate_stats_table(raw_trials, os.path.join(args.outdir, "statistical_test.md"))
        print(f"Generated {os.path.join(args.outdir, 'statistical_test.md')}")
        
    if args.phases:
        avg_phases = parse_phases(args.phases)
        plot_phase_breakdown(avg_phases, os.path.join(args.outdir, "phase_breakdown.pdf"))
        print(f"Generated {os.path.join(args.outdir, 'phase_breakdown.pdf')}")
        
    if args.oversubscribe:
        plot_batch_scalability(args.oversubscribe[0], os.path.join(args.outdir, "batch_scalability.pdf"))
        print(f"Generated {os.path.join(args.outdir, 'batch_scalability.pdf')}")
