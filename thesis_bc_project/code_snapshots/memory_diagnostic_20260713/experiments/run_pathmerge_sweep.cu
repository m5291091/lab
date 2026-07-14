// ============================================================
//  run_pathmerge_sweep — PathMerge (ベースライン) のバッチサイズ探索ランナー
//
//  Galliot path-merging BC のバッチサイズ (per-batch のソース数) を可変にして
//  複数サイズを順に計測し、最速のバッチサイズを報告する。
//  現行実装は int2 フロンティア + per-source 配列方式のため、旧来の 64 上限は
//  無く、HBM3 メモリの許す限り任意サイズを探索できる (超過分は自動クランプ)。
//
//  使い方:
//    run_pathmerge_sweep <graph_file> [batch_list] [--dump-bc]
//      batch_list : カンマ区切り (例: "16,32,64,128,256")
//                   既定 "1,2,4,8,16,32,64,128,256"
//
//  stdout は run_benchmark と同じ TSV 契約 (バッチごとに 1 行):
//    PathMerge_b<N> <TAB> Graph <TAB> Time_sec <TAB> GTEPS
// ============================================================

#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "common.hpp"
#include "graph.hpp"
#include "runner.hpp"
#include "brandes_baseline.hpp"

using namespace std;

static void print_usage(const char* prog)
{
    cerr << "Usage: " << prog << " <graph_file> [batch_list] [--dump-bc]" << endl;
    cerr << "  batch_list : カンマ区切りのバッチサイズ (例: 16,32,64,128,256)" << endl;
    cerr << "               既定 1,2,4,8,16,32,64,128,256" << endl;
}

static vector<int> parse_batch_list(const string& s)
{
    vector<int> out;
    stringstream ss(s);
    string tok;
    while (getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        int v = atoi(tok.c_str());
        if (v > 0) out.push_back(v);
    }
    return out;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    string graph_file_path = argv[1];
    string list_arg        = "1,2,4,8,16,32,64,128,256";
    bool   dump_bc         = false;

    for (int i = 2; i < argc; ++i) {
        string a = argv[i];
        if (a == "--dump-bc") dump_bc = true;
        else                  list_arg = a;
    }

    auto resolved = resolve_graph_path(graph_file_path);
    if (!resolved.has_value()) {
        cerr << "Error: Could not open graph file " << graph_file_path << endl;
        cerr << "       cwd: " << filesystem::current_path().string() << endl;
        return 1;
    }
    graph_file_path = *resolved;

    Graph graph;
    if (!load_graph(graph_file_path, graph)) {
        return 1;
    }

    vector<int> batches = parse_batch_list(list_arg);
    if (batches.empty()) {
        cerr << "Error: no valid batch sizes in '" << list_arg << "'" << endl;
        return 1;
    }

    double best_time  = numeric_limits<double>::max();
    int    best_batch = -1;

    for (int b : batches) {
        const string name = "PathMerge_b" + to_string(b);
        cerr << "=== PathMerge batch_size=" << b << " ===" << endl;
        double t = run_brandes(name, graph_file_path,
                               [b](Graph& g) { return brandes_pathmerge_bc_batch(g, b); },
                               graph, dump_bc);
        if (t > 0.0 && t < best_time) {
            best_time  = t;
            best_batch = b;
        }
    }

    if (best_batch > 0) {
        fprintf(stderr, "\n  > [PathMerge Sweep] BEST batch_size=%d, time=%.6f s\n",
                best_batch, best_time);
    }

    return 0;
}
