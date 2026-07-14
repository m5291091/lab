// ============================================================
//  run_benchmark — 全 BC 実装の統一ベンチマーク/正確性検証ランナー
//
//  グラフを 1 度だけ読み込み、選択した実装 (または all) を共通ランナーで
//  計測する。stdout/stderr 契約は core/runner.hpp を参照。
// ============================================================

#include <filesystem>
#include <iostream>
#include <string>

#include "common.hpp"
#include "graph.hpp"
#include "runner.hpp"
#include "brandes_gpu.hpp"        // 提案手法の宣言
#include "brandes_baseline.hpp"   // ベースラインの宣言

using namespace std;

static void print_usage(const char* prog)
{
    cerr << "Usage: " << prog << " <implementation> <graph_file> [--dump-bc]" << endl;
    cerr << "Available implementations: sequential, omp, gpu, gpu_opt, gpu_opt_pure, "
            "gpu_opt_pure_chunked, cugraph_bc, pathmerge_bc, all" << endl;
    cerr << "  --dump-bc : 全 BC 値を stdout に出力 (正確性検証用)" << endl;
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    string impl_choice     = argv[1];
    string graph_file_path = argv[2];
    bool   dump_bc         = false;

    for (int i = 3; i < argc; ++i) {
        if (string(argv[i]) == "--dump-bc") dump_bc = true;
    }

    auto resolved = resolve_graph_path(graph_file_path);
    if (!resolved.has_value()) {
        cerr << "Error: Could not open graph file " << graph_file_path << endl;
        cerr << "       cwd: " << filesystem::current_path().string() << endl;
        return 1;
    }
    if (*resolved != graph_file_path) {
        cerr << "[PathResolve] Graph path resolved to: " << *resolved << endl;
    }
    graph_file_path = *resolved;

    Graph graph;
    if (!load_graph(graph_file_path, graph)) {
        return 1;
    }

    const bool run_all = (impl_choice == "all");

    // 実装名 → 表示名 → 関数 の対応表 (all は登録順に全実行)
    struct Impl { const char* key; const char* label; vector<double>(*fn)(Graph&); };
    const Impl impls[] = {
        {"sequential",           "Sequential",           brandes_sequential},
        {"omp",                  "OpenMP",                brandes_omp},
        {"gpu",                  "GPU",                   brandes_gpu},
        {"gpu_opt",              "GPU_Opt",               brandes_gpu_opt},
        {"gpu_opt_pure",         "GPU_Opt_Pure",          brandes_gpu_opt_pure},
        {"gpu_opt_pure_chunked", "GPU_Opt_Pure_Chunked",  brandes_gpu_opt_pure_chunked},
        {"cugraph_bc",           "cuGraph_BC",            brandes_cugraph_bc},
        {"pathmerge_bc",         "PathMerge_BC",          brandes_pathmerge_bc},
    };

    bool matched = false;
    for (const auto& im : impls) {
        if (run_all || impl_choice == im.key) {
            run_brandes(im.label, graph_file_path, im.fn, graph, dump_bc);
            matched = true;
        }
    }

    if (!matched) {
        cerr << "Error: Unknown implementation '" << impl_choice << "'" << endl;
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
