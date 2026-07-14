// ============================================================
//  run_ablation — 提案手法の 3 工夫の寄与を個別測定するランナー
//
//  3 つの工夫 (ハイブリッド BFS / Warp 協調依存蓄積 / 2 ストリーム非同期初期化)
//  はコンパイル時テンプレートで ON/OFF される。本ランナーは指定された構成を
//  実行時に選択し、共通ランナーで計測する。
//
//  使い方:
//    run_ablation <graph_file> [mode] [--dump-bc]
//      mode:
//        all       すべての 8 構成を実行 (既定, フル要因ablation)
//        baseline  H0_W0_A0 (全工夫 OFF)
//        full      H1_W1_A1 (全工夫 ON)
//        H<b>W<b>A<b>  個別構成 (例: H1W0A1 / H1_W0_A1)
//
//  stdout は run_benchmark と同じ TSV 契約:
//    Ablation_H<b>_W<b>_A<b> <TAB> Graph <TAB> Time_sec <TAB> GTEPS
// ============================================================

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "common.hpp"
#include "graph.hpp"
#include "runner.hpp"
#include "ablation_config.hpp"

using namespace std;

static void print_usage(const char* prog)
{
    cerr << "Usage: " << prog << " <graph_file> [mode] [--dump-bc]" << endl;
    cerr << "  mode: all (default) | baseline | full | H<b>W<b>A<b> (例: H1W0A1)" << endl;
}

// "H1W0A1" / "H1_W0_A1" 形式を解釈する
static bool parse_config_token(const string& tok, AblationConfig& cfg)
{
    auto get_flag = [&](char key, bool& out) -> bool {
        size_t p = tok.find(key);
        if (p == string::npos || p + 1 >= tok.size()) return false;
        char c = tok[p + 1];
        if (c == '0') { out = false; return true; }
        if (c == '1') { out = true;  return true; }
        return false;
    };
    bool h = false, w = false, a = false;
    if (get_flag('H', h) && get_flag('W', w) && get_flag('A', a)) {
        cfg.hybrid_bfs = h;
        cfg.warp_coop  = w;
        cfg.async_init = a;
        return true;
    }
    return false;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    string graph_file_path = argv[1];
    string mode            = "all";
    bool   dump_bc         = false;

    for (int i = 2; i < argc; ++i) {
        string a = argv[i];
        if (a == "--dump-bc") dump_bc = true;
        else                  mode = a;
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

    // 実行する構成リストを組み立てる
    vector<AblationConfig> configs;
    if (mode == "all") {
        for (int h = 0; h < 2; ++h)
            for (int w = 0; w < 2; ++w)
                for (int a = 0; a < 2; ++a)
                    configs.push_back(AblationConfig{(bool)h, (bool)w, (bool)a});
    } else if (mode == "baseline") {
        configs.push_back(AblationConfig{false, false, false});
    } else if (mode == "full") {
        configs.push_back(AblationConfig{true, true, true});
    } else {
        AblationConfig cfg;
        if (!parse_config_token(mode, cfg)) {
            cerr << "Error: Unknown mode '" << mode << "'" << endl;
            print_usage(argv[0]);
            return 1;
        }
        configs.push_back(cfg);
    }

    // Warmup (計測外): CUDA コンテキスト生成等の一回性コストを
    // 先頭構成 (baseline) が吸収して寄与を過大評価しないための捨て実行
    const char* warm_env = getenv("BC_ABLATION_WARMUP");
    if (!warm_env || string(warm_env) != string("0")) {
        cerr << "=== Warmup (untimed, H1W1A1) ===" << endl;
        AblationConfig warm{true, true, true};
        (void)brandes_gpu_opt_ablation(graph, warm);
    }

    for (const auto& cfg : configs) {
        const string name = "Ablation_" + ablation_label(cfg);
        cerr << "=== " << ablation_describe(cfg) << " ===" << endl;
        run_brandes(name, graph_file_path,
                    [&cfg](Graph& g) { return brandes_gpu_opt_ablation(g, cfg); },
                    graph, dump_bc);
    }

    return 0;
}
