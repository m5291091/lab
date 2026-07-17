#include "runner.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>

#include <omp.h>

using namespace std;

optional<string> resolve_graph_path(const string& graph_path)
{
    namespace fs = std::filesystem;
    std::error_code ec;

    fs::path input(graph_path);
    if (fs::exists(input, ec)) {
        return input.string();
    }

    fs::path cwd = fs::current_path(ec);
    if (!cwd.empty()) {
        fs::path direct = input.is_absolute() ? input : (cwd / input);
        if (fs::exists(direct, ec)) {
            return direct.lexically_normal().string();
        }
    }

    const string generic = input.generic_string();
    string data_suffix;
    size_t data_pos = generic.find("data/");
    if (data_pos != string::npos) {
        data_suffix = generic.substr(data_pos + 5);
    }

    fs::path cur = cwd;
    while (!cur.empty()) {
        if (!data_suffix.empty()) {
            fs::path by_suffix = cur / "data" / data_suffix;
            if (fs::exists(by_suffix, ec)) {
                return by_suffix.lexically_normal().string();
            }
        }

        fs::path by_basename = cur / "data" / input.filename();
        if (fs::exists(by_basename, ec)) {
            return by_basename.lexically_normal().string();
        }

        if (cur == cur.root_path()) {
            break;
        }
        fs::path parent = cur.parent_path();
        if (parent == cur) {
            break;
        }
        cur = parent;
    }

    return std::nullopt;
}

bool validate_graph(const Graph& graph, const string& path)
{
    // ホスト側 CSR 整合性検査 (BC 計測区間の外側・GPU 起動前)。
    //   O(n + m) の構造/範囲検査のみ。不正入力を黙って skip / clamp / 補正しない。
    //   対称性・多重度・連結成分などの重い解析は tools/validate_graph_csr.py が担う
    //   (実験スクリプトが runner 実行前に呼ぶ)。
    const string name = filesystem::path(path).filename().string();
    auto fail = [&](const string& detail) {
        cerr << "[GraphValidate] FATAL: invalid CSR graph '" << name << "'" << endl;
        cerr << "                path  : " << path << endl;
        cerr << "                reason: " << detail << endl;
        cerr << "                -> 不正入力のため GPU 実行前に停止する (補正・skip はしない)" << endl;
        return false;
    };

    const int n = graph.getNodeCount();
    const long long m = graph.getEdgeCount();
    const int* ptr = graph.getAdjacencyListPointers();
    const int* adj = graph.getAdjacencyList();

    if (n <= 0)        return fail("n_nodes=" + to_string(n) + " (>0 が必要)");
    if (m < 0)         return fail("n_edges=" + to_string(m) + " (>=0 が必要)");
    if (!ptr || !adj)  return fail("CSR 配列が未確保");

    const long long expect_adj = 2 * m;

    if (ptr[0] != 0)
        return fail("check=ptr[0] expected=0 actual=" + to_string(ptr[0]));

    for (int i = 0; i < n; ++i) {
        if (ptr[i + 1] < ptr[i])
            return fail("check=ptr_monotonic expected=ptr[" + to_string(i) + "]<=ptr["
                        + to_string(i + 1) + "] actual=" + to_string(ptr[i]) + ">"
                        + to_string(ptr[i + 1]));
    }

    if ((long long)ptr[n] != expect_adj)
        return fail("check=ptr[n] expected=2*m=" + to_string(expect_adj)
                    + " actual=" + to_string(ptr[n]));

    for (long long i = 0; i < expect_adj; ++i) {
        const int v = adj[i];
        if (v < 0 || v >= n)
            return fail("check=adjacency_range index=" + to_string(i) + " expected=[0,"
                        + to_string(n - 1) + "] actual=" + to_string(v));
    }

    return true;
}

bool load_graph(const string& resolved_path, Graph& graph)
{
    // 既存の Graph::readGraph が stdin を使うため freopen で流し込む
    if (freopen(resolved_path.c_str(), "r", stdin) == nullptr) {
        cerr << "Error: Could not open graph file " << resolved_path << endl;
        return false;
    }
    const bool read_ok = graph.readGraph();
    graph.setSourcePath(resolved_path);
    fclose(stdin);

    if (!read_ok) {
        cerr << "[GraphValidate] FATAL: could not read CSR graph '"
             << filesystem::path(resolved_path).filename().string() << "'" << endl;
        cerr << "                path  : " << resolved_path << endl;
        cerr << "                reason: " << graph.getReadError() << endl;
        return false;
    }
    // 計測区間の外側で検査する (run_brandes の omp_get_wtime 区間には含まれない)
    return validate_graph(graph, resolved_path);
}

double run_brandes(const string& impl_name,
                   const string& graph_path,
                   function<vector<double>(Graph&)> brandes_func,
                   Graph& graph,
                   bool dump_bc)
{
    string graph_name = filesystem::path(graph_path).filename().string();
    int n_nodes = graph.getNodeCount();
    long long n_edges = graph.getEdgeCount();
    cerr << "Running: " << impl_name << " on " << graph_name << "..." << endl;

    double start_time = omp_get_wtime();
    vector<double> bc = brandes_func(graph);
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    // GTEPS: all-pairs BC は n_nodes 回の BFS で各 n_edges 本の辺を辿る
    double gteps = (elapsed_time > 0.0)
        ? ((double)n_nodes * (double)n_edges / elapsed_time / 1e9)
        : 0.0;

    // サマリ用に最大 BC 値とその index を求める
    double max_bc = -1.0;
    int max_idx = -1;
    if (!bc.empty()) {
        for (int i = 0; i < n_nodes; ++i) {
            if (bc[i] > max_bc) {
                max_bc = bc[i];
                max_idx = i;
            }
        }
    }

    // stderr へサマリ出力
    fprintf(stderr, "  > index : %d, Maximum Betweenness Centrality ==> %0.2lf\n", max_idx, max_bc);
    fprintf(stderr, "  > Elapse time [sec.] = %lf \n", elapsed_time);
    fprintf(stderr, "  > GTEPS = %.4f (nodes=%d, edges=%lld)\n", gteps, n_nodes, n_edges);

    if (dump_bc) {
        // --dump-bc モード: 正確性検証のため全 BC 値を出力
        // 1 行目はヘッダ、以降は node_idx\tbc_value
        printf("# impl=%s graph=%s nodes=%d\n", impl_name.c_str(), graph_name.c_str(), n_nodes);
        for (int i = 0; i < n_nodes; ++i) {
            printf("%d\t%.15e\n", i, bc.empty() ? 0.0 : bc[i]);
        }
    } else {
        // 通常モード: タブ区切りサマリ行を stdout に出力
        // Format: Impl  Graph  Time_sec  GTEPS
        printf("%s\t%s\t%.6f\t%.4f\n", impl_name.c_str(), graph_name.c_str(), elapsed_time, gteps);
    }

    return elapsed_time;
}
