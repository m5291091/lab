#include "common.h"
#include "brandes.h"
#include "Graph.h"

#include <string>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <optional>
#include <vector>

using namespace std;

// BC ダンプモード: 全 BC 値を stdout に出力 (正確性検証用)
// --dump-bc フラグが渡された場合に使用
static bool g_dump_bc = false;

static optional<string> resolve_graph_path(const string& graph_path) {
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

// Helper function to run and time a brandes implementation
void run_brandes(const string& impl_name, const string& graph_path, function<vector<double>(Graph&)> brandes_func, Graph& graph) {
    string graph_name = filesystem::path(graph_path).filename().string();
    int n_nodes = graph.getNodeCount();
    long long n_edges = graph.getEdgeCount();
    cerr << "Running: " << impl_name << " on " << graph_name << "..." << endl;

    double start_time = omp_get_wtime();
    vector<double> bc = brandes_func(graph);
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    // GTEPS: all-pairs BC processes n_nodes BFS traversals each visiting n_edges edges
    double gteps = (elapsed_time > 0.0)
        ? ((double)n_nodes * (double)n_edges / elapsed_time / 1e9)
        : 0.0;

    // Find max BC value and index for summary
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

    // Print summary to stderr
    fprintf(stderr, "  > index : %d, Maximum Betweenness Centrality ==> %0.2lf\n", max_idx, max_bc);
    fprintf(stderr, "  > Elapse time [sec.] = %lf \n", elapsed_time);
    fprintf(stderr, "  > GTEPS = %.4f (nodes=%d, edges=%lld)\n", gteps, n_nodes, n_edges);

    if (g_dump_bc) {
        // --dump-bc モード: 正確性検証のために全 BC 値を出力
        // 1行目はヘッダ、以降は node_idx\tbc_value
        printf("# impl=%s graph=%s nodes=%d\n", impl_name.c_str(), graph_name.c_str(), n_nodes);
        for (int i = 0; i < n_nodes; ++i) {
            printf("%d\t%.15e\n", i, bc.empty() ? 0.0 : bc[i]);
        }
    } else {
        // 通常モード: タブ区切りサマリ行を stdout に出力
        // Format: Impl  Graph  Time_sec  GTEPS
        printf("%s\t%s\t%.6f\t%.4f\n", impl_name.c_str(), graph_name.c_str(), elapsed_time, gteps);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <implementation> <graph_file> [--dump-bc]" << endl;
        cerr << "Available implementations: sequential, omp, gpu, gpu_opt, gpu_opt_pure, gpu_opt_pure_chunked, cugraph_bc, pathmerge_bc, all" << endl;
        cerr << "  --dump-bc : 全 BC 値を stdout に出力 (正確性検証用)" << endl;
        return 1;
    }

    string impl_choice = argv[1];
    string graph_file_path = argv[2];

    // オプション引数の解析
    for (int i = 3; i < argc; ++i) {
        if (string(argv[i]) == "--dump-bc") g_dump_bc = true;
    }

    auto resolved_graph_path = resolve_graph_path(graph_file_path);
    if (!resolved_graph_path.has_value()) {
        cerr << "Error: Could not open graph file " << graph_file_path << endl;
        cerr << "       cwd: " << filesystem::current_path().string() << endl;
        return 1;
    }
    if (*resolved_graph_path != graph_file_path) {
        cerr << "[PathResolve] Graph path resolved to: " << *resolved_graph_path << endl;
    }
    graph_file_path = *resolved_graph_path;

    // freopen is used because the existing Graph::readGraph uses stdin
    if (freopen(graph_file_path.c_str(), "r", stdin) == nullptr) {
        cerr << "Error: Could not open graph file " << graph_file_path << endl;
        return 1;
    }

    Graph graph;
    graph.readGraph();
    graph.setSourcePath(graph_file_path);
    
    // Close the file stream after reading
    fclose(stdin);

    bool run_all = (impl_choice == "all");

    if (run_all || impl_choice == "sequential") {
        run_brandes("Sequential", graph_file_path, brandes_sequential, graph);
    }
    if (run_all || impl_choice == "omp") {
        run_brandes("OpenMP", graph_file_path, brandes_omp, graph);
    }
    if (run_all || impl_choice == "gpu") {
        run_brandes("GPU", graph_file_path, brandes_gpu, graph);
    }
    if (run_all || impl_choice == "gpu_opt") {
        run_brandes("GPU_Opt", graph_file_path, brandes_gpu_opt, graph);
    }
    if (run_all || impl_choice == "gpu_opt_pure") {
        run_brandes("GPU_Opt_Pure", graph_file_path, brandes_gpu_opt_pure, graph);
    }
    if (run_all || impl_choice == "gpu_opt_pure_chunked") {
        run_brandes("GPU_Opt_Pure_Chunked", graph_file_path, brandes_gpu_opt_pure_chunked, graph);
    }
    if (run_all || impl_choice == "cugraph_bc") {
        run_brandes("cuGraph_BC", graph_file_path, brandes_cugraph_bc, graph);
    }
    if (run_all || impl_choice == "pathmerge_bc") {
        run_brandes("PathMerge_BC", graph_file_path, brandes_pathmerge_bc, graph);
    }
    if (!run_all && impl_choice != "sequential" && impl_choice != "omp"
        && impl_choice != "gpu"
        && impl_choice != "gpu_opt"
        && impl_choice != "gpu_opt_pure"
        && impl_choice != "gpu_opt_pure_chunked"
        && impl_choice != "cugraph_bc"
        && impl_choice != "pathmerge_bc") {
        cerr << "Error: Unknown implementation '" << impl_choice << "'" << endl;
        cerr << "Available implementations: sequential, omp, gpu, gpu_opt, gpu_opt_pure, gpu_opt_pure_chunked, cugraph_bc, pathmerge_bc, all" << endl;
        return 1;
    }

    return 0;
}
