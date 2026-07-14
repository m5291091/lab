// brandes_cugraph.cu — libcugraph C++/CUDA コアを直接呼び出す Exact BC 実装
//
// Graph が保持する CSR (offsets, indices) をそのままデバイスへ転送し、
// cugraph::graph_t を直接構築して
// cugraph::betweenness_centrality() を実行する。

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include "Graph.h"

#include <cstdio>
#include <cstdint>
#include <chrono>
#include <memory>
#include <optional>
#include <vector>

using vertex_t = int32_t;
using edge_t   = int32_t;
using weight_t = double;

static constexpr bool store_transposed = false;
static constexpr bool multi_gpu        = false;

std::vector<double> brandes_cugraph_bc(Graph& graph) {
    static_assert(sizeof(int) == sizeof(edge_t), "Graph CSR offset type must match edge_t.");
    static_assert(sizeof(int) == sizeof(vertex_t), "Graph CSR index type must match vertex_t.");

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    const int n_nodes  = graph.getNodeCount();
    const int* offsets = graph.getAdjacencyListPointers();
    const int* indices = graph.getAdjacencyList();
    // CSR の adjacencyList は双方向 (2 * edgeCount 要素)
    const int nnz = offsets[n_nodes];

    // 1. CUDA / RMM / RAFT 初期化
    RAFT_CUDA_TRY(cudaSetDevice(0));
    // gpu_opt と同じく cudaMallocManaged ベースのメモリリソースを使用。
    // GH200 NVLink-C2C 環境では GPU HBM + CPU DRAM をシームレスに利用でき、
    // デバイスメモリ (cudaMalloc) の 50 GB 制限を回避する。
    // プールを使わず直接 managed_memory_resource を使用することで、
    // 大規模グラフでの page fault 処理を OS/ドライバに委ねる。
    auto mr = std::make_shared<rmm::mr::managed_memory_resource>();
    rmm::mr::set_current_device_resource(mr.get());

    raft::handle_t handle(rmm::cuda_stream_per_thread,
                          std::shared_ptr<rmm::cuda_stream_pool>{nullptr},
                          mr);

    auto t1 = clock::now();

    // 2. CSR をデバイスへ転送して cugraph::graph_t を直接構築
    rmm::device_uvector<edge_t> d_offsets(static_cast<size_t>(n_nodes) + 1, handle.get_stream());
    rmm::device_uvector<vertex_t> d_indices(nnz, handle.get_stream());
    raft::update_device(d_offsets.data(),
                        reinterpret_cast<edge_t const*>(offsets),
                        d_offsets.size(),
                        handle.get_stream());
    raft::update_device(d_indices.data(),
                        reinterpret_cast<vertex_t const*>(indices),
                        d_indices.size(),
                        handle.get_stream());

    cugraph::graph_meta_t<vertex_t, edge_t, multi_gpu> meta{};
    meta.number_of_vertices = static_cast<vertex_t>(n_nodes);
    meta.properties         = cugraph::graph_properties_t{
      true /* is_symmetric */, false /* is_multigraph */};

    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> graph_cg(
      handle, std::move(d_offsets), std::move(d_indices), std::move(meta), false);

    handle.sync_stream();
    auto t2 = clock::now();

    // 3. Exact Betweenness Centrality 計算
    //    vertices = std::nullopt → 全頂点ソース (Exact)
    //    normalized = false      → 非正規化 (既存実装と同じ)
    //    include_endpoints = false
    auto graph_view = graph_cg.view();
    auto d_bc = cugraph::betweenness_centrality<vertex_t, edge_t, weight_t, multi_gpu>(
        handle,
        graph_view,
        std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>(std::nullopt),
        std::optional<raft::device_span<vertex_t const>>(std::nullopt),
        false,  // normalized
        false,  // include_endpoints
        false); // do_expensive_check

    handle.sync_stream();
    auto t3 = clock::now();

    // 4. 結果をホストへ転送 (CSR の元頂点順)
    std::vector<double> bc(n_nodes, 0.0);
    raft::update_host(bc.data(), d_bc.data(), d_bc.size(), handle.get_stream());
    handle.sync_stream();

    auto t4 = clock::now();

    auto sec = [](auto d) { return std::chrono::duration<double>(d).count(); };
    // Phase breakdown:
    //   Init     = CUDA context init + RMM pool setup + RAFT handle
    //   H2D+Build= host→device transfer + cugraph::graph_t construction
    //   BC       = 純粋な BC 計算時間 (アルゴリズム本体)
    //   D2H      = device→host 結果転送
    // Elapse time (main.cpp 計測) = Init + H2D+Build + BC + D2H
    double bc_sec = sec(t3 - t2);
    std::fprintf(stderr,
                 "  > [cuGraph Phase] Init: %.4f s, H2D+Build: %.4f s, BC: %.4f s, D2H: %.4f s\n",
                 sec(t1 - t0), sec(t2 - t1), bc_sec, sec(t4 - t3));
    std::fprintf(stderr,
                 "  > [cuGraph BC-only] %.4f s (純粋 BC 計算, CUDA init 等除く)\n",
                 bc_sec);

    return bc;
}
