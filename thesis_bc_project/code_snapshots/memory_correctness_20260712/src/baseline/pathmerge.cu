#include "common.hpp"

#include "baseline_utils.h"
#include "baseline_graph.h"
#include "graph.hpp"
#include "galliot.h"

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

// ============================================================
//  path-merging-bc の utils.cpp / graph.cpp のスタブ実装
//  (nvcc でコンパイルするため .cu 内に配置)
// ============================================================

// --- CSRGraph ---
CSRGraph::~CSRGraph() { free(); }

void CSRGraph::allocate(int vertices, int edges) {
    numVertices = vertices;
    numEdges = edges;
    CUDA_CHECK(cudaMallocManaged(&rowPtr, (size_t)(numVertices + 1) * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&colIdx, (size_t)numEdges * sizeof(int)));
    std::memset(rowPtr, 0, (numVertices + 1) * sizeof(int));
    std::memset(colIdx, 0, numEdges * sizeof(int));
}

void CSRGraph::free() {
    if (rowPtr) { cudaFree(rowPtr); rowPtr = nullptr; }
    if (colIdx) { cudaFree(colIdx); colIdx = nullptr; }
    numVertices = 0; numEdges = 0;
}

// --- Timer ---
Timer::Timer() : started(false) {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
}
Timer::~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
void Timer::startTimer() { CUDA_CHECK(cudaEventRecord(start, 0)); started = true; }
float Timer::stopTimer() {
    if (!started) return 0.0f;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    started = false; return ms;
}

// --- I/O ユーティリティ (galliot.cu の compute() がリンク参照) ---
void saveBCToFile(const std::string&, const std::vector<double>&, bool) {}
void printProgress(int, int, const std::string&) {}
void printVerbose(const std::string&, bool) {}
void printDeviceInfo() {}

// loadGraphFromTSV / convertToCSR / printGraphStats — ラッパーでは不使用だがリンク用
CSRGraph* loadGraphFromTSV(const std::string&, bool) { return nullptr; }
void convertToCSR(const std::vector<Edge>&, int, CSRGraph*) {}
void printGraphStats(const CSRGraph*) {}

// ============================================================
//  PathMerge (Galliot) の 1 ソースあたり動的メモリ量 (バイト)
//    dist(int) + sigma(double) + delta(double)          = 20
//    Qcurr + Qnext + stack (それぞれ int2 = 8B, ×N)      = 24
//    → 合計 44 * N バイト / source
// ============================================================
static size_t pathmerge_bytes_per_source(int n_nodes) {
    return (size_t)n_nodes * (sizeof(int) + 2 * sizeof(double) + 3 * sizeof(int2));
}

// バッチサイズを HBM3 に収まる範囲へクランプする (メモリ枯渇による異常終了を防ぐ)。
// 返り値は実際に使用するバッチサイズ。
static int clamp_batch_to_memory(int batchSize, int n_nodes) {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    const size_t per_src = pathmerge_bytes_per_source(n_nodes);
    if (per_src == 0) return max(1, batchSize);
    const size_t budget = (size_t)(free_mem * 0.85);
    int max_fit = (int)(budget / per_src);
    max_fit = max(1, max_fit);
    if (batchSize > max_fit) {
        fprintf(stderr,
            "  > [PathMerge] WARNING: batch_size=%d exceeds HBM3 budget; clamping to %d "
            "(free=%.1f GB, %zu B/source)\n",
            batchSize, max_fit, free_mem / 1e9, per_src);
        batchSize = max_fit;
    }
    return batchSize;
}

// ============================================================
//  brandes_pathmerge_bc_batch — バッチサイズを明示指定するコア実装
//  (可変バッチサイズ探索用。人工的な 64 上限は撤廃済み)
// ============================================================
std::vector<double> brandes_pathmerge_bc_batch(Graph &graph, int batchSize) {
    int n_nodes  = graph.getNodeCount();
    const int* h_R = graph.getAdjacencyListPointers();
    const int* h_C = graph.getAdjacencyList();

    if (batchSize < 1) batchSize = 1;
    batchSize = clamp_batch_to_memory(batchSize, n_nodes);

    // 有向辺数は rowPtr[n_nodes] から取得 (getEdgeCount() は無向辺数のため半分)
    int directed_edges = h_R[n_nodes];

    // --- CSRGraph を作成 (cudaMallocManaged, .cu 内から直接呼び出し) ---
    CSRGraph* csr = new CSRGraph();
    csr->numVertices = n_nodes;
    csr->numEdges    = directed_edges;
    CUDA_ERR_CHK(cudaMallocManaged(&csr->rowPtr, (size_t)(n_nodes + 1) * sizeof(int)));
    CUDA_ERR_CHK(cudaMallocManaged(&csr->colIdx, (size_t)directed_edges * sizeof(int)));
    std::memcpy(csr->rowPtr, h_R, (size_t)(n_nodes + 1) * sizeof(int));
    std::memcpy(csr->colIdx, h_C, (size_t)directed_edges * sizeof(int));

    // メモリ使用量を表示
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    int numBatches = (n_nodes + batchSize - 1) / batchSize;
    fprintf(stderr, "  > [PathMerge] free_mem=%.1f GB, batch_size=%d, num_sources=%d, num_batches=%d\n",
            free_mem / 1e9, batchSize, n_nodes, numBatches);

    // --- GalloitBC を構築 (verbose=false: 進捗バー抑制) ---
    GalloitBC bcComputer(csr, batchSize, false);

    // --- フェーズ別計測付きバッチループ ---
    cudaEvent_t ev_bfs_start, ev_bfs_end, ev_back_end;
    CUDA_ERR_CHK(cudaEventCreate(&ev_bfs_start));
    CUDA_ERR_CHK(cudaEventCreate(&ev_bfs_end));
    CUDA_ERR_CHK(cudaEventCreate(&ev_back_end));

    float total_bfs_ms = 0.0f, total_back_ms = 0.0f;

    for (int b = 0; b < numBatches; b++) {
        int start = b * batchSize;
        int end   = std::min(start + batchSize, n_nodes);

        std::vector<int> sources;
        sources.reserve(end - start);
        for (int i = start; i < end; i++) {
            sources.push_back(i);
        }

        bcComputer.initializeBatch(sources);

        // BFS (forward) フェーズ計測
        CUDA_ERR_CHK(cudaEventRecord(ev_bfs_start, 0));
        bcComputer.forwardPhase(sources);
        CUDA_ERR_CHK(cudaEventRecord(ev_bfs_end, 0));

        // Backward (dependency) フェーズ計測
        bcComputer.backwardPhase(sources);
        CUDA_ERR_CHK(cudaEventRecord(ev_back_end, 0));
        CUDA_ERR_CHK(cudaEventSynchronize(ev_back_end));

        float bfs_ms = 0.0f, back_ms = 0.0f;
        CUDA_ERR_CHK(cudaEventElapsedTime(&bfs_ms, ev_bfs_start, ev_bfs_end));
        CUDA_ERR_CHK(cudaEventElapsedTime(&back_ms, ev_bfs_end, ev_back_end));
        total_bfs_ms  += bfs_ms;
        total_back_ms += back_ms;
    }

    fprintf(stderr, "  > [GPU Phase] BFS: %.4f sec, Backward: %.4f sec\n",
            total_bfs_ms / 1000.0f, total_back_ms / 1000.0f);

    // --- 結果取得: 無向グラフ補正 (/2) ---
    // Brandes on undirected graph double-counts each path (both edge directions).
    const double* d_bc = bcComputer.getBC();
    std::vector<double> result(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        result[i] = d_bc[i] / 2.0;
    }

    // --- クリーンアップ ---
    CUDA_ERR_CHK(cudaEventDestroy(ev_bfs_start));
    CUDA_ERR_CHK(cudaEventDestroy(ev_bfs_end));
    CUDA_ERR_CHK(cudaEventDestroy(ev_back_end));
    // csr は GalloitBC のデストラクタでは解放されないので手動で解放
    delete csr;

    return result;
}

// ============================================================
//  brandes_pathmerge_bc — 共通インターフェース (env 変数でバッチ指定)
//    PATHMERGE_BC_BATCH_SIZE で上書き可能 (既定 64, 上限なし=メモリ上限まで)
// ============================================================
std::vector<double> brandes_pathmerge_bc(Graph &graph) {
    int batchSize = 64;  // 既定値
    const char* env_batch = std::getenv("PATHMERGE_BC_BATCH_SIZE");
    if (env_batch) {
        int val = std::atoi(env_batch);
        if (val > 0) batchSize = val;  // 人工的な 64 上限を撤廃 (メモリ上限まで許可)
    }
    return brandes_pathmerge_bc_batch(graph, batchSize);
}
