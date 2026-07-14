#ifndef BRANDES_BASELINE_HPP
#define BRANDES_BASELINE_HPP

#include <vector>

#include "graph.hpp"

// ============================================================
//  ベースライン実装の宣言シーム
//  すべて共通シグネチャ std::vector<double>(Graph&) を持つ。
// ============================================================

// CPU 逐次 Brandes
std::vector<double> brandes_sequential(Graph &G);
// CPU OpenMP 並列 Brandes
std::vector<double> brandes_omp(Graph &G);
// 素朴 GPU 実装 (gpu_unopt)
std::vector<double> brandes_gpu(Graph &G);
// RAPIDS cuGraph Exact BC ラッパー
std::vector<double> brandes_cugraph_bc(Graph &G);
// Galliot path-merging バッチ並列 BFS BC (gobardhanm/path-merging-bc)
std::vector<double> brandes_pathmerge_bc(Graph &G);
// 同上・バッチサイズを明示指定 (可変バッチサイズ探索用, 64 上限なし)
std::vector<double> brandes_pathmerge_bc_batch(Graph &G, int batchSize);

#endif // BRANDES_BASELINE_HPP
