#ifndef BRANDES_GPU_HPP
#define BRANDES_GPU_HPP

#include <vector>

#include "graph.hpp"

// ============================================================
//  提案手法 (GH200 Grace Hopper 特化) の宣言シーム
//  すべて共通シグネチャ std::vector<double>(Graph&) を持つ。
// ============================================================

// Unified Memory + ReadMostly + 適応配置 + cudaMemsetAsync + 2ストリーム
std::vector<double> brandes_gpu_opt(Graph &G);
// 純 GPU メモリ版 (UM 不使用): cudaMalloc + cudaMemcpy
std::vector<double> brandes_gpu_opt_pure(Graph &G);
// 純 GPU メモリ版 + 手動チャンク: SUB_BATCH×N を outer ループで反復
std::vector<double> brandes_gpu_opt_pure_chunked(Graph &G);

#endif // BRANDES_GPU_HPP
