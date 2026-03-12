#ifndef BRANDES_H
#define BRANDES_H

#include <vector>
#include "Graph.h"

// Function prototypes for different Brandes algorithm implementations
std::vector<double> brandes_sequential(Graph &graph);
std::vector<double> brandes_omp(Graph &graph);
std::vector<double> brandes_gpu(Graph &graph);
// GH200 Unified Memory 最適化版 (フェーズ2)
std::vector<double> brandes_gpu_managed(Graph &graph);
// GH200 中間版: ReadMostly + 適応型配置のみ (2-stream なし) — アブレーション手法1
std::vector<double> brandes_gpu_readmostly(Graph &graph);
// GH200 高速化版: ReadMostly + 適応型配置 + cudaMemsetAsync + 2ストリーム (フェーズ2 改良)
std::vector<double> brandes_gpu_opt(Graph &graph);

#endif // BRANDES_H
