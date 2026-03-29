#pragma once

#include <vector>
#include <string>
#include <iostream>

// ============================================================
// Graph Reduction (Method 1) — データ構造と関数宣言
// ============================================================

// ----------------------------------------------------------
// Degree-1 Peeling: 除去された頂点の情報
// ----------------------------------------------------------
struct PeeledVertex {
    int originalId;  // 元グラフでの頂点 ID
    int neighborId;  // 除去時点での（唯一の）隣接頂点 ID（元グラフの ID）
};

// ----------------------------------------------------------
// 縮約後のグラフ（CSR 形式）
// ----------------------------------------------------------
struct ReducedGraph {
    int nodeCount;
    int edgeCount;
    std::vector<int> adjacencyListPointers;  // CSR row pointers  (size: nodeCount+1)
    std::vector<int> adjacencyList;          // CSR column indices (size: 2*edgeCount)

    // 頂点 ID マッピング
    std::vector<int> newToOrig;  // newToOrig[new_id] = original_id
    std::vector<int> origToNew;  // origToNew[orig_id] = new_id  (-1 if removed)
};

// ----------------------------------------------------------
// Step 1: Degree-1 Peeling の結果
// ----------------------------------------------------------
struct Degree1PeelResult {
    ReducedGraph reducedGraph;

    // 除去順にプッシュ（peeledStack[0] = 最初に除去された頂点）
    // 復元（Step 5）ではこの配列を逆順に走査する
    std::vector<PeeledVertex> peeledStack;
};

// ----------------------------------------------------------
// Step 1: Degree-1 頂点除去（Peeling）
//   - 反復的に Degree-1 頂点を除去し、除去順序を記録
//   - 除去後のグラフを新しい CSR 形式で返す
// ----------------------------------------------------------
Degree1PeelResult degree1Peel(const int* adjacencyListPointers,
                              const int* adjacencyList,
                              int nodeCount, int edgeCount);

// ----------------------------------------------------------
// 検証ヘルパー: 縮約後グラフに Degree-1 頂点がないことを確認
// ----------------------------------------------------------
bool verifyNoDegree1(const ReducedGraph& g);
