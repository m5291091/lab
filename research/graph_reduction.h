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
// Degree-2 Chain Compression: 圧縮されたチェーンの情報
// ----------------------------------------------------------
struct CompressedChain {
    int endpointA;                    // 入力グラフでのチェーン端点 A の ID
    int endpointB;                    // 入力グラフでのチェーン端点 B の ID
    std::vector<int> internalVertices; // 入力グラフでの内部頂点列 (A→B の順)
    int pathLength;                   // 元の辺数 = internalVertices.size() + 1
};

// ----------------------------------------------------------
// Step 2: Degree-2 Chain Compression の結果
// ----------------------------------------------------------
struct Degree2CompressResult {
    ReducedGraph reducedGraph;
    std::vector<CompressedChain> chains;
};

// ----------------------------------------------------------
// Step 2: Degree-2 チェーン圧縮
//   - Degree-2 頂点の連鎖（チェーン）を単一辺に置換
//   - 純粋な Degree-2 サイクル成分はそのまま保持
//   - 圧縮後のグラフを新しい CSR 形式で返す
// ----------------------------------------------------------
Degree2CompressResult degree2Compress(const int* adjacencyListPointers,
                                      const int* adjacencyList,
                                      int nodeCount, int edgeCount);

// ----------------------------------------------------------
// 検証ヘルパー: 縮約後グラフに Degree-1 頂点がないことを確認
// ----------------------------------------------------------
bool verifyNoDegree1(const ReducedGraph& g);

// ----------------------------------------------------------
// 検証ヘルパー: 圧縮後グラフに Degree-2 チェーンがないことを確認
//   (純粋な Degree-2 サイクル成分は許容)
// ----------------------------------------------------------
bool verifyNoDegree2Chain(const ReducedGraph& g);

// ----------------------------------------------------------
// Twin Vertex Merging: 同一構造（隣接リスト一致）頂点の統合情報
// ----------------------------------------------------------
struct TwinGroup {
    int representative;          // 代表頂点 ID（入力グラフの ID）
    std::vector<int> members;    // 統合された頂点 ID 群（代表含む、入力グラフの ID）
};

// ----------------------------------------------------------
// Step 3: Twin Vertex Merging の結果
// ----------------------------------------------------------
struct TwinMergeResult {
    ReducedGraph reducedGraph;
    std::vector<TwinGroup> twinGroups;  // サイズ>=2 のグループのみ格納
};

// ----------------------------------------------------------
// Step 3: 同一構造頂点の統合（Identical/Twin Vertex Merging）
//   - 隣接リストが完全一致する頂点群を代表頂点に統合
//   - 統合情報を記録し、新しい CSR 形式で返す
// ----------------------------------------------------------
TwinMergeResult twinMerge(const int* adjacencyListPointers,
                          const int* adjacencyList,
                          int nodeCount, int edgeCount);

// ----------------------------------------------------------
// 検証ヘルパー: 統合後グラフに同一構造頂点が存在しないことを確認
// ----------------------------------------------------------
bool verifyNoTwins(const ReducedGraph& g);
