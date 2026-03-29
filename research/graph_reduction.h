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

// ============================================================
// 安全条件付き Degree-2 チェーン圧縮
// ============================================================

// ----------------------------------------------------------
// 安全条件チェック結果を含む Degree-2 圧縮結果
//   - safeChains: 安全条件を満たし、実際に圧縮されたチェーン
//   - unsafeChains: 安全条件を満たさず、スキップされたチェーン
// ----------------------------------------------------------
struct SafeDegree2CompressResult {
    ReducedGraph reducedGraph;
    std::vector<CompressedChain> safeChains;
    std::vector<CompressedChain> unsafeChains;
};

// ----------------------------------------------------------
// 安全条件付き Degree-2 チェーン圧縮
//   安全条件 (Sariyüce 2013):
//     C1: a ≠ b (サイクルでない)
//     C2: a-b 間に直接辺がない
//     C3: a と b に {v₁...vₖ} 以外の共通隣接頂点がない
//   安全でないチェーンは圧縮せず、元のまま残す
// ----------------------------------------------------------
SafeDegree2CompressResult safeDegree2Compress(
    const int* adjacencyListPointers,
    const int* adjacencyList,
    int nodeCount, int edgeCount);

// ============================================================
// 重み付き縮約グラフ（CSR 形式 + 辺重み）
// ============================================================
struct WeightedReducedGraph {
    int nodeCount;
    int edgeCount;
    std::vector<int> adjacencyListPointers;  // CSR row pointers  (size: nodeCount+1)
    std::vector<int> adjacencyList;          // CSR column indices (size: 2*edgeCount)
    std::vector<int> edgeWeight;             // edgeWeight[i] = adjacencyList[i] のホップ数

    // 頂点 ID マッピング
    std::vector<int> newToOrig;  // newToOrig[new_id] = original_id
    std::vector<int> origToNew;  // origToNew[orig_id] = new_id  (-1 if removed)
};

// ============================================================
// Step 4: 縮約グラフ上での厳密 BC 計算 (CPU Brandes)
// ============================================================

// ----------------------------------------------------------
// ReducedGraph 上で Brandes 法 (逐次, 無重み) を実行し、BC 値を返す
//   - 返り値は縮約グラフの新 ID 順 (size = reducedGraph.nodeCount)
// ----------------------------------------------------------
std::vector<double> computeBCOnReducedGraph(const ReducedGraph& rg);

// ----------------------------------------------------------
// 辺依存性の結果 (Brandes back-propagation 中に記録)
//   edgeDep[i] = CSR adjacencyList[i] に対応する辺の依存性合計
// ----------------------------------------------------------
struct WeightedBrandesResult {
    std::vector<double> bc;       // 各頂点の BC 値 (size: nodeCount)
    std::vector<double> edgeDep;  // 各辺の依存性 (size: adjacencyList.size())
};

// ----------------------------------------------------------
// WeightedReducedGraph 上で重み付き Brandes 法 (Dial's algorithm) を実行
//   - 辺重みは正の整数
//   - 辺依存性も記録 (Degree-2 復元に必要)
//   - 返り値は BC 値 + 辺依存性
// ----------------------------------------------------------
WeightedBrandesResult computeWeightedBCWithEdgeDep(const WeightedReducedGraph& wg);

// ============================================================
// Step 5: BC 復元（逆変換）
// ============================================================

// ----------------------------------------------------------
// Step 1 逆変換: Degree-1 頂点の BC 復元 (BFS ベース正確復元)
//   - peeledStack を逆順に走査し、各頂点を順に復元
//   - 各復元で parent p から BFS+back-prop を実行して依存性を計算
//   - BC(v) = 0, BC(p) += (n-2), BC(u) += δ_p(u)
// ----------------------------------------------------------
void restoreDegree1BC(std::vector<double>& bc,
                      const Degree1PeelResult& peelResult,
                      int origNodeCount);

// ----------------------------------------------------------
// Step 2 逆変換: Degree-2 チェーンの BC 復元 (Bentert 2018)
//   - 安全条件を満たすチェーンの内部頂点 BC を辺依存性から復元
//   - チェーン内部全頂点は同一 BC 値を持つ
//   - 端点 a, b の BC 値は重み付き Brandes で正しく計算済み
// ----------------------------------------------------------
void restoreDegree2BC(std::vector<double>& bc,
                      const SafeDegree2CompressResult& compResult,
                      const WeightedBrandesResult& brandesResult,
                      const WeightedReducedGraph& wg,
                      int origNodeCount);

// ============================================================
// Step 6: End-to-End パイプライン
// ============================================================

// ----------------------------------------------------------
// End-to-End パイプライン (Degree-1 + Degree-2 統合版):
//   Step 1 (Degree-1 Peeling) → Step 2 (安全 Degree-2 圧縮) →
//   重み付き Brandes → Degree-2 復元 → Degree-1 復元 → 全頂点 BC
//   返り値は元グラフの全頂点に対する正確な BC 値
// ----------------------------------------------------------
std::vector<double> runReductionPipeline(const int* adjacencyListPointers,
                                         const int* adjacencyList,
                                         int nodeCount, int edgeCount);
