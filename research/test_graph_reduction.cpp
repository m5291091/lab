#include "graph_reduction.h"
#include "Graph.h"

#include <vector>
#include <queue>
#include <stack>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>

using namespace std;

// ============================================================
// テスト用: 手動で CSR グラフを構築するヘルパー
// ============================================================
struct TestGraph {
    int nodeCount;
    int edgeCount;
    vector<int> adjacencyListPointers;
    vector<int> adjacencyList;
};

// 隣接リスト (undirected) → CSR
static TestGraph buildCSR(int n, const vector<pair<int,int>>& edges) {
    TestGraph g;
    g.nodeCount = n;
    g.edgeCount = static_cast<int>(edges.size());

    // 各頂点の次数をカウント
    vector<vector<int>> adj(n);
    for (auto& e : edges) {
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
    }

    g.adjacencyListPointers.resize(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        g.adjacencyListPointers[i + 1] = g.adjacencyListPointers[i] + static_cast<int>(adj[i].size());
    }

    g.adjacencyList.resize(g.adjacencyListPointers[n]);
    vector<int> offset(n, 0);
    for (int i = 0; i < n; ++i) {
        int base = g.adjacencyListPointers[i];
        for (int j = 0; j < static_cast<int>(adj[i].size()); ++j) {
            g.adjacencyList[base + j] = adj[i][j];
        }
    }

    return g;
}

// ============================================================
// テストケース 1: スターグラフ（中心 + 葉）
//   0 -- 1, 0 -- 2, 0 -- 3, 0 -- 4
//   頂点 1,2,3,4 は Degree-1 → すべて除去
//   頂点 0 は Degree-4 → 除去後 Degree-0 で残る
// ============================================================
static bool test_star_graph() {
    fprintf(stderr, "\n=== Test: Star Graph (5 vertices) ===\n");

    vector<pair<int,int>> edges = {{0,1}, {0,2}, {0,3}, {0,4}};
    TestGraph tg = buildCSR(5, edges);

    Degree1PeelResult result = degree1Peel(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    // 4 頂点が除去されるはず
    if (result.peeledStack.size() != 4) {
        fprintf(stderr, "FAIL: Expected 4 peeled vertices, got %zu\n",
                result.peeledStack.size());
        return false;
    }

    // 残りは 1 頂点（中心の 0）
    if (result.reducedGraph.nodeCount != 1) {
        fprintf(stderr, "FAIL: Expected 1 remaining vertex, got %d\n",
                result.reducedGraph.nodeCount);
        return false;
    }

    // Degree-1 頂点がないことを確認
    if (!verifyNoDegree1(result.reducedGraph)) {
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース 2: パスグラフ 0-1-2-3-4
//   反復的 peeling で 0,4 → 1,3 → 2 (Degree-0) が残る
// ============================================================
static bool test_path_graph() {
    fprintf(stderr, "\n=== Test: Path Graph (5 vertices: 0-1-2-3-4) ===\n");

    vector<pair<int,int>> edges = {{0,1}, {1,2}, {2,3}, {3,4}};
    TestGraph tg = buildCSR(5, edges);

    Degree1PeelResult result = degree1Peel(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    // 4 頂点が除去されるはず (0,4 → 1,3)
    if (result.peeledStack.size() != 4) {
        fprintf(stderr, "FAIL: Expected 4 peeled vertices, got %zu\n",
                result.peeledStack.size());
        return false;
    }

    // 残りは 1 頂点（中央の 2）
    if (result.reducedGraph.nodeCount != 1) {
        fprintf(stderr, "FAIL: Expected 1 remaining vertex, got %d\n",
                result.reducedGraph.nodeCount);
        return false;
    }

    if (!verifyNoDegree1(result.reducedGraph)) {
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース 3: 三角形 + ペンダント
//   0-1, 1-2, 2-0 (三角形) + 0-3, 1-4 (ペンダント)
//   頂点 3,4 が Degree-1 → 除去
//   頂点 0,1,2 は Degree>=2 で残る
// ============================================================
static bool test_triangle_with_pendants() {
    fprintf(stderr, "\n=== Test: Triangle + Pendants ===\n");

    vector<pair<int,int>> edges = {{0,1}, {1,2}, {2,0}, {0,3}, {1,4}};
    TestGraph tg = buildCSR(5, edges);

    Degree1PeelResult result = degree1Peel(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    // 2 頂点 (3, 4) が除去されるはず
    if (result.peeledStack.size() != 2) {
        fprintf(stderr, "FAIL: Expected 2 peeled vertices, got %zu\n",
                result.peeledStack.size());
        return false;
    }

    // 残りは 3 頂点
    if (result.reducedGraph.nodeCount != 3) {
        fprintf(stderr, "FAIL: Expected 3 remaining vertices, got %d\n",
                result.reducedGraph.nodeCount);
        return false;
    }

    // 残りグラフは三角形 (3 頂点, 3 辺)
    if (result.reducedGraph.edgeCount != 3) {
        fprintf(stderr, "FAIL: Expected 3 edges, got %d\n",
                result.reducedGraph.edgeCount);
        return false;
    }

    if (!verifyNoDegree1(result.reducedGraph)) {
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース 4: 完全グラフ K4
//   全頂点が Degree-3 → 除去なし
// ============================================================
static bool test_complete_graph() {
    fprintf(stderr, "\n=== Test: Complete Graph K4 ===\n");

    vector<pair<int,int>> edges = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
    TestGraph tg = buildCSR(4, edges);

    Degree1PeelResult result = degree1Peel(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    // 除去なし
    if (result.peeledStack.size() != 0) {
        fprintf(stderr, "FAIL: Expected 0 peeled vertices, got %zu\n",
                result.peeledStack.size());
        return false;
    }

    if (result.reducedGraph.nodeCount != 4) {
        fprintf(stderr, "FAIL: Expected 4 remaining vertices, got %d\n",
                result.reducedGraph.nodeCount);
        return false;
    }

    if (!verifyNoDegree1(result.reducedGraph)) {
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース 5: 木グラフ（全頂点が最終的に除去される）
//   vertex 0 connects to 1, 2
//   vertex 1 connects to 0, 3, 4
// ============================================================
static bool test_tree_graph() {
    fprintf(stderr, "\n=== Test: Tree Graph ===\n");

    vector<pair<int,int>> edges = {{0,1}, {0,2}, {1,3}, {1,4}};
    TestGraph tg = buildCSR(5, edges);

    Degree1PeelResult result = degree1Peel(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    // 全頂点除去 (2,3,4 → 0,1 → ... 最後に残るのは?)
    // 2 は degree-1 → 除去, 3 は degree-1 → 除去, 4 は degree-1 → 除去
    // 0: degree 2→1 → 除去, 1: degree 3→2→1 → 除去
    // 実際は: 2,3,4 除去 → 0 は degree 1 → 除去 → 1 は degree 0 → 残る
    // ただし 1 は degree 3 → 2,3,4 除去後 degree 0 にはならない
    // 2 除去 → 0 の degree: 2→1 → queue に入る
    // 3 除去 → 1 の degree: 3→2
    // 4 除去 → 1 の degree: 2→1 → queue に入る
    // 0 除去 (degree 1, neighbor = 1) → 1 の degree: 1→0
    // 1 除去 (degree 1, neighbor = ?) 
    // wait, 1's degree is 1→0 after 0 is removed. So 1 has degree 0 and stays.
    // Let me recalculate:
    // Initial degrees: 0:2, 1:3, 2:1, 3:1, 4:1
    // Queue: [2, 3, 4]
    // Remove 2 (neighbor=0): degree[0] = 2→1, push 0
    // Remove 3 (neighbor=1): degree[1] = 3→2
    // Remove 4 (neighbor=1): degree[1] = 2→1, push 1
    // Queue: [0, 1]
    // Remove 0 (degree=1, find non-removed neighbor): neighbor=1, degree[1] = 1→0
    // Remove 1 (degree=1, find non-removed neighbor): no non-removed neighbor → skip
    // Wait, 1 has degree 0 now, not 1. So when we dequeue 1, degree[1]=0 != 1, skip.
    // Result: 4 vertices removed (2,3,4,0), 1 vertex remains (1 with degree 0)

    if (result.peeledStack.size() != 4) {
        fprintf(stderr, "FAIL: Expected 4 peeled vertices, got %zu\n",
                result.peeledStack.size());
        return false;
    }

    if (result.reducedGraph.nodeCount != 1) {
        fprintf(stderr, "FAIL: Expected 1 remaining vertex, got %d\n",
                result.reducedGraph.nodeCount);
        return false;
    }

    if (!verifyNoDegree1(result.reducedGraph)) {
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース 6: ID マッピングの検証
// ============================================================
static bool test_id_mapping() {
    fprintf(stderr, "\n=== Test: ID Mapping Consistency ===\n");

    // 三角形 (0,1,2) + ペンダント (0-3)
    vector<pair<int,int>> edges = {{0,1}, {1,2}, {2,0}, {0,3}};
    TestGraph tg = buildCSR(4, edges);

    Degree1PeelResult result = degree1Peel(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    const ReducedGraph& rg = result.reducedGraph;

    // 頂点 3 が除去、0,1,2 が残る
    if (rg.nodeCount != 3) {
        fprintf(stderr, "FAIL: Expected 3 remaining vertices, got %d\n",
                rg.nodeCount);
        return false;
    }

    // origToNew と newToOrig の整合性チェック
    for (int newV = 0; newV < rg.nodeCount; ++newV) {
        int origV = rg.newToOrig[newV];
        if (rg.origToNew[origV] != newV) {
            fprintf(stderr, "FAIL: Mapping inconsistency at newV=%d, origV=%d\n",
                    newV, origV);
            return false;
        }
    }

    // 除去された頂点の origToNew は -1
    if (rg.origToNew[3] != -1) {
        fprintf(stderr, "FAIL: Removed vertex 3 should map to -1, got %d\n",
                rg.origToNew[3]);
        return false;
    }

    // 縮約グラフの隣接リストが有効な新 ID のみを含むことを確認
    for (int i = 0; i < static_cast<int>(rg.adjacencyList.size()); ++i) {
        int v = rg.adjacencyList[i];
        if (v < 0 || v >= rg.nodeCount) {
            fprintf(stderr, "FAIL: Invalid vertex ID %d in adjacency list\n", v);
            return false;
        }
    }

    if (!verifyNoDegree1(rg)) {
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース 7: ファイルからグラフを読み込んでテスト
// ============================================================
static bool test_with_file(const char* filepath) {
    fprintf(stderr, "\n=== Test: File Graph (%s) ===\n", filepath);

    if (freopen(filepath, "r", stdin) == nullptr) {
        fprintf(stderr, "SKIP: Could not open file %s\n", filepath);
        return true;  // ファイルがない場合はスキップ
    }

    Graph graph;
    graph.readGraph();
    fclose(stdin);

    int n = graph.getNodeCount();
    int m = graph.getEdgeCount();
    fprintf(stderr, "  Original graph: %d vertices, %d edges\n", n, m);

    Degree1PeelResult result = degree1Peel(
        graph.getAdjacencyListPointers(),
        graph.getAdjacencyList(),
        n, m);

    if (!verifyNoDegree1(result.reducedGraph)) {
        return false;
    }

    // 基本的な整合性チェック
    const ReducedGraph& rg = result.reducedGraph;
    int peeled = static_cast<int>(result.peeledStack.size());
    if (rg.nodeCount + peeled != n) {
        fprintf(stderr, "FAIL: nodeCount(%d) + peeled(%d) != original(%d)\n",
                rg.nodeCount, peeled, n);
        return false;
    }

    // 辺数チェック: 縮約グラフの辺数 + 除去された辺数 == 元の辺数
    // 除去された辺数 == peeled (各 degree-1 頂点の除去で 1 辺が消える)
    if (rg.edgeCount + peeled != m) {
        fprintf(stderr, "FAIL: edgeCount(%d) + peeled(%d) != original(%d)\n",
                rg.edgeCount, peeled, m);
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
//                    Step 2: Degree-2 Chain Compression テスト
// ============================================================

// ============================================================
// テストケース S2-1: 三角形の各辺にチェーンを挿入
//   0-1 直接 + 0-3-1 チェーン
//   1-2 直接 + 1-4-2 チェーン
//   2-0 直接 + 2-5-0 チェーン
//   頂点 3,4,5 は Degree-2 → 圧縮
//   頂点 0,1,2 は Degree-4 → 残る
//   結果: 三角形 0-1-2 (直接辺のみ、チェーン辺は重複排除)
// ============================================================
static bool test_s2_triangle_chains() {
    fprintf(stderr, "\n=== S2 Test: Triangle with Degree-2 chains ===\n");

    // 0-1, 1-2, 2-0 (direct triangle)
    // 0-3, 3-1, 1-4, 4-2, 2-5, 5-0 (chains)
    vector<pair<int,int>> edges = {
        {0,1}, {1,2}, {2,0},
        {0,3}, {3,1}, {1,4}, {4,2}, {2,5}, {5,0}
    };
    TestGraph tg = buildCSR(6, edges);

    Degree2CompressResult result = degree2Compress(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    // 3 chains: 0-3-1, 1-4-2, 2-5-0
    if (result.chains.size() != 3) {
        fprintf(stderr, "FAIL: Expected 3 chains, got %zu\n", result.chains.size());
        return false;
    }

    // 3 vertices remain (0, 1, 2)
    if (result.reducedGraph.nodeCount != 3) {
        fprintf(stderr, "FAIL: Expected 3 remaining vertices, got %d\n",
                result.reducedGraph.nodeCount);
        return false;
    }

    // 3 edges (triangle)
    if (result.reducedGraph.edgeCount != 3) {
        fprintf(stderr, "FAIL: Expected 3 edges, got %d\n",
                result.reducedGraph.edgeCount);
        return false;
    }

    if (!verifyNoDegree2Chain(result.reducedGraph)) {
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース S2-2: 完全グラフ K4（Degree-2 頂点なし → 変化なし）
// ============================================================
static bool test_s2_complete_graph() {
    fprintf(stderr, "\n=== S2 Test: Complete Graph K4 (no compression) ===\n");

    vector<pair<int,int>> edges = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
    TestGraph tg = buildCSR(4, edges);

    Degree2CompressResult result = degree2Compress(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    if (result.chains.size() != 0) {
        fprintf(stderr, "FAIL: Expected 0 chains, got %zu\n", result.chains.size());
        return false;
    }

    if (result.reducedGraph.nodeCount != 4) {
        fprintf(stderr, "FAIL: Expected 4 remaining vertices, got %d\n",
                result.reducedGraph.nodeCount);
        return false;
    }

    if (!verifyNoDegree2Chain(result.reducedGraph)) {
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース S2-3: 純粋な Degree-2 サイクル (0-1-2-3-0)
// 全頂点 Degree-2 → サイクル → 圧縮しない（保持）
// ============================================================
static bool test_s2_pure_cycle() {
    fprintf(stderr, "\n=== S2 Test: Pure Degree-2 Cycle ===\n");

    vector<pair<int,int>> edges = {{0,1}, {1,2}, {2,3}, {3,0}};
    TestGraph tg = buildCSR(4, edges);

    Degree2CompressResult result = degree2Compress(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    // 圧縮なし（純粋サイクル）
    if (result.chains.size() != 0) {
        fprintf(stderr, "FAIL: Expected 0 chains, got %zu\n", result.chains.size());
        return false;
    }

    if (result.reducedGraph.nodeCount != 4) {
        fprintf(stderr, "FAIL: Expected 4 remaining vertices, got %d\n",
                result.reducedGraph.nodeCount);
        return false;
    }

    if (!verifyNoDegree2Chain(result.reducedGraph)) {
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース S2-4: 長いチェーン (ハブ2つ + 中間パス)
//   0(hub, deg3) - 2 - 3 - 4 - 1(hub, deg3)
//   0-5, 1-6  (extra edges to make 0,1 have deg >= 3)
//   5-6 (to connect)
// ============================================================
static bool test_s2_long_chain() {
    fprintf(stderr, "\n=== S2 Test: Long Degree-2 Chain ===\n");

    // 0-2, 2-3, 3-4, 4-1  (chain: 0-2-3-4-1)
    // 0-5, 1-6, 5-6        (make 0,1 non-degree-2; 5,6 also non-degree-2)
    // Also: 0-1 NOT directly connected
    vector<pair<int,int>> edges = {
        {0,2}, {2,3}, {3,4}, {4,1},
        {0,5}, {1,6}, {5,6}
    };
    TestGraph tg = buildCSR(7, edges);

    // degrees: 0:2, 1:2, 2:2, 3:2, 4:2, 5:2, 6:2
    // Hmm, all degree-2! That's a long path 5-0-2-3-4-1-6-5? No, that's a cycle.
    // Wait: 0 has neighbors [2,5], 1 has neighbors [4,6], 5 has neighbors [0,6], 6 has neighbors [1,5]
    // All degree-2. This forms a cycle: 0-2-3-4-1-6-5-0
    // Need to add more edges to make some vertices non-degree-2

    // Let me redesign: add edge 0-1 to make 0 and 1 degree-3
    // edges: 0-2, 2-3, 3-4, 4-1, 0-5, 1-6, 5-6, 0-1
    fprintf(stderr, "  (Redesigning: adding 0-1 edge)\n");

    vector<pair<int,int>> edges2 = {
        {0,2}, {2,3}, {3,4}, {4,1},
        {0,5}, {1,6}, {5,6}, {0,1}
    };
    TestGraph tg2 = buildCSR(7, edges2);
    // degrees: 0:[2,5,1]=3, 1:[4,6,0]=3, 2:[0,3]=2, 3:[2,4]=2, 4:[3,1]=2, 5:[0,6]=2, 6:[1,5]=2

    Degree2CompressResult result = degree2Compress(
        tg2.adjacencyListPointers.data(),
        tg2.adjacencyList.data(),
        tg2.nodeCount, tg2.edgeCount);

    // Chains: 0-2-3-4-1 (internal: 2,3,4), 0-5-6-1 (internal: 5,6)
    if (result.chains.size() != 2) {
        fprintf(stderr, "FAIL: Expected 2 chains, got %zu\n", result.chains.size());
        return false;
    }

    // 2 vertices remain (0, 1)
    if (result.reducedGraph.nodeCount != 2) {
        fprintf(stderr, "FAIL: Expected 2 remaining vertices, got %d\n",
                result.reducedGraph.nodeCount);
        return false;
    }

    if (!verifyNoDegree2Chain(result.reducedGraph)) {
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース S2-5: Step 1 + Step 2 の連続適用
//   三角形 (0,1,2) + ペンダント (0-3) + チェーン (1-4-5-2)
//   Step1: 頂点 3 を除去 → 0 は degree 3→2
//   Step2: 0 は degree-2 (chain: 1-0-2), 4,5 は degree-2 (chain: 1-4-5-2)
//     → 圧縮後: 頂点 1,2 のみ残る
// ============================================================
static bool test_s2_pipeline_step1_step2() {
    fprintf(stderr, "\n=== S2 Test: Step1 + Step2 Pipeline ===\n");

    // 0-1, 1-2, 2-0  (triangle)
    // 0-3             (pendant)
    // 1-4, 4-5, 5-2   (chain bypassing 1-2 edge)
    vector<pair<int,int>> edges = {
        {0,1}, {1,2}, {2,0},
        {0,3},
        {1,4}, {4,5}, {5,2}
    };
    TestGraph tg = buildCSR(6, edges);

    // Step 1: Degree-1 peeling
    Degree1PeelResult s1 = degree1Peel(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    if (!verifyNoDegree1(s1.reducedGraph)) {
        return false;
    }

    // After step1: vertex 3 removed. Remaining: 0,1,2,4,5 (5 vertices)
    if (s1.reducedGraph.nodeCount != 5) {
        fprintf(stderr, "FAIL: After Step1, expected 5 vertices, got %d\n",
                s1.reducedGraph.nodeCount);
        return false;
    }

    // Step 2: Degree-2 chain compression
    const ReducedGraph& g1 = s1.reducedGraph;
    Degree2CompressResult s2 = degree2Compress(
        g1.adjacencyListPointers.data(),
        g1.adjacencyList.data(),
        g1.nodeCount, g1.edgeCount);

    if (!verifyNoDegree2Chain(s2.reducedGraph)) {
        return false;
    }

    // After step2: vertex 0 (now deg-2), 4, 5 are degree-2 chains
    // Chains: 1-0-2 and 1-4-5-2 → endpoints 1,2 already adjacent
    // Result: vertices 1,2 remain (2 vertices, 1 edge)
    if (s2.reducedGraph.nodeCount != 2) {
        fprintf(stderr, "FAIL: After Step2, expected 2 vertices, got %d\n",
                s2.reducedGraph.nodeCount);
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース S2-6: チェーン情報の整合性検証
// ============================================================
static bool test_s2_chain_info() {
    fprintf(stderr, "\n=== S2 Test: Chain Info Consistency ===\n");

    // Hub 0 (deg 4) connected to chains ending at Hub 1 (deg 4)
    // 0-2-3-1, 0-4-1, 0-1 direct
    // Additional: 0-5, 1-6, 5-6 to increase hub degrees
    vector<pair<int,int>> edges = {
        {0,2}, {2,3}, {3,1},    // chain: 0-2-3-1
        {0,4}, {4,1},           // chain: 0-4-1
        {0,1},                  // direct edge
        {0,5}, {1,6}, {5,6}    // extra to boost degrees
    };
    TestGraph tg = buildCSR(7, edges);
    // Degrees: 0:[2,4,1,5]=4, 1:[3,4,0,6]=4, 2:[0,3]=2, 3:[2,1]=2, 4:[0,1]=2, 5:[0,6]=2, 6:[1,5]=2

    Degree2CompressResult result = degree2Compress(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    // Chains should include: 0-2-3-1, 0-4-1, 0-5-6-1
    // All chains have endpoints 0 and 1
    for (const auto& chain : result.chains) {
        fprintf(stderr, "  Chain: %d -> [", chain.endpointA);
        for (size_t i = 0; i < chain.internalVertices.size(); ++i) {
            if (i > 0) fprintf(stderr, ",");
            fprintf(stderr, "%d", chain.internalVertices[i]);
        }
        fprintf(stderr, "] -> %d (len=%d)\n", chain.endpointB, chain.pathLength);

        // Each chain's path length should equal internal vertices + 1
        if (chain.pathLength != static_cast<int>(chain.internalVertices.size()) + 1) {
            fprintf(stderr, "FAIL: pathLength mismatch\n");
            return false;
        }
    }

    // After compression: only vertices 0 and 1 remain
    if (result.reducedGraph.nodeCount != 2) {
        fprintf(stderr, "FAIL: Expected 2 remaining vertices, got %d\n",
                result.reducedGraph.nodeCount);
        return false;
    }

    if (!verifyNoDegree2Chain(result.reducedGraph)) {
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
//                    Step 3: Twin Vertex Merging テスト
// ============================================================

// ============================================================
// テストケース S3-1: 明確な open twin
//   0 -- 2, 0 -- 3
//   1 -- 2, 1 -- 3
//   頂点 0 と 1 は同じ隣接リスト {2,3} → twin
//   頂点 2 と 3 は同じ隣接リスト {0,1} → twin
//   2 グループ統合 → 2 頂点残り
// ============================================================
static bool test_s3_open_twins() {
    fprintf(stderr, "\n=== S3 Test: Open Twins ===\n");

    vector<pair<int,int>> edges = {{0,2}, {0,3}, {1,2}, {1,3}};
    TestGraph tg = buildCSR(4, edges);

    TwinMergeResult result = twinMerge(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    // 2 twin groups: {0,1} and {2,3}
    if (result.twinGroups.size() != 2) {
        fprintf(stderr, "FAIL: Expected 2 twin groups, got %zu\n",
                result.twinGroups.size());
        return false;
    }

    // 2 vertices remain (representatives)
    if (result.reducedGraph.nodeCount != 2) {
        fprintf(stderr, "FAIL: Expected 2 remaining vertices, got %d\n",
                result.reducedGraph.nodeCount);
        return false;
    }

    if (!verifyNoTwins(result.reducedGraph)) {
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース S3-2: twin なし (完全グラフ K4)
//   全頂点が全他頂点に隣接 → 隣接リストは異なる（自分を含まないため）
//   実際: vertex 0 neighbors = {1,2,3}, vertex 1 neighbors = {0,2,3}
//   → 0 と 1 は異なる隣接リスト → twin ではない
// ============================================================
static bool test_s3_no_twins() {
    fprintf(stderr, "\n=== S3 Test: No Twins (K4) ===\n");

    vector<pair<int,int>> edges = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
    TestGraph tg = buildCSR(4, edges);

    TwinMergeResult result = twinMerge(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    if (result.twinGroups.size() != 0) {
        fprintf(stderr, "FAIL: Expected 0 twin groups, got %zu\n",
                result.twinGroups.size());
        return false;
    }

    if (result.reducedGraph.nodeCount != 4) {
        fprintf(stderr, "FAIL: Expected 4 remaining vertices, got %d\n",
                result.reducedGraph.nodeCount);
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース S3-3: 3頂点の open twin グループ + 2頂点グループ
//   0,1,2 all connect to {3,4} → 3-way twin
//   3,4 both connect to {0,1,2} → 2-way twin
//   結果: 2 twin groups, 2 vertices remain
// ============================================================
static bool test_s3_triple_twins() {
    fprintf(stderr, "\n=== S3 Test: Triple Open Twins ===\n");

    vector<pair<int,int>> edges = {
        {0,3}, {0,4},
        {1,3}, {1,4},
        {2,3}, {2,4}
    };
    TestGraph tg = buildCSR(5, edges);

    TwinMergeResult result = twinMerge(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    // 2 twin groups: {0,1,2} and {3,4}
    if (result.twinGroups.size() != 2) {
        fprintf(stderr, "FAIL: Expected 2 twin groups, got %zu\n",
                result.twinGroups.size());
        return false;
    }

    // 2 vertices remain
    if (result.reducedGraph.nodeCount != 2) {
        fprintf(stderr, "FAIL: Expected 2 remaining vertices, got %d\n",
                result.reducedGraph.nodeCount);
        return false;
    }

    if (!verifyNoTwins(result.reducedGraph)) {
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース S3-4: Step1 + Step2 + Step3 パイプライン
// ============================================================
static bool test_s3_full_pipeline() {
    fprintf(stderr, "\n=== S3 Test: Full Pipeline (Step1+Step2+Step3) ===\n");

    // グラフ: ハブ H(0) に 2 本のペンダント(1,2)と 2 本のチェーン(3-4-5, 6-7-8) が接続
    // ハブ H2(9) に同じチェーンの他端が接続、さらに twin 候補を作る
    // 簡単化: ハブ 0 と 9, チェーン 0-3-4-9, 0-5-6-9, ペンダント 0-1, 0-2
    vector<pair<int,int>> edges = {
        {0,1}, {0,2},           // pendants
        {0,3}, {3,4}, {4,9},    // chain 1
        {0,5}, {5,6}, {6,9},    // chain 2
        {0,9}                   // direct edge
    };
    TestGraph tg = buildCSR(10, edges);

    // Step 1: Degree-1 peeling
    fprintf(stderr, "  --- Step 1 ---\n");
    Degree1PeelResult s1 = degree1Peel(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    if (!verifyNoDegree1(s1.reducedGraph)) return false;

    // Step 2: Degree-2 chain compression
    fprintf(stderr, "  --- Step 2 ---\n");
    const ReducedGraph& g1 = s1.reducedGraph;
    Degree2CompressResult s2 = degree2Compress(
        g1.adjacencyListPointers.data(),
        g1.adjacencyList.data(),
        g1.nodeCount, g1.edgeCount);

    if (!verifyNoDegree2Chain(s2.reducedGraph)) return false;

    // Step 3: Twin merge
    fprintf(stderr, "  --- Step 3 ---\n");
    const ReducedGraph& g2 = s2.reducedGraph;
    TwinMergeResult s3 = twinMerge(
        g2.adjacencyListPointers.data(),
        g2.adjacencyList.data(),
        g2.nodeCount, g2.edgeCount);

    if (!verifyNoTwins(s3.reducedGraph)) return false;

    fprintf(stderr, "  Final graph: %d vertices, %d edges\n",
            s3.reducedGraph.nodeCount, s3.reducedGraph.edgeCount);

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
//           Step 4-6: BC 計算・復元・パイプライン テスト
// ============================================================

// リファレンス Brandes 法（テスト用、test_graph_reduction.cpp 内蔵）
static vector<double> referenceBrandes(const int* ap, const int* adj, int n) {
    vector<double> bc(n, 0.0);
    if (n <= 1) return bc;

    for (int s = 0; s < n; ++s) {
        vector<int> S;
        S.reserve(n);
        vector<vector<int>> pred(n);
        vector<long long> sigma(n, 0);
        vector<int> dist(n, -1);
        vector<double> delta(n, 0.0);

        dist[s] = 0;
        sigma[s] = 1;
        queue<int> Q;
        Q.push(s);

        while (!Q.empty()) {
            int v = Q.front(); Q.pop();
            S.push_back(v);
            for (int i = ap[v]; i < ap[v + 1]; ++i) {
                int w = adj[i];
                if (dist[w] < 0) { Q.push(w); dist[w] = dist[v] + 1; }
                if (dist[w] == dist[v] + 1) { sigma[w] += sigma[v]; pred[w].push_back(v); }
            }
        }
        for (int i = static_cast<int>(S.size()) - 1; i >= 0; --i) {
            int w = S[i];
            for (int v : pred[w]) {
                if (sigma[w] != 0)
                    delta[v] += (static_cast<double>(sigma[v]) / sigma[w]) * (1.0 + delta[w]);
            }
            if (w != s) bc[w] += delta[w] / 2.0;
        }
    }
    return bc;
}

// BC 比較ヘルパー: 最大許容誤差で比較
static bool compareBCVectors(const vector<double>& actual,
                             const vector<double>& expected,
                             double tolerance = 1e-9) {
    if (actual.size() != expected.size()) {
        fprintf(stderr, "  BC size mismatch: got %zu, expected %zu\n",
                actual.size(), expected.size());
        return false;
    }
    int mismatches = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (fabs(actual[i] - expected[i]) > tolerance) {
            if (mismatches < 10) {
                fprintf(stderr, "  BC mismatch at vertex %zu: got %.10f, expected %.10f (diff=%.2e)\n",
                        i, actual[i], expected[i], fabs(actual[i] - expected[i]));
            }
            mismatches++;
        }
    }
    if (mismatches > 0) {
        fprintf(stderr, "  Total mismatches: %d / %zu\n", mismatches, actual.size());
    }
    return mismatches == 0;
}

// ============================================================
// テストケース S4-1: 縮約グラフ上の BC 計算（三角形）
// ============================================================
static bool test_s4_bc_on_triangle() {
    fprintf(stderr, "\n=== S4 Test: BC on Triangle ===\n");

    // 三角形 0-1-2
    vector<pair<int,int>> edges = {{0,1}, {1,2}, {2,0}};
    TestGraph tg = buildCSR(3, edges);

    ReducedGraph rg;
    rg.nodeCount = tg.nodeCount;
    rg.edgeCount = tg.edgeCount;
    rg.adjacencyListPointers = tg.adjacencyListPointers;
    rg.adjacencyList = tg.adjacencyList;
    rg.newToOrig = {0, 1, 2};
    rg.origToNew = {0, 1, 2};

    vector<double> bc = computeBCOnReducedGraph(rg);

    // 三角形: 各頂点の BC = 0 (全頂点が直接接続)
    vector<double> expected = referenceBrandes(
        tg.adjacencyListPointers.data(), tg.adjacencyList.data(), 3);

    if (!compareBCVectors(bc, expected)) {
        fprintf(stderr, "FAIL\n");
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース S4-2: 縮約グラフ上の BC 計算（パスグラフ）
// ============================================================
static bool test_s4_bc_on_path() {
    fprintf(stderr, "\n=== S4 Test: BC on Path ===\n");

    // パス 0-1-2-3-4
    vector<pair<int,int>> edges = {{0,1}, {1,2}, {2,3}, {3,4}};
    TestGraph tg = buildCSR(5, edges);

    ReducedGraph rg;
    rg.nodeCount = tg.nodeCount;
    rg.edgeCount = tg.edgeCount;
    rg.adjacencyListPointers = tg.adjacencyListPointers;
    rg.adjacencyList = tg.adjacencyList;
    rg.newToOrig = {0, 1, 2, 3, 4};
    rg.origToNew = {0, 1, 2, 3, 4};

    vector<double> bc = computeBCOnReducedGraph(rg);

    vector<double> expected = referenceBrandes(
        tg.adjacencyListPointers.data(), tg.adjacencyList.data(), 5);

    if (!compareBCVectors(bc, expected)) {
        fprintf(stderr, "FAIL\n");
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース S6-1: End-to-End パイプライン（完全グラフ K4）
//   K4 は縮約不可 → パイプライン結果 == 直接計算結果
// ============================================================
static bool test_s6_pipeline_k4() {
    fprintf(stderr, "\n=== S6 Test: Pipeline on K4 (no reduction) ===\n");

    vector<pair<int,int>> edges = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
    TestGraph tg = buildCSR(4, edges);

    vector<double> pipelineBC = runReductionPipeline(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    vector<double> directBC = referenceBrandes(
        tg.adjacencyListPointers.data(), tg.adjacencyList.data(), 4);

    if (!compareBCVectors(pipelineBC, directBC)) {
        fprintf(stderr, "FAIL\n");
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース S6-2: End-to-End パイプライン（三角形 + ペンダント）
//   頂点 3,4 が除去される → 復元後の BC が直接計算と一致するか
// ============================================================
static bool test_s6_pipeline_triangle_pendants() {
    fprintf(stderr, "\n=== S6 Test: Pipeline on Triangle+Pendants ===\n");

    vector<pair<int,int>> edges = {{0,1}, {1,2}, {2,0}, {0,3}, {1,4}};
    TestGraph tg = buildCSR(5, edges);

    vector<double> pipelineBC = runReductionPipeline(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    vector<double> directBC = referenceBrandes(
        tg.adjacencyListPointers.data(), tg.adjacencyList.data(), 5);

    fprintf(stderr, "  Pipeline BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", pipelineBC[i]);
    fprintf(stderr, "\n  Direct   BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", directBC[i]);
    fprintf(stderr, "\n");

    if (!compareBCVectors(pipelineBC, directBC)) {
        fprintf(stderr, "FAIL\n");
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース S6-3: End-to-End パイプライン（スターグラフ）
//   4 つの葉が除去 → BC = 0 (葉), BC = 0 (中心, 全ペアが中心経由だが Brandes 式で BC(中心) = (n-1)(n-2)/2 ... ?)
// ============================================================
static bool test_s6_pipeline_star() {
    fprintf(stderr, "\n=== S6 Test: Pipeline on Star Graph ===\n");

    vector<pair<int,int>> edges = {{0,1}, {0,2}, {0,3}, {0,4}};
    TestGraph tg = buildCSR(5, edges);

    vector<double> pipelineBC = runReductionPipeline(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    vector<double> directBC = referenceBrandes(
        tg.adjacencyListPointers.data(), tg.adjacencyList.data(), 5);

    fprintf(stderr, "  Pipeline BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", pipelineBC[i]);
    fprintf(stderr, "\n  Direct   BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", directBC[i]);
    fprintf(stderr, "\n");

    if (!compareBCVectors(pipelineBC, directBC)) {
        fprintf(stderr, "FAIL\n");
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース S6-4: End-to-End パイプライン（パスグラフ）
// ============================================================
static bool test_s6_pipeline_path() {
    fprintf(stderr, "\n=== S6 Test: Pipeline on Path Graph ===\n");

    vector<pair<int,int>> edges = {{0,1}, {1,2}, {2,3}, {3,4}};
    TestGraph tg = buildCSR(5, edges);

    vector<double> pipelineBC = runReductionPipeline(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    vector<double> directBC = referenceBrandes(
        tg.adjacencyListPointers.data(), tg.adjacencyList.data(), 5);

    fprintf(stderr, "  Pipeline BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", pipelineBC[i]);
    fprintf(stderr, "\n  Direct   BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", directBC[i]);
    fprintf(stderr, "\n");

    if (!compareBCVectors(pipelineBC, directBC)) {
        fprintf(stderr, "FAIL\n");
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース S6-5: End-to-End パイプライン（三角形 + Degree-2 チェーン）
//   0-1, 1-2, 2-0, 0-3-4-1  (チェーン 0-3-4-1)
// ============================================================
static bool test_s6_pipeline_triangle_chain() {
    fprintf(stderr, "\n=== S6 Test: Pipeline on Triangle + Chain ===\n");

    vector<pair<int,int>> edges = {{0,1}, {1,2}, {2,0}, {0,3}, {3,4}, {4,1}};
    TestGraph tg = buildCSR(5, edges);

    vector<double> pipelineBC = runReductionPipeline(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    vector<double> directBC = referenceBrandes(
        tg.adjacencyListPointers.data(), tg.adjacencyList.data(), 5);

    fprintf(stderr, "  Pipeline BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", pipelineBC[i]);
    fprintf(stderr, "\n  Direct   BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", directBC[i]);
    fprintf(stderr, "\n");

    if (!compareBCVectors(pipelineBC, directBC)) {
        fprintf(stderr, "FAIL\n");
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース S6-6: End-to-End パイプライン（open twins）
//   0-2, 0-3, 1-2, 1-3  → 0,1 は twins, 2,3 は twins
// ============================================================
static bool test_s6_pipeline_twins() {
    fprintf(stderr, "\n=== S6 Test: Pipeline on Open Twins ===\n");

    vector<pair<int,int>> edges = {{0,2}, {0,3}, {1,2}, {1,3}};
    TestGraph tg = buildCSR(4, edges);

    vector<double> pipelineBC = runReductionPipeline(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    vector<double> directBC = referenceBrandes(
        tg.adjacencyListPointers.data(), tg.adjacencyList.data(), 4);

    fprintf(stderr, "  Pipeline BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", pipelineBC[i]);
    fprintf(stderr, "\n  Direct   BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", directBC[i]);
    fprintf(stderr, "\n");

    if (!compareBCVectors(pipelineBC, directBC)) {
        fprintf(stderr, "FAIL\n");
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース S6-7: End-to-End パイプライン（複合グラフ: ペンダント + チェーン + twin）
// ============================================================
static bool test_s6_pipeline_complex() {
    fprintf(stderr, "\n=== S6 Test: Pipeline on Complex Graph ===\n");

    // ハブ 0 と 5:
    // 0-1 (pendant), 0-2-3-5 (chain), 0-4-5 (chain), 0-5 (direct)
    // 5-6, 5-7 (pendants)
    vector<pair<int,int>> edges = {
        {0,1},              // pendant from 0
        {0,2}, {2,3}, {3,5}, // chain 0-2-3-5
        {0,4}, {4,5},       // chain 0-4-5
        {0,5},              // direct 0-5
        {5,6}, {5,7}        // pendants from 5
    };
    TestGraph tg = buildCSR(8, edges);

    vector<double> pipelineBC = runReductionPipeline(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    vector<double> directBC = referenceBrandes(
        tg.adjacencyListPointers.data(), tg.adjacencyList.data(), 8);

    fprintf(stderr, "  Pipeline BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", pipelineBC[i]);
    fprintf(stderr, "\n  Direct   BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", directBC[i]);
    fprintf(stderr, "\n");

    if (!compareBCVectors(pipelineBC, directBC)) {
        fprintf(stderr, "FAIL\n");
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
//     新規テスト: 安全 Degree-2 圧縮 + 重み付き Brandes + BC 復元
// ============================================================

// ============================================================
// テストケース SAFE-1: 安全条件チェッカーの基本動作
//   2つの三角形をチェーンで接続: 安全チェーンが1つ存在
//   Triangle1: 0-1-2-0, Chain: 2-3-4-5, Triangle2: 5-6-7-5
// ============================================================
static bool test_safe_dumbbell() {
    fprintf(stderr, "\n=== SAFE Test: Dumbbell graph (two triangles connected by chain) ===\n");

    vector<pair<int,int>> edges = {
        {0,1}, {1,2}, {2,0},     // triangle 1
        {2,3}, {3,4}, {4,5},     // chain (3,4 are degree-2)
        {5,6}, {6,7}, {7,5}      // triangle 2
    };
    TestGraph tg = buildCSR(8, edges);

    SafeDegree2CompressResult result = safeDegree2Compress(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    // 1 safe chain: 2-3-4-5 (internal: [3,4])
    if (result.safeChains.size() != 1) {
        fprintf(stderr, "FAIL: Expected 1 safe chain, got %zu\n", result.safeChains.size());
        return false;
    }

    // 三角形がサイクルチェーン (a==b) を形成するため C1 条件違反 → 2 つの非安全チェーン
    if (result.unsafeChains.size() != 2) {
        fprintf(stderr, "FAIL: Expected 2 unsafe chains (C1 violated: a==b cycle), got %zu\n", result.unsafeChains.size());
        return false;
    }

    // 内部頂点は 3 と 4
    const auto& chain = result.safeChains[0];
    if (chain.internalVertices.size() != 2) {
        fprintf(stderr, "FAIL: Expected 2 internal vertices, got %zu\n", chain.internalVertices.size());
        return false;
    }

    // pathLength = 3 (2 internal + 1)
    if (chain.pathLength != 3) {
        fprintf(stderr, "FAIL: Expected pathLength=3, got %d\n", chain.pathLength);
        return false;
    }

    // 縮約グラフ: 6 vertices (0,1,2,5,6,7), 6 edges
    if (result.reducedGraph.nodeCount != 6) {
        fprintf(stderr, "FAIL: Expected 6 remaining vertices, got %d\n", result.reducedGraph.nodeCount);
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース SAFE-2: 安全条件 C2 違反 (直接辺が存在)
//   Triangle + parallel chain: 0-1 (direct) + 0-2-1 (chain)
// ============================================================
static bool test_safe_c2_violation() {
    fprintf(stderr, "\n=== SAFE Test: C2 violation (direct edge exists) ===\n");

    vector<pair<int,int>> edges = {
        {0,1}, {0,2}, {2,1},    // 0-1 direct, chain 0-2-1
        {0,3}, {1,4}            // extra edges to make 0,1 non-degree-2
    };
    TestGraph tg = buildCSR(5, edges);

    // After degree-1 peeling would remove 3,4, but we test safeDegree2Compress directly
    SafeDegree2CompressResult result = safeDegree2Compress(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    // Chain 0-2-1: C2 violated (0-1 direct edge exists) → unsafe
    if (result.safeChains.size() != 0) {
        fprintf(stderr, "FAIL: Expected 0 safe chains, got %zu\n", result.safeChains.size());
        return false;
    }

    if (result.unsafeChains.size() != 1) {
        fprintf(stderr, "FAIL: Expected 1 unsafe chain, got %zu\n", result.unsafeChains.size());
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース SAFE-3: 安全条件 C3 違反 (共通隣接頂点が存在)
//   0-2-1 (chain), 0-3, 1-3 (共通隣接 3)
// ============================================================
static bool test_safe_c3_violation() {
    fprintf(stderr, "\n=== SAFE Test: C3 violation (common neighbor) ===\n");

    // 0: adj = [2, 3, 4] degree 3
    // 1: adj = [2, 3, 5] degree 3
    // 2: adj = [0, 1] degree 2 → chain 0-2-1
    // 3: adj = [0, 1] degree 2 → chain 0-3-1
    // 4: adj = [0] degree 1 (pendant)
    // 5: adj = [1] degree 1 (pendant)
    // chains: 0-2-1, 0-3-1
    // C3 check for chain 0-2-1: common neighbors of 0,1 excluding {2}?
    //   neighbors(0)\{2} = {3,4}, neighbors(1)\{2} = {3,5}
    //   common = {3} → C3 violated → unsafe
    // C3 check for chain 0-3-1: common neighbors of 0,1 excluding {3}?
    //   neighbors(0)\{3} = {2,4}, neighbors(1)\{3} = {2,5}
    //   common = {2} → C3 violated → unsafe
    vector<pair<int,int>> edges = {
        {0,2}, {2,1},   // chain 0-2-1
        {0,3}, {3,1},   // chain 0-3-1
        {0,4}, {1,5}    // pendants to keep 0,1 non-degree-2
    };
    TestGraph tg = buildCSR(6, edges);

    SafeDegree2CompressResult result = safeDegree2Compress(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    // Both chains should be unsafe due to C3
    if (result.safeChains.size() != 0) {
        fprintf(stderr, "FAIL: Expected 0 safe chains, got %zu\n", result.safeChains.size());
        return false;
    }

    fprintf(stderr, "  Unsafe chains: %zu\n", result.unsafeChains.size());
    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース SAFE-4: 安全条件 C1 違反 (サイクルチェーン)
//   0-1-2-0 where 1,2 are degree-2
//   But this forms a cycle → C1: a==b → unsafe
//   Actually in our tracing, this is detected as a pure cycle since
//   all vertices are degree-2, so no chains are found at all.
//   Let me create a specific test: 0-1-2-0 with 0 having extra edge
//   0: adj = [1,2,3] degree 3
//   1: adj = [0,2] degree 2
//   2: adj = [1,0] degree 2
//   Chain from 0, through 1: 0-1-2-0 → endpointA=0, endpointB=0 → C1 violated
// ============================================================
static bool test_safe_c1_violation() {
    fprintf(stderr, "\n=== SAFE Test: C1 violation (cycle chain) ===\n");

    // 0-1-2-0 (triangle) + 0-3 pendant → 0 has degree 3, 1 and 2 have degree 2
    // Chain: 0-1-2-0 → endpointA = endpointB = 0 → C1 violated
    vector<pair<int,int>> edges = {
        {0,1}, {1,2}, {2,0}, {0,3}
    };
    TestGraph tg = buildCSR(4, edges);

    SafeDegree2CompressResult result = safeDegree2Compress(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    // The chain 0-1-2-0 is unsafe (C1: a==b)
    if (result.safeChains.size() != 0) {
        fprintf(stderr, "FAIL: Expected 0 safe chains, got %zu\n", result.safeChains.size());
        return false;
    }

    fprintf(stderr, "  Unsafe chains: %zu\n", result.unsafeChains.size());
    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース SAFE-5: 重み付き Brandes (Dial's algorithm) の正確性
//   パスグラフ 0-1-2 を手動で重み付き CSR として渡す
//   weight(0-1)=1, weight(1-2)=2
//   → これは 0-a-1-b-c-2 のような元グラフに対応
//   → 距離: d(0,1)=1, d(0,2)=3, d(1,2)=2
//   → BC(0)=0, BC(1)=1, BC(2)=0 (全パス: 0-1, 0-2 via 1, 1-2. 1 は 0-2 の中間)
//   (無向グラフ: BC = Σ pairs / 2)
// ============================================================
static bool test_weighted_brandes_basic() {
    fprintf(stderr, "\n=== WEIGHTED Test: Basic weighted Brandes on path ===\n");

    // 重み付き CSR: 0-1 (w=1), 1-2 (w=2)
    WeightedReducedGraph wg;
    wg.nodeCount = 3;
    wg.edgeCount = 2;
    wg.adjacencyListPointers = {0, 1, 3, 4};
    wg.adjacencyList = {1, 0, 2, 1};
    wg.edgeWeight = {1, 1, 2, 2};
    wg.newToOrig = {0, 1, 2};
    wg.origToNew = {0, 1, 2};

    WeightedBrandesResult result = computeWeightedBCWithEdgeDep(wg);

    // BC(1) = 1.0 (中間点), BC(0) = BC(2) = 0
    fprintf(stderr, "  BC: ");
    for (int i = 0; i < 3; ++i) fprintf(stderr, "%.4f ", result.bc[i]);
    fprintf(stderr, "\n");

    if (fabs(result.bc[0] - 0.0) > 1e-9 ||
        fabs(result.bc[1] - 1.0) > 1e-9 ||
        fabs(result.bc[2] - 0.0) > 1e-9) {
        fprintf(stderr, "FAIL: BC values don't match expected\n");
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース SAFE-6: 重み=1 の場合、重み付き Brandes == 無重み Brandes
//   全辺重み1 の三角形で検証
// ============================================================
static bool test_weighted_brandes_unit_weight() {
    fprintf(stderr, "\n=== WEIGHTED Test: Unit weight == unweighted Brandes ===\n");

    // K4 グラフ (完全グラフ、全辺重み 1)
    vector<pair<int,int>> edges = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
    TestGraph tg = buildCSR(4, edges);

    // 無重み参照
    vector<double> refBC = referenceBrandes(
        tg.adjacencyListPointers.data(), tg.adjacencyList.data(), 4);

    // 重み付き (全辺重み 1)
    WeightedReducedGraph wg;
    wg.nodeCount = 4;
    wg.edgeCount = 6;
    wg.adjacencyListPointers = tg.adjacencyListPointers;
    wg.adjacencyList = tg.adjacencyList;
    wg.edgeWeight.assign(tg.adjacencyList.size(), 1);
    wg.newToOrig = {0, 1, 2, 3};
    wg.origToNew = {0, 1, 2, 3};

    WeightedBrandesResult result = computeWeightedBCWithEdgeDep(wg);

    fprintf(stderr, "  Weighted BC:   ");
    for (int i = 0; i < 4; ++i) fprintf(stderr, "%.4f ", result.bc[i]);
    fprintf(stderr, "\n  Reference BC:  ");
    for (int i = 0; i < 4; ++i) fprintf(stderr, "%.4f ", refBC[i]);
    fprintf(stderr, "\n");

    if (!compareBCVectors(result.bc, refBC)) {
        fprintf(stderr, "FAIL\n");
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース PIPELINE-D2-1: ダンベルグラフの E2E パイプライン
//   2つの三角形をチェーンで接続 → 安全チェーンが圧縮され、
//   重み付き Brandes + Degree-2 復元 → 元グラフの BC と一致
// ============================================================
static bool test_pipeline_dumbbell() {
    fprintf(stderr, "\n=== PIPELINE-D2 Test: Dumbbell (two triangles + safe chain) ===\n");

    vector<pair<int,int>> edges = {
        {0,1}, {1,2}, {2,0},     // triangle 1
        {2,3}, {3,4}, {4,5},     // chain (3,4 are degree-2) → safe
        {5,6}, {6,7}, {7,5}      // triangle 2
    };
    TestGraph tg = buildCSR(8, edges);

    vector<double> pipelineBC = runReductionPipeline(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    vector<double> directBC = referenceBrandes(
        tg.adjacencyListPointers.data(), tg.adjacencyList.data(), 8);

    fprintf(stderr, "  Pipeline BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", pipelineBC[i]);
    fprintf(stderr, "\n  Direct   BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", directBC[i]);
    fprintf(stderr, "\n");

    if (!compareBCVectors(pipelineBC, directBC)) {
        fprintf(stderr, "FAIL\n");
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース PIPELINE-D2-2: 長いチェーン (安全) + ペンダント
//   ハブ A(0) とハブ B(5): 0-1-2-3-4-5 のチェーン + ペンダント
//   0 は degree >= 3 (ペンダント追加), 5 も degree >= 3
// ============================================================
static bool test_pipeline_long_safe_chain() {
    fprintf(stderr, "\n=== PIPELINE-D2 Test: Long safe chain + pendants ===\n");

    // Two K3 hubs connected by a long chain
    // Hub 0: 0-8-9-0 (triangle)
    // Hub 5: 5-10-11-5 (triangle)
    // Chain: 0-1-2-3-4-5 (1,2,3,4 are degree-2)
    vector<pair<int,int>> edges = {
        {0,8}, {8,9}, {9,0},       // triangle 1
        {0,1}, {1,2}, {2,3}, {3,4}, {4,5},  // chain
        {5,10}, {10,11}, {11,5}    // triangle 2
    };
    TestGraph tg = buildCSR(12, edges);

    vector<double> pipelineBC = runReductionPipeline(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    vector<double> directBC = referenceBrandes(
        tg.adjacencyListPointers.data(), tg.adjacencyList.data(), 12);

    fprintf(stderr, "  Pipeline BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", pipelineBC[i]);
    fprintf(stderr, "\n  Direct   BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", directBC[i]);
    fprintf(stderr, "\n");

    if (!compareBCVectors(pipelineBC, directBC)) {
        fprintf(stderr, "FAIL\n");
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース PIPELINE-D2-3: 安全チェーンと非安全チェーンの混合
//   安全: 2つのハブ間に直接辺なしのチェーン
//   非安全: 同じハブ間に直接辺ありのチェーン
// ============================================================
static bool test_pipeline_mixed_safe_unsafe() {
    fprintf(stderr, "\n=== PIPELINE-D2 Test: Mixed safe/unsafe chains ===\n");

    // Hub 0 (degree 4), Hub 3 (degree 4)
    // Chain 0-1-2-3 (via 1,2) → check safety:
    //   C2: no direct 0-3 edge → safe... wait, let me add a direct edge
    // Actually let me create:
    // Hub 0 (triangle 0-4-5)
    // Hub 3 (triangle 3-6-7)
    // Safe chain: 0-1-2-3 (no direct 0-3 edge)
    //
    // Hub 8 (triangle 8-10-11)
    // Hub 9 (triangle 9-12-13)
    // Unsafe chain: 8-14-9 + direct edge 8-9 → C2 violated
    vector<pair<int,int>> edges = {
        {0,4}, {4,5}, {5,0},       // triangle for hub 0
        {3,6}, {6,7}, {7,3},       // triangle for hub 3
        {0,1}, {1,2}, {2,3},       // safe chain 0-1-2-3
        {8,10}, {10,11}, {11,8},   // triangle for hub 8
        {9,12}, {12,13}, {13,9},   // triangle for hub 9
        {8,14}, {14,9},            // chain 8-14-9
        {8,9}                      // direct edge → makes chain unsafe
    };
    TestGraph tg = buildCSR(15, edges);

    vector<double> pipelineBC = runReductionPipeline(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    vector<double> directBC = referenceBrandes(
        tg.adjacencyListPointers.data(), tg.adjacencyList.data(), 15);

    fprintf(stderr, "  Pipeline BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", pipelineBC[i]);
    fprintf(stderr, "\n  Direct   BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", directBC[i]);
    fprintf(stderr, "\n");

    if (!compareBCVectors(pipelineBC, directBC)) {
        fprintf(stderr, "FAIL\n");
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース PIPELINE-D2-4: 複数の安全チェーンが並列に存在
//   0-1-2-3 と 0-4-5-3 (parallel safe chains between hubs 0 and 3)
//   但し C3 チェック: chain1 の端点 0,3 は chain2 の内部頂点 4,5 を共有しない
//   → common neighbors? neighbors(0) = {1,4,...}, neighbors(3) = {2,5,...}
//   chain1 内部 = {1,2}, chain2 内部 = {4,5}
//   For chain1: common neighbors of 0,3 excluding {1,2}:
//     neighbors(0)\{1,2} includes 4, neighbors(3)\{1,2} includes 5
//     Common = {} (4 ∉ neighbors(3), 5 ∉ neighbors(0)) → safe!
//   For chain2: common neighbors of 0,3 excluding {4,5}:
//     neighbors(0)\{4,5} includes 1, neighbors(3)\{4,5} includes 2
//     Common = {} → safe!
// ============================================================
static bool test_pipeline_parallel_safe_chains() {
    fprintf(stderr, "\n=== PIPELINE-D2 Test: Parallel safe chains ===\n");

    // Hub 0: needs degree >= 3 (has edges to 1, 4, and 6)
    // Hub 3: needs degree >= 3 (has edges to 2, 5, and 7)
    vector<pair<int,int>> edges = {
        {0,1}, {1,2}, {2,3},       // safe chain 1: 0-1-2-3
        {0,4}, {4,5}, {5,3},       // safe chain 2: 0-4-5-3
        {0,6}, {3,7}               // extra edges to make 0,3 non-degree-2
    };
    TestGraph tg = buildCSR(8, edges);

    // Note: 6 and 7 are degree-1, will be peeled by degree-1 peeling

    vector<double> pipelineBC = runReductionPipeline(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    vector<double> directBC = referenceBrandes(
        tg.adjacencyListPointers.data(), tg.adjacencyList.data(), 8);

    fprintf(stderr, "  Pipeline BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", pipelineBC[i]);
    fprintf(stderr, "\n  Direct   BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", directBC[i]);
    fprintf(stderr, "\n");

    if (!compareBCVectors(pipelineBC, directBC)) {
        fprintf(stderr, "FAIL\n");
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// テストケース PIPELINE-D2-5: Degree-1 + Degree-2 統合テスト
//   ペンダントと安全チェーンが混在するシンプルな複合グラフ
//   Hub A=0 (triangle 0-5-6), Safe chain 0-1-2-3-4, Hub B=4 (triangle 4-7-8)
//   Pendants: 0-9, 4-10
// ============================================================
static bool test_pipeline_d2_complex() {
    fprintf(stderr, "\n=== PIPELINE-D2 Test: Complex graph with deg1+deg2 ===\n");

    // Hub A=0: triangle 0-5-6, pendant 0-9
    // Hub B=4: triangle 4-7-8, pendant 4-10
    // Safe chain: 0-1-2-3-4 (k=3 internal vertices: 1,2,3)
    // After deg-1 peel: remove 9,10 → 9 vertices remain
    // After deg-2 compress: chain 0-1-2-3-4 compressed → 6 vertices remain
    // n_a=3 ({0,5,6}), n_b=3 ({4,7,8})
    // BC(v1,i=1)=(3+0)*(3+2)=15, BC(v2,i=2)=(3+1)*(3+1)=16, BC(v3,i=3)=(3+2)*(3+0)=15
    vector<pair<int,int>> edges = {
        {0,5}, {5,6}, {6,0},         // triangle A (hub 0)
        {0,1}, {1,2}, {2,3}, {3,4},  // safe chain 0-1-2-3-4 (k=3)
        {4,7}, {7,8}, {8,4},         // triangle B (hub 4)
        {0,9},                        // pendant A
        {4,10}                        // pendant B
    };
    TestGraph tg = buildCSR(11, edges);

    vector<double> pipelineBC = runReductionPipeline(
        tg.adjacencyListPointers.data(),
        tg.adjacencyList.data(),
        tg.nodeCount, tg.edgeCount);

    vector<double> directBC = referenceBrandes(
        tg.adjacencyListPointers.data(), tg.adjacencyList.data(), 11);

    fprintf(stderr, "  Pipeline BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", pipelineBC[i]);
    fprintf(stderr, "\n  Direct   BC: ");
    for (int i = 0; i < tg.nodeCount; ++i) fprintf(stderr, "%.4f ", directBC[i]);
    fprintf(stderr, "\n");

    if (!compareBCVectors(pipelineBC, directBC)) {
        fprintf(stderr, "FAIL\n");
        return false;
    }

    fprintf(stderr, "PASS\n");
    return true;
}

// ============================================================
// メイン
// ============================================================
int main(int argc, char* argv[]) {
    int pass = 0, fail = 0;

    auto run = [&](bool result) {
        if (result) pass++;
        else fail++;
    };

    // Step 1 テスト
    run(test_star_graph());
    run(test_path_graph());
    run(test_triangle_with_pendants());
    run(test_complete_graph());
    run(test_tree_graph());
    run(test_id_mapping());

    // Step 2 テスト
    run(test_s2_triangle_chains());
    run(test_s2_complete_graph());
    run(test_s2_pure_cycle());
    run(test_s2_long_chain());
    run(test_s2_pipeline_step1_step2());
    run(test_s2_chain_info());

    // Step 3 テスト
    run(test_s3_open_twins());
    run(test_s3_no_twins());
    run(test_s3_triple_twins());
    run(test_s3_full_pipeline());

    // Step 4 テスト: 縮約グラフ上の BC 計算
    run(test_s4_bc_on_triangle());
    run(test_s4_bc_on_path());

    // Step 6 テスト: End-to-End パイプライン正確性検証
    run(test_s6_pipeline_k4());
    run(test_s6_pipeline_triangle_pendants());
    run(test_s6_pipeline_star());
    run(test_s6_pipeline_path());
    run(test_s6_pipeline_triangle_chain());
    run(test_s6_pipeline_twins());
    run(test_s6_pipeline_complex());

    // 安全 Degree-2 圧縮テスト
    run(test_safe_dumbbell());
    run(test_safe_c2_violation());
    run(test_safe_c3_violation());
    run(test_safe_c1_violation());

    // 重み付き Brandes テスト
    run(test_weighted_brandes_basic());
    run(test_weighted_brandes_unit_weight());

    // Degree-2 統合パイプラインテスト
    run(test_pipeline_dumbbell());
    run(test_pipeline_long_safe_chain());
    run(test_pipeline_mixed_safe_unsafe());
    run(test_pipeline_parallel_safe_chains());
    run(test_pipeline_d2_complex());

    // ファイルからのテスト（引数で指定）
    for (int i = 1; i < argc; ++i) {
        run(test_with_file(argv[i]));
    }

    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "  PASS: %d  FAIL: %d\n", pass, fail);
    fprintf(stderr, "========================================\n");

    return fail > 0 ? 1 : 0;
}
