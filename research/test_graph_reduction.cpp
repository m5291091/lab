#include "graph_reduction.h"
#include "Graph.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

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
//        0
//       /  \  (backslash)
//      1    2
//     /  \  (backslash)
//    3    4
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

    // ファイルからのテスト（引数で指定）
    for (int i = 1; i < argc; ++i) {
        run(test_with_file(argv[i]));
    }

    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "  PASS: %d  FAIL: %d\n", pass, fail);
    fprintf(stderr, "========================================\n");

    return fail > 0 ? 1 : 0;
}
