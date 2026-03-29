#include "graph_reduction.h"

#include <queue>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <unordered_map>
#include <set>

using namespace std;

// ============================================================
// Step 1: Degree-1 頂点除去（Peeling）
// ============================================================
Degree1PeelResult degree1Peel(const int* ap, const int* adj,
                              int nodeCount, int edgeCount) {
    Degree1PeelResult result;

    // --- 1. 全頂点の次数を計算 ---
    vector<int> degree(nodeCount);
    for (int i = 0; i < nodeCount; ++i) {
        degree[i] = ap[i + 1] - ap[i];
    }

    // --- 2. 除去フラグ ---
    vector<bool> removed(nodeCount, false);

    // --- 3. Degree-1 の頂点をキューに追加 ---
    queue<int> q;
    for (int i = 0; i < nodeCount; ++i) {
        if (degree[i] == 1) {
            q.push(i);
        }
    }

    // --- 4. 反復的に Degree-1 頂点を除去 ---
    while (!q.empty()) {
        int v = q.front();
        q.pop();

        // 既に除去済み or 次数が 1 でなくなっている場合はスキップ
        if (removed[v] || degree[v] != 1) {
            continue;
        }

        // v の唯一の非除去隣接頂点を見つける
        int neighbor = -1;
        for (int i = ap[v]; i < ap[v + 1]; ++i) {
            int w = adj[i];
            if (!removed[w]) {
                neighbor = w;
                break;
            }
        }

        // neighbor が見つからない場合は孤立頂点化（安全のためスキップ）
        if (neighbor == -1) {
            continue;
        }

        // 除去を記録
        removed[v] = true;
        PeeledVertex pv;
        pv.originalId = v;
        pv.neighborId = neighbor;
        result.peeledStack.push_back(pv);

        // 隣接頂点の実効次数を更新
        degree[neighbor]--;
        if (degree[neighbor] == 1) {
            q.push(neighbor);
        }
    }

    // --- 5. 残った頂点・辺から新しい CSR グラフを構築 ---
    ReducedGraph& rg = result.reducedGraph;
    rg.origToNew.assign(nodeCount, -1);

    // 新旧 ID マッピングの構築
    int newId = 0;
    for (int i = 0; i < nodeCount; ++i) {
        if (!removed[i]) {
            rg.origToNew[i] = newId;
            rg.newToOrig.push_back(i);
            newId++;
        }
    }
    rg.nodeCount = newId;

    // CSR の構築
    rg.adjacencyListPointers.resize(rg.nodeCount + 1, 0);

    // まず各新頂点の次数をカウント
    for (int newV = 0; newV < rg.nodeCount; ++newV) {
        int origV = rg.newToOrig[newV];
        int cnt = 0;
        for (int i = ap[origV]; i < ap[origV + 1]; ++i) {
            int origW = adj[i];
            if (!removed[origW]) {
                cnt++;
            }
        }
        rg.adjacencyListPointers[newV + 1] = cnt;
    }

    // プレフィックスサム
    for (int i = 0; i < rg.nodeCount; ++i) {
        rg.adjacencyListPointers[i + 1] += rg.adjacencyListPointers[i];
    }

    int totalAdj = rg.adjacencyListPointers[rg.nodeCount];
    rg.edgeCount = totalAdj / 2;
    rg.adjacencyList.resize(totalAdj);

    // 隣接リストを埋める
    vector<int> offset(rg.nodeCount, 0);
    for (int newV = 0; newV < rg.nodeCount; ++newV) {
        int origV = rg.newToOrig[newV];
        int base = rg.adjacencyListPointers[newV];
        for (int i = ap[origV]; i < ap[origV + 1]; ++i) {
            int origW = adj[i];
            if (!removed[origW]) {
                rg.adjacencyList[base + offset[newV]] = rg.origToNew[origW];
                offset[newV]++;
            }
        }
    }

    int removedCount = nodeCount - rg.nodeCount;
    int removedEdges = edgeCount - rg.edgeCount;
    fprintf(stderr, "[Degree1Peel] Removed %d vertices (%.1f%%), %d edges (%.1f%%)\n",
            removedCount,
            nodeCount > 0 ? 100.0 * removedCount / nodeCount : 0.0,
            removedEdges,
            edgeCount > 0 ? 100.0 * removedEdges / edgeCount : 0.0);
    fprintf(stderr, "[Degree1Peel] Reduced graph: %d vertices, %d edges\n",
            rg.nodeCount, rg.edgeCount);

    return result;
}

// ============================================================
// 検証: 縮約後グラフに Degree-1 頂点がないことを確認
// ============================================================
bool verifyNoDegree1(const ReducedGraph& g) {
    for (int i = 0; i < g.nodeCount; ++i) {
        int deg = g.adjacencyListPointers[i + 1] - g.adjacencyListPointers[i];
        if (deg == 1) {
            fprintf(stderr, "[Verify] FAIL: vertex %d (orig=%d) has degree 1\n",
                    i, g.newToOrig[i]);
            return false;
        }
    }
    return true;
}

// ============================================================
// Step 2: Degree-2 チェーン圧縮（Path/Chain Compression）
// ============================================================
Degree2CompressResult degree2Compress(const int* ap, const int* adj,
                                      int nodeCount, int edgeCount) {
    Degree2CompressResult result;

    // --- 1. 全頂点の次数を計算し、Degree-2 を識別 ---
    vector<int> degree(nodeCount);
    vector<bool> isDeg2(nodeCount, false);
    for (int i = 0; i < nodeCount; ++i) {
        degree[i] = ap[i + 1] - ap[i];
        if (degree[i] == 2) {
            isDeg2[i] = true;
        }
    }

    // --- 2. 非 Degree-2 頂点から出発してチェーンをたどる ---
    vector<bool> visited(nodeCount, false);
    // chainOf[v] = チェーン番号 (v がチェーン内部頂点の場合)、-1 otherwise
    vector<int> chainOf(nodeCount, -1);

    for (int v = 0; v < nodeCount; ++v) {
        if (isDeg2[v]) continue;  // 非 Degree-2 頂点のみ開始点

        for (int i = ap[v]; i < ap[v + 1]; ++i) {
            int w = adj[i];
            if (!isDeg2[w] || visited[w]) continue;

            // v から w を通るチェーンをたどる
            CompressedChain chain;
            chain.endpointA = v;

            vector<int> internal;
            int prev = v;
            int curr = w;

            while (isDeg2[curr] && !visited[curr]) {
                visited[curr] = true;
                internal.push_back(curr);

                // curr のもう一方の隣接頂点を見つける
                int next = -1;
                for (int j = ap[curr]; j < ap[curr + 1]; ++j) {
                    if (adj[j] != prev) {
                        next = adj[j];
                        break;
                    }
                }

                prev = curr;
                curr = next;
            }

            chain.endpointB = curr;  // 非 Degree-2 頂点 or 既訪問
            chain.internalVertices = internal;
            chain.pathLength = static_cast<int>(internal.size()) + 1;

            int chainIdx = static_cast<int>(result.chains.size());
            for (int iv : internal) {
                chainOf[iv] = chainIdx;
            }

            result.chains.push_back(chain);
        }
    }

    // --- 3. 未訪問の Degree-2 頂点はサイクル成分（そのまま保持） ---
    vector<bool> isCycleVertex(nodeCount, false);
    for (int v = 0; v < nodeCount; ++v) {
        if (isDeg2[v] && !visited[v]) {
            isCycleVertex[v] = true;
        }
    }

    // --- 4. 除去対象の決定: チェーン内部頂点のみ除去 ---
    vector<bool> removed(nodeCount, false);
    for (int v = 0; v < nodeCount; ++v) {
        if (isDeg2[v] && !isCycleVertex[v]) {
            removed[v] = true;
        }
    }

    // --- 5. 縮約グラフの構築 ---
    ReducedGraph& rg = result.reducedGraph;
    rg.origToNew.assign(nodeCount, -1);

    int newId = 0;
    for (int i = 0; i < nodeCount; ++i) {
        if (!removed[i]) {
            rg.origToNew[i] = newId;
            rg.newToOrig.push_back(i);
            newId++;
        }
    }
    rg.nodeCount = newId;

    // 各頂点の隣接リストを構築（重複排除に set を使用）
    vector<vector<int>> newAdj(rg.nodeCount);
    for (int newV = 0; newV < rg.nodeCount; ++newV) {
        int origV = rg.newToOrig[newV];
        // set で重複排除
        vector<int> neighbors;
        vector<bool> seen(rg.nodeCount, false);

        for (int i = ap[origV]; i < ap[origV + 1]; ++i) {
            int origW = adj[i];

            if (!removed[origW]) {
                // 直接の隣接頂点が保持されている
                int newW = rg.origToNew[origW];
                if (newW != newV && !seen[newW]) {
                    neighbors.push_back(newW);
                    seen[newW] = true;
                }
            } else {
                // origW はチェーン内部頂点 → チェーンの他端を接続
                int ci = chainOf[origW];
                if (ci < 0) continue;
                const CompressedChain& c = result.chains[ci];
                int otherEndpoint = (c.endpointA == origV) ? c.endpointB : c.endpointA;
                int newOther = rg.origToNew[otherEndpoint];
                if (newOther >= 0 && newOther != newV && !seen[newOther]) {
                    neighbors.push_back(newOther);
                    seen[newOther] = true;
                }
            }
        }

        newAdj[newV] = neighbors;
    }

    // CSR の構築
    rg.adjacencyListPointers.resize(rg.nodeCount + 1, 0);
    for (int i = 0; i < rg.nodeCount; ++i) {
        rg.adjacencyListPointers[i + 1] =
            rg.adjacencyListPointers[i] + static_cast<int>(newAdj[i].size());
    }

    int totalAdj = rg.adjacencyListPointers[rg.nodeCount];
    rg.edgeCount = totalAdj / 2;
    rg.adjacencyList.resize(totalAdj);

    for (int i = 0; i < rg.nodeCount; ++i) {
        int base = rg.adjacencyListPointers[i];
        for (int j = 0; j < static_cast<int>(newAdj[i].size()); ++j) {
            rg.adjacencyList[base + j] = newAdj[i][j];
        }
    }

    int removedCount = nodeCount - rg.nodeCount;
    int removedEdges = edgeCount - rg.edgeCount;
    fprintf(stderr, "[Degree2Compress] Removed %d vertices (%.1f%%), %d edges (%.1f%%)\n",
            removedCount,
            nodeCount > 0 ? 100.0 * removedCount / nodeCount : 0.0,
            removedEdges,
            edgeCount > 0 ? 100.0 * removedEdges / edgeCount : 0.0);
    fprintf(stderr, "[Degree2Compress] Compressed %zu chains, reduced graph: %d vertices, %d edges\n",
            result.chains.size(), rg.nodeCount, rg.edgeCount);

    return result;
}

// ============================================================
// 検証: 圧縮後グラフに Degree-2 チェーンがないことを確認
//   (純粋な Degree-2 サイクル成分は許容)
// ============================================================
bool verifyNoDegree2Chain(const ReducedGraph& g) {
    const auto& ap = g.adjacencyListPointers;
    const auto& adj = g.adjacencyList;

    for (int v = 0; v < g.nodeCount; ++v) {
        int deg = ap[v + 1] - ap[v];
        if (deg != 2) continue;

        // v は Degree-2。その両隣接頂点が非 Degree-2 ならチェーンが残っている
        bool neighborIsNonDeg2 = false;
        for (int i = ap[v]; i < ap[v + 1]; ++i) {
            int w = adj[i];
            int wdeg = ap[w + 1] - ap[w];
            if (wdeg != 2) {
                neighborIsNonDeg2 = true;
                break;
            }
        }

        if (neighborIsNonDeg2) {
            fprintf(stderr, "[Verify] FAIL: vertex %d (orig=%d) is a degree-2 chain vertex\n",
                    v, g.newToOrig[v]);
            return false;
        }
    }
    return true;
}

// ============================================================
// Step 3: 同一構造頂点の統合（Identical/Twin Vertex Merging）
// ============================================================

// ハッシュ関数: ソート済み隣接リストのハッシュ値を計算 (Boost hash_combine 方式)
static size_t hashNeighborList(const vector<int>& sorted_neighbors) {
    size_t h = 0;
    for (int v : sorted_neighbors) {
        // hash_combine 方式
        h ^= static_cast<size_t>(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
}

TwinMergeResult twinMerge(const int* ap, const int* adj,
                          int nodeCount, int edgeCount) {
    TwinMergeResult result;

    // --- 1. 各頂点の隣接リストをソートしハッシュ化 ---
    vector<vector<int>> sortedNeighbors(nodeCount);
    vector<size_t> hashes(nodeCount);

    for (int v = 0; v < nodeCount; ++v) {
        sortedNeighbors[v].assign(adj + ap[v], adj + ap[v + 1]);
        sort(sortedNeighbors[v].begin(), sortedNeighbors[v].end());
        hashes[v] = hashNeighborList(sortedNeighbors[v]);
    }

    // --- 2. 同一ハッシュの頂点をグループ化 ---
    unordered_map<size_t, vector<int>> hashGroups;
    for (int v = 0; v < nodeCount; ++v) {
        hashGroups[hashes[v]].push_back(v);
    }

    // --- 3. 隣接リストの完全一致を確認し、twin グループを構築 ---
    // representative[v] = v の代表頂点 (入力グラフの ID)
    vector<int> representative(nodeCount);
    for (int v = 0; v < nodeCount; ++v) {
        representative[v] = v;  // 初期値: 自分自身
    }

    for (auto& [h, vertices] : hashGroups) {
        if (vertices.size() < 2) continue;

        // ハッシュ衝突排除: 隣接リストの完全一致でサブグループに分割
        vector<bool> assigned(vertices.size(), false);

        for (size_t i = 0; i < vertices.size(); ++i) {
            if (assigned[i]) continue;

            vector<int> group;
            group.push_back(vertices[i]);
            assigned[i] = true;

            for (size_t j = i + 1; j < vertices.size(); ++j) {
                if (assigned[j]) continue;

                // 完全一致チェック
                if (sortedNeighbors[vertices[i]] == sortedNeighbors[vertices[j]]) {
                    group.push_back(vertices[j]);
                    assigned[j] = true;
                }
            }

            if (group.size() >= 2) {
                // 代表頂点は最小 ID
                int rep = *min_element(group.begin(), group.end());
                TwinGroup tg;
                tg.representative = rep;
                tg.members = group;

                for (int v : group) {
                    representative[v] = rep;
                }

                result.twinGroups.push_back(tg);
            }
        }
    }

    // --- 4. 縮約グラフの構築 ---
    ReducedGraph& rg = result.reducedGraph;

    // 代表頂点のみ保持
    vector<bool> isRepresentative(nodeCount, false);
    for (int v = 0; v < nodeCount; ++v) {
        if (representative[v] == v) {
            isRepresentative[v] = true;
        }
    }

    rg.origToNew.assign(nodeCount, -1);
    int newId = 0;
    for (int i = 0; i < nodeCount; ++i) {
        if (isRepresentative[i]) {
            rg.origToNew[i] = newId;
            rg.newToOrig.push_back(i);
            newId++;
        }
    }
    rg.nodeCount = newId;

    // 各代表頂点の隣接リストを構築（隣接頂点を代表に変換＋重複排除）
    vector<vector<int>> newAdj(rg.nodeCount);
    for (int newV = 0; newV < rg.nodeCount; ++newV) {
        int origV = rg.newToOrig[newV];
        set<int> neighborSet;

        for (int i = ap[origV]; i < ap[origV + 1]; ++i) {
            int origW = adj[i];
            int repW = representative[origW];
            int newW = rg.origToNew[repW];

            // 自己ループ回避、重複排除
            if (newW >= 0 && newW != newV) {
                neighborSet.insert(newW);
            }
        }

        newAdj[newV].assign(neighborSet.begin(), neighborSet.end());
    }

    // CSR の構築
    rg.adjacencyListPointers.resize(rg.nodeCount + 1, 0);
    for (int i = 0; i < rg.nodeCount; ++i) {
        rg.adjacencyListPointers[i + 1] =
            rg.adjacencyListPointers[i] + static_cast<int>(newAdj[i].size());
    }

    int totalAdj = rg.adjacencyListPointers[rg.nodeCount];
    rg.edgeCount = totalAdj / 2;
    rg.adjacencyList.resize(totalAdj);

    for (int i = 0; i < rg.nodeCount; ++i) {
        int base = rg.adjacencyListPointers[i];
        for (int j = 0; j < static_cast<int>(newAdj[i].size()); ++j) {
            rg.adjacencyList[base + j] = newAdj[i][j];
        }
    }

    int removedCount = nodeCount - rg.nodeCount;
    int removedEdges = edgeCount - rg.edgeCount;
    fprintf(stderr, "[TwinMerge] Found %zu twin groups, removed %d vertices (%.1f%%), %d edges (%.1f%%)\n",
            result.twinGroups.size(), removedCount,
            nodeCount > 0 ? 100.0 * removedCount / nodeCount : 0.0,
            removedEdges,
            edgeCount > 0 ? 100.0 * removedEdges / edgeCount : 0.0);
    fprintf(stderr, "[TwinMerge] Reduced graph: %d vertices, %d edges\n",
            rg.nodeCount, rg.edgeCount);

    return result;
}

// ============================================================
// 検証: 統合後グラフに同一構造頂点が存在しないことを確認
// ============================================================
bool verifyNoTwins(const ReducedGraph& g) {
    const auto& ap = g.adjacencyListPointers;
    const auto& adj = g.adjacencyList;

    // 各頂点の隣接リストをソートしハッシュ化
    unordered_map<size_t, vector<int>> hashGroups;

    for (int v = 0; v < g.nodeCount; ++v) {
        vector<int> sorted_neighbors(adj.data() + ap[v], adj.data() + ap[v + 1]);
        sort(sorted_neighbors.begin(), sorted_neighbors.end());
        size_t h = hashNeighborList(sorted_neighbors);
        hashGroups[h].push_back(v);
    }

    for (auto& [h, vertices] : hashGroups) {
        if (vertices.size() < 2) continue;

        // 完全一致チェック
        for (size_t i = 0; i < vertices.size(); ++i) {
            vector<int> ni(adj.data() + ap[vertices[i]],
                           adj.data() + ap[vertices[i] + 1]);
            sort(ni.begin(), ni.end());

            for (size_t j = i + 1; j < vertices.size(); ++j) {
                vector<int> nj(adj.data() + ap[vertices[j]],
                               adj.data() + ap[vertices[j] + 1]);
                sort(nj.begin(), nj.end());

                if (ni == nj) {
                    fprintf(stderr, "[Verify] FAIL: vertices %d (orig=%d) and %d (orig=%d) "
                            "have identical neighbor sets\n",
                            vertices[i], g.newToOrig[vertices[i]],
                            vertices[j], g.newToOrig[vertices[j]]);
                    return false;
                }
            }
        }
    }

    return true;
}
