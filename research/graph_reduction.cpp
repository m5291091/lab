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

// ============================================================
// 安全条件付き Degree-2 チェーン圧縮 (Sariyüce 2013 の安全条件)
// ============================================================
SafeDegree2CompressResult safeDegree2Compress(const int* ap, const int* adj,
                                              int nodeCount, int edgeCount) {
    SafeDegree2CompressResult result;

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
    vector<int> chainOf(nodeCount, -1);
    vector<CompressedChain> allChains;

    for (int v = 0; v < nodeCount; ++v) {
        if (isDeg2[v]) continue;

        for (int i = ap[v]; i < ap[v + 1]; ++i) {
            int w = adj[i];
            if (!isDeg2[w] || visited[w]) continue;

            CompressedChain chain;
            chain.endpointA = v;

            vector<int> internal;
            int prev = v;
            int curr = w;

            while (isDeg2[curr] && !visited[curr]) {
                visited[curr] = true;
                internal.push_back(curr);

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

            chain.endpointB = curr;
            chain.internalVertices = internal;
            chain.pathLength = static_cast<int>(internal.size()) + 1;
            allChains.push_back(chain);
        }
    }

    // --- 3. 未訪問の Degree-2 頂点はサイクル成分 ---
    vector<bool> isCycleVertex(nodeCount, false);
    for (int v = 0; v < nodeCount; ++v) {
        if (isDeg2[v] && !visited[v]) {
            isCycleVertex[v] = true;
        }
    }

    // --- 4. 各チェーンの安全条件を判定 ---
    // 高速隣接チェック用: 各頂点の隣接をセット化
    vector<set<int>> neighborSet(nodeCount);
    for (int v = 0; v < nodeCount; ++v) {
        for (int i = ap[v]; i < ap[v + 1]; ++i) {
            neighborSet[v].insert(adj[i]);
        }
    }

    vector<bool> isSafe(allChains.size(), true);
    for (size_t ci = 0; ci < allChains.size(); ++ci) {
        const CompressedChain& c = allChains[ci];
        int a = c.endpointA;
        int b = c.endpointB;

        // C1: a ≠ b（サイクルでない）
        if (a == b) {
            isSafe[ci] = false;
            continue;
        }

        // C2: a-b 間に直接辺がない
        if (neighborSet[a].count(b) > 0) {
            isSafe[ci] = false;
            continue;
        }

        // C3: a と b に {v₁...vₖ} 以外の共通隣接頂点がない
        set<int> internalSet(c.internalVertices.begin(), c.internalVertices.end());
        bool hasCommonNeighbor = false;
        for (int na : neighborSet[a]) {
            if (internalSet.count(na) > 0) continue;  // チェーン内部は除外
            if (na == b) continue;  // C2 で既にチェック済みだが念のため
            if (neighborSet[b].count(na) > 0) {
                hasCommonNeighbor = true;
                break;
            }
        }
        if (hasCommonNeighbor) {
            isSafe[ci] = false;
            continue;
        }
    }

    // --- 5. 安全/非安全チェーンを分類 ---
    vector<bool> removedByChain(nodeCount, false);
    int safeChainCount = 0;
    for (size_t ci = 0; ci < allChains.size(); ++ci) {
        if (isSafe[ci]) {
            result.safeChains.push_back(allChains[ci]);
            int scIdx = safeChainCount++;
            for (int iv : allChains[ci].internalVertices) {
                chainOf[iv] = scIdx;
                removedByChain[iv] = true;
            }
        } else {
            result.unsafeChains.push_back(allChains[ci]);
        }
    }

    // --- 6. 除去対象: 安全チェーンの内部頂点のみ ---
    vector<bool> removed(nodeCount, false);
    for (int v = 0; v < nodeCount; ++v) {
        if (removedByChain[v]) {
            removed[v] = true;
        }
    }

    // --- 7. 縮約グラフの構築 ---
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

    // 各頂点の隣接リストを構築
    vector<vector<int>> newAdj(rg.nodeCount);
    for (int newV = 0; newV < rg.nodeCount; ++newV) {
        int origV = rg.newToOrig[newV];
        vector<bool> seen(rg.nodeCount, false);

        for (int i = ap[origV]; i < ap[origV + 1]; ++i) {
            int origW = adj[i];

            if (!removed[origW]) {
                int newW = rg.origToNew[origW];
                if (newW != newV && !seen[newW]) {
                    newAdj[newV].push_back(newW);
                    seen[newW] = true;
                }
            } else {
                // origW は安全チェーン内部頂点 → チェーンの他端を接続
                int ci = chainOf[origW];
                if (ci < 0) continue;
                const CompressedChain& c = result.safeChains[ci];
                int otherEndpoint = (c.endpointA == origV) ? c.endpointB : c.endpointA;
                int newOther = rg.origToNew[otherEndpoint];
                if (newOther >= 0 && newOther != newV && !seen[newOther]) {
                    newAdj[newV].push_back(newOther);
                    seen[newOther] = true;
                }
            }
        }
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
    fprintf(stderr, "[SafeDeg2Compress] Safe chains: %zu, Unsafe chains: %zu\n",
            result.safeChains.size(), result.unsafeChains.size());
    fprintf(stderr, "[SafeDeg2Compress] Removed %d vertices (%.1f%%), %d edges (%.1f%%)\n",
            removedCount,
            nodeCount > 0 ? 100.0 * removedCount / nodeCount : 0.0,
            removedEdges,
            edgeCount > 0 ? 100.0 * removedEdges / edgeCount : 0.0);
    fprintf(stderr, "[SafeDeg2Compress] Reduced graph: %d vertices, %d edges\n",
            rg.nodeCount, rg.edgeCount);

    return result;
}

// ============================================================
// Step 4: 縮約グラフ上での厳密 BC 計算 (CPU Brandes)
// ============================================================
vector<double> computeBCOnReducedGraph(const ReducedGraph& rg) {
    const int n = rg.nodeCount;
    const int* ap = rg.adjacencyListPointers.data();
    const int* adj = rg.adjacencyList.data();
    vector<double> bc(n, 0.0);

    if (n <= 1) return bc;

    // Brandes 法 (逐次版)
    for (int s = 0; s < n; ++s) {
        // BFS + 最短経路カウント
        vector<int> S;              // スタック（BFS 順）
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
            int v = Q.front();
            Q.pop();
            S.push_back(v);

            for (int i = ap[v]; i < ap[v + 1]; ++i) {
                int w = adj[i];
                if (dist[w] < 0) {
                    Q.push(w);
                    dist[w] = dist[v] + 1;
                }
                if (dist[w] == dist[v] + 1) {
                    sigma[w] += sigma[v];
                    pred[w].push_back(v);
                }
            }
        }

        // バックプロパゲーション
        for (int i = static_cast<int>(S.size()) - 1; i >= 0; --i) {
            int w = S[i];
            for (int v : pred[w]) {
                if (sigma[w] != 0) {
                    delta[v] += (static_cast<double>(sigma[v]) / sigma[w]) * (1.0 + delta[w]);
                }
            }
            if (w != s) {
                bc[w] += delta[w] / 2.0;  // 無向グラフのため 2 で割る
            }
        }
    }

    fprintf(stderr, "[ComputeBC] Computed BC on reduced graph (%d vertices, %d edges)\n",
            n, rg.edgeCount);
    return bc;
}

// ============================================================
// 重み付き Brandes 法 (Dial's algorithm + 辺依存性記録)
// ============================================================
//
// Dial's algorithm: 整数重み SSSP のための bucket queue
//   - 各 bucket[d] に距離 d の頂点を格納
//   - 最大距離 D_max は辺重みの合計の上界
//   - 計算量: O(V + E + D_max)
//
// 辺依存性:
//   Brandes の back-propagation で辺 (v,w) の依存性を記録:
//     edge_dep(v,w) += (σ[v]/σ[w]) × (1 + δ[w])  when dist[w] = dist[v] + weight(v,w)
//   安全チェーンの縮約辺の edge_dep がチェーン内部頂点の BC に対応
// ============================================================
WeightedBrandesResult computeWeightedBCWithEdgeDep(const WeightedReducedGraph& wg) {
    const int n = wg.nodeCount;
    const int* ap = wg.adjacencyListPointers.data();
    const int* adj = wg.adjacencyList.data();
    const int* ew = wg.edgeWeight.data();
    int totalAdj = static_cast<int>(wg.adjacencyList.size());

    WeightedBrandesResult result;
    result.bc.assign(n, 0.0);
    result.edgeDep.assign(totalAdj, 0.0);

    if (n <= 1) return result;

    // D_max の計算 (全辺重みの合計 / 2 が上界)
    int maxWeight = 0;
    for (int i = 0; i < totalAdj; ++i) {
        if (ew[i] > maxWeight) maxWeight = ew[i];
    }
    // 最大距離 = (n-1) * maxWeight が上界
    int dMax = (n - 1) * maxWeight;

    for (int s = 0; s < n; ++s) {
        // --- Dial's algorithm (bucket queue SSSP) ---
        vector<int> S;
        S.reserve(n);
        vector<vector<int>> pred(n);
        vector<long long> sigma(n, 0);
        vector<int> dist(n, -1);
        vector<double> delta(n, 0.0);

        // bucket queue
        vector<vector<int>> bucket(dMax + 1);

        dist[s] = 0;
        sigma[s] = 1;
        bucket[0].push_back(s);

        int settled = 0;
        for (int d = 0; d <= dMax && settled < n; ++d) {
            for (size_t bi = 0; bi < bucket[d].size(); ++bi) {
                int v = bucket[d][bi];
                if (dist[v] != d) continue;  // 古いエントリをスキップ
                S.push_back(v);
                settled++;

                for (int i = ap[v]; i < ap[v + 1]; ++i) {
                    int w = adj[i];
                    int newDist = d + ew[i];
                    if (newDist > dMax) continue;

                    if (dist[w] < 0) {
                        // 未到達
                        dist[w] = newDist;
                        sigma[w] = sigma[v];
                        pred[w].push_back(v);
                        bucket[newDist].push_back(w);
                    } else if (dist[w] == newDist) {
                        // 同距離の別経路
                        sigma[w] += sigma[v];
                        pred[w].push_back(v);
                    }
                    // dist[w] < newDist の場合は何もしない
                }
            }
        }

        // --- バックプロパゲーション + 辺依存性記録 ---
        // 逆 pred マップ: w の predecessor v のうち、CSR 上の辺インデックスを検索
        for (int i = static_cast<int>(S.size()) - 1; i >= 0; --i) {
            int w = S[i];
            for (int v : pred[w]) {
                if (sigma[w] != 0) {
                    double contrib = (static_cast<double>(sigma[v]) / sigma[w]) * (1.0 + delta[w]);
                    delta[v] += contrib;

                    // 辺依存性: v→w の辺インデックスを探索
                    for (int ei = ap[v]; ei < ap[v + 1]; ++ei) {
                        if (adj[ei] == w && dist[w] == dist[v] + ew[ei]) {
                            result.edgeDep[ei] += contrib / 2.0;  // 無向グラフのため
                            break;
                        }
                    }
                    // 辺依存性: w→v の辺インデックスも記録 (対称)
                    for (int ei = ap[w]; ei < ap[w + 1]; ++ei) {
                        if (adj[ei] == v && dist[w] == dist[v] + ew[ei]) {
                            result.edgeDep[ei] += contrib / 2.0;
                            break;
                        }
                    }
                }
            }
            if (w != s) {
                result.bc[w] += delta[w] / 2.0;
            }
        }
    }

    fprintf(stderr, "[WeightedBC] Computed weighted BC on graph (%d vertices, %d edges, maxWeight=%d)\n",
            n, wg.edgeCount, maxWeight);
    return result;
}

// ============================================================
// Step 5: BC 復元（逆変換）
// ============================================================
//
// 重要な設計判断:
//   Degree-1 頂点の除去は、残存頂点間の最短経路を変えない。
//   したがって、縮約グラフ上の Brandes 結果は、元グラフの
//   「残存頂点ソースのみの部分和」と一致する。除去頂点分の
//   ソース寄与を BFS で追加するだけで正確な復元が可能。
//
//   Degree-2 チェーン圧縮は安全条件を満たすチェーンのみ縮約し、
//   Bentert et al. (2018) の復元公式により辺依存性から内部頂点の
//   BC を復元する。安全でないチェーンは元のままグラフに残す。
//
// Degree-1 復元公式（数学的導出）:
//   葉頂点 v (唯一の隣接 = p) を除去した G' → 元グラフ G への復元:
//   - BC(v) = 0  (葉は他の s-t 最短経路上に存在しない)
//   - BC(p)_G = BC(p)_{G'} + (n-2)
//       n = |V(G)| = |V(G')| + 1
//       理由: 全ソース s!=v,p について delta_s(p) が +1 増加 (t=v の項)、
//             ソース v からの delta_v(p) = n-2 (全頂点が p を経由)
//   - BC(u)_G = BC(u)_{G'} + delta^{G'}_p(u)  (u != p, v)
//       delta^{G'}_p(u) = Brandes の source=p での u の依存性 (G' 上)
//       理由: 対称性により t=v の寄与 = source=p での u の寄与
// ----------------------------------------------------------

// BFS + back-propagation from single source in adjacency list graph
// Returns dependency delta[u] for all u
static vector<double> singleSourceDependency(
        const vector<vector<int>>& adjList, int n, int src) {
    vector<int> S;
    S.reserve(n);
    vector<vector<int>> pred(n);
    vector<long long> sigma(n, 0);
    vector<int> dist(n, -1);
    vector<double> delta(n, 0.0);

    dist[src] = 0;
    sigma[src] = 1;
    queue<int> Q;
    Q.push(src);

    while (!Q.empty()) {
        int v = Q.front(); Q.pop();
        S.push_back(v);
        for (int w : adjList[v]) {
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
    }

    return delta;
}

// ----------------------------------------------------------
// Step 1 逆変換: Degree-1 頂点の BC 復元 (BFS ベース正確復元)
//
// 処理: peeledStack を逆順に走査し、各頂点を順に復元する。
//   各復元で parent p から BFS を実行して全頂点の依存性を計算し、
//   BC 値を更新する。
// ----------------------------------------------------------
void restoreDegree1BC(vector<double>& bc,
                      const Degree1PeelResult& peelResult,
                      int origNodeCount) {
    const ReducedGraph& rg = peelResult.reducedGraph;
    const auto& stack = peelResult.peeledStack;

    if (stack.empty()) {
        // 除去なし: 単純にマッピングのみ
        vector<double> expanded(origNodeCount, 0.0);
        for (int newV = 0; newV < rg.nodeCount; ++newV) {
            expanded[rg.newToOrig[newV]] = bc[newV];
        }
        bc = expanded;
        fprintf(stderr, "[RestoreDeg1BC] No peeled vertices, mapping only\n");
        return;
    }

    // 動的隣接リストを構築 (元グラフの ID で管理)
    vector<vector<int>> adjList(origNodeCount);

    // 縮約グラフの辺を追加
    for (int newV = 0; newV < rg.nodeCount; ++newV) {
        int origV = rg.newToOrig[newV];
        for (int i = rg.adjacencyListPointers[newV]; i < rg.adjacencyListPointers[newV + 1]; ++i) {
            int origW = rg.newToOrig[rg.adjacencyList[i]];
            adjList[origV].push_back(origW);
        }
    }

    // BC を元グラフの ID にマッピング
    vector<double> origBC(origNodeCount, 0.0);
    for (int newV = 0; newV < rg.nodeCount; ++newV) {
        origBC[rg.newToOrig[newV]] = bc[newV];
    }

    // 現在のグラフのアクティブ頂点数
    int currentN = rg.nodeCount;

    // peeledStack を逆順に処理
    for (int i = static_cast<int>(stack.size()) - 1; i >= 0; --i) {
        int v = stack[i].originalId;
        int p = stack[i].neighborId;

        // (1) p から BFS を実行 (v はまだグラフにいない)
        vector<double> delta = singleSourceDependency(adjList, origNodeCount, p);

        // (2) BC(p) += currentN - 1
        //     (n = currentN + 1 が復元後のサイズ、n-2 = currentN - 1)
        origBC[p] += static_cast<double>(currentN - 1);

        // (3) BC(u) += delta_p(u) for all u != p
        for (int u = 0; u < origNodeCount; ++u) {
            if (u != p && delta[u] > 0.0) {
                origBC[u] += delta[u];
            }
        }

        // (4) v をグラフに追加
        adjList[v].push_back(p);
        adjList[p].push_back(v);

        // (5) BC(v) = 0 (既に 0 初期化済み)

        currentN++;
    }

    bc = origBC;

    fprintf(stderr, "[RestoreDeg1BC] Restored BC for %zu peeled vertices (BFS-based)\n",
            stack.size());
}

// ============================================================
// Step 2 逆変換: Degree-2 チェーンの BC 復元 (Bentert 2018)
// ============================================================
//
// 安全条件を満たすチェーン a - v₁ - ... - vₖ - b について:
//   BC_G(vᵢ) = 辺 (a_new, b_new) の辺依存性
//   ∵ 安全条件により s-t 最短経路がチェーンを通過 ⟺ 縮約辺 a-b を通過
//     かつチェーン内部全頂点は一意にこの経路上にある
//
// 端点 a, b の BC は重み付き Brandes で正しく計算済み（g1 の ID で管理）
// ============================================================
void restoreDegree2BC(vector<double>& bc,
                      const SafeDegree2CompressResult& compResult,
                      const WeightedBrandesResult& brandesResult,
                      const WeightedReducedGraph& wg,
                      int origNodeCount) {
    const ReducedGraph& rg = compResult.reducedGraph;

    // BC を元の g1 (Degree-1 Peeling 後) グラフの ID にマッピング
    vector<double> expanded(origNodeCount, 0.0);
    for (int newV = 0; newV < rg.nodeCount; ++newV) {
        int origV = rg.newToOrig[newV];
        expanded[origV] = brandesResult.bc[newV];
    }

    // 各安全チェーンの内部頂点に BC を復元
    const int* ap = wg.adjacencyListPointers.data();
    const int* adj = wg.adjacencyList.data();

    for (const auto& chain : compResult.safeChains) {
        int a = chain.endpointA;
        int b = chain.endpointB;

        // 縮約グラフ上での a, b の新 ID
        int newA = rg.origToNew[a];
        int newB = rg.origToNew[b];

        if (newA < 0 || newB < 0) continue;

        // 辺 (newA, newB) の辺依存性を取得
        double chainDep = 0.0;
        for (int ei = ap[newA]; ei < ap[newA + 1]; ++ei) {
            if (adj[ei] == newB) {
                chainDep = brandesResult.edgeDep[ei];
                break;
            }
        }

        // チェーン内部全頂点に同一の BC 値を割り当て
        for (int iv : chain.internalVertices) {
            expanded[iv] = chainDep;
        }
    }

    bc = expanded;

    fprintf(stderr, "[RestoreDeg2BC] Restored BC for %zu safe chains (%zu unsafe skipped)\n",
            compResult.safeChains.size(), compResult.unsafeChains.size());
}

// ============================================================
// Step 6: End-to-End パイプライン (Degree-1 + Degree-2 統合版)
// ============================================================
//
// パイプライン構成:
//   Step 1 (Degree-1 Peeling) →
//   Step 2 (安全 Degree-2 圧縮) →
//   重み付き Brandes (Dial's algorithm + 辺依存性記録) →
//   Degree-2 BC 復元 →
//   Degree-1 BC 復元 →
//   全頂点 BC
// ============================================================
vector<double> runReductionPipeline(const int* ap, const int* adj,
                                    int nodeCount, int edgeCount) {
    fprintf(stderr, "\n[Pipeline] Original graph: %d vertices, %d edges\n",
            nodeCount, edgeCount);

    // --- Step 1: Degree-1 Peeling ---
    fprintf(stderr, "[Pipeline] Step 1: Degree-1 Peeling...\n");
    Degree1PeelResult s1 = degree1Peel(ap, adj, nodeCount, edgeCount);

    const ReducedGraph& g1 = s1.reducedGraph;
    fprintf(stderr, "[Pipeline] After Step 1: %d vertices, %d edges (%.1f%% vertex reduction)\n",
            g1.nodeCount, g1.edgeCount,
            nodeCount > 0 ? 100.0 * (1.0 - static_cast<double>(g1.nodeCount) / nodeCount) : 0.0);

    // --- Step 2: 安全条件付き Degree-2 圧縮 ---
    fprintf(stderr, "[Pipeline] Step 2: Safe Degree-2 Compression...\n");
    SafeDegree2CompressResult s2 = safeDegree2Compress(
        g1.adjacencyListPointers.data(), g1.adjacencyList.data(),
        g1.nodeCount, g1.edgeCount);

    const ReducedGraph& g2 = s2.reducedGraph;
    fprintf(stderr, "[Pipeline] After Step 2: %d vertices, %d edges (safe: %zu chains, unsafe: %zu chains)\n",
            g2.nodeCount, g2.edgeCount,
            s2.safeChains.size(), s2.unsafeChains.size());

    // --- (参考情報) Step 3 の縮約率を計測 ---
    {
        TwinMergeResult s3 = twinMerge(
            g2.adjacencyListPointers.data(), g2.adjacencyList.data(),
            g2.nodeCount, g2.edgeCount);
        fprintf(stderr, "[Pipeline] (Info) Full reduction Step1+2+3: %d -> %d vertices, %d -> %d edges\n",
                nodeCount, s3.reducedGraph.nodeCount, edgeCount, s3.reducedGraph.edgeCount);
    }

    // --- 重み付き CSR の構築 ---
    // 安全チェーンの縮約辺に pathLength を重みとして付与
    // 通常辺は重み 1
    WeightedReducedGraph wg;
    wg.nodeCount = g2.nodeCount;
    wg.edgeCount = g2.edgeCount;
    wg.adjacencyListPointers = g2.adjacencyListPointers;
    wg.adjacencyList = g2.adjacencyList;
    wg.newToOrig = g2.newToOrig;
    wg.origToNew = g2.origToNew;

    int totalAdj2 = static_cast<int>(g2.adjacencyList.size());
    wg.edgeWeight.assign(totalAdj2, 1);  // デフォルト重み 1

    // 安全チェーンの縮約辺に pathLength を設定
    for (const auto& chain : s2.safeChains) {
        int newA = g2.origToNew[chain.endpointA];
        int newB = g2.origToNew[chain.endpointB];

        if (newA < 0 || newB < 0) continue;

        // newA → newB の辺に重みを設定
        for (int ei = g2.adjacencyListPointers[newA]; ei < g2.adjacencyListPointers[newA + 1]; ++ei) {
            if (g2.adjacencyList[ei] == newB) {
                wg.edgeWeight[ei] = chain.pathLength;
                break;
            }
        }
        // newB → newA の辺にも同じ重みを設定
        for (int ei = g2.adjacencyListPointers[newB]; ei < g2.adjacencyListPointers[newB + 1]; ++ei) {
            if (g2.adjacencyList[ei] == newA) {
                wg.edgeWeight[ei] = chain.pathLength;
                break;
            }
        }
    }

    // --- Step 4: 重み付き Brandes (Dial's algorithm + 辺依存性) ---
    fprintf(stderr, "[Pipeline] Step 4: Computing weighted BC with edge dependencies...\n");
    WeightedBrandesResult brandesResult = computeWeightedBCWithEdgeDep(wg);

    // --- Step 5a: Degree-2 BC 復元 ---
    fprintf(stderr, "[Pipeline] Step 5a: Restoring Degree-2 BC...\n");
    vector<double> bc = brandesResult.bc;
    restoreDegree2BC(bc, s2, brandesResult, wg, g1.nodeCount);

    // --- Step 5b: Degree-1 BC 復元 ---
    fprintf(stderr, "[Pipeline] Step 5b: Restoring Degree-1 BC (BFS-based)...\n");
    restoreDegree1BC(bc, s1, nodeCount);

    fprintf(stderr, "[Pipeline] Complete. Restored BC for %d vertices.\n", nodeCount);
    return bc;
}
