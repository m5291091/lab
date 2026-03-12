#include "common.h"
#include "brandes.h"

using namespace std;

// Brandesのアルゴリズムを逐次実行する関数
vector<double> brandes_sequential(Graph &graph) {
    const int nodeCount = graph.getNodeCount();
    int *ap = graph.getAdjacencyListPointers();
    int *adj = graph.getAdjacencyList();
    vector<double> bc(nodeCount, 0.0);

    // フェーズ別時間計測
    double bfs_total = 0.0, back_total = 0.0;

    // 各ノードを始点として逐次処理
    for (int s = 0; s < nodeCount; ++s) {
        stack<int> st;
        vector<vector<int>> predecessors(nodeCount);
        vector<long long> sigma(nodeCount, 0);
        vector<int> distance(nodeCount, -1);
        vector<double> dependency(nodeCount, 0.0);

        distance[s] = 0;
        sigma[s] = 1;
        queue<int> q;
        q.push(s);

        double t_bfs_start = omp_get_wtime();

        // 1. BFSでsから各頂点への最短距離と最短経路の数を計算
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            st.push(v);

            for (int i = ap[v]; i < ap[v+1]; ++i) {
                int w = adj[i];

                // 未訪問の場合
                if (distance[w] < 0) {
                    q.push(w);
                    distance[w] = distance[v] + 1;
                }

                // 最短経路の場合
                if (distance[w] == distance[v] + 1) {
                    sigma[w] += sigma[v];
                    predecessors[w].push_back(v);
                }
            }
        }

        double t_back_start = omp_get_wtime();
        bfs_total += t_back_start - t_bfs_start;

        // 2. スタックでバックトレースを行いペア依存性を累積
        while(!st.empty()) {
            int w = st.top();
            st.pop();

            for (const int &v : predecessors[w]) {
                if (sigma[w] != 0) {
                    dependency[v] += (double(sigma[v]) / sigma[w]) * (1.0 + dependency[w]);
                }
            }

            if (w != s) {
                bc[w] += dependency[w] / 2.0; // 無向グラフのため2で割る
            }
        }

        back_total += omp_get_wtime() - t_back_start;
    }

    fprintf(stderr, "  > [Phase] BFS: %.4f sec, Backward: %.4f sec\n", bfs_total, back_total);

    return bc;
}
