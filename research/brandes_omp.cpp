#include "common.h"
#include "brandes.h"

using namespace std;

vector<double> brandes_omp(Graph &graph) {
    const int nodeCount = graph.getNodeCount();
    int *ap = graph.getAdjacencyListPointers();
    int *adj = graph.getAdjacencyList();
    vector<double> bc(nodeCount, 0.0);

    double bfs_total = 0.0, back_total = 0.0;

#pragma omp parallel reduction(+:bfs_total, back_total)
    {
        vector<vector<int>> predecessors(nodeCount);
        vector<double> dependency(nodeCount);
        vector<long long> sigma(nodeCount);
        vector<int> distance(nodeCount);
        
#pragma omp for
        for (int s = 0; s < nodeCount; ++s) {
            stack<int> st;

            // Initialization
            for (int i = 0; i < nodeCount; ++i) {
                predecessors[i].clear();
                distance[i] = -1;
                sigma[i] = 0;
                dependency[i] = 0.0;
            }

            distance[s] = 0;
            sigma[s] = 1;
            queue<int> q;
            q.push(s);

            double t_bfs_start = omp_get_wtime();

            // BFS
            while (!q.empty()) {
                int v = q.front();
                q.pop();
                st.push(v);

                for (int i = ap[v]; i < ap[v+1]; ++i) {
                    int w = adj[i];
                    if (distance[w] < 0) {
                        q.push(w);
                        distance[w] = distance[v] + 1;
                    }
                    if (distance[w] == distance[v] + 1) {
                        sigma[w] += sigma[v];
                        predecessors[w].push_back(v);
                    }
                }
            }

            double t_back_start = omp_get_wtime();
            bfs_total += t_back_start - t_bfs_start;

            // Dependency accumulation
            while(!st.empty()) {
                int w = st.top();
                st.pop();

                for (const int &v : predecessors[w]) {
                    if (sigma[w] != 0) {
                        dependency[v] += (double(sigma[v]) / sigma[w]) * (1.0 + dependency[w]);
                    }
                }

                if (w != s) {
#pragma omp atomic
                    bc[w] += dependency[w] / 2.0; // Undirected graph
                }
            }

            back_total += omp_get_wtime() - t_back_start;
        }
    }

    fprintf(stderr, "  > [Phase] BFS: %.4f sec, Backward: %.4f sec (total thread-time)\n",
            bfs_total, back_total);

    return bc;
}