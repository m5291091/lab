#include "galliot_kernels.cuh"

#define INF 0x7FFFFFFF
#define BLOCK_SIZE 256

// ---- initBatchKernel ----
// Set dist=INF, sigma=0, delta=0 for all (source, vertex) pairs.
__global__ void initBatchKernel(
    int* dist, double* sigma, double* delta,
    int N, int batchSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * batchSize;
    if (idx < total) {
        dist[idx]  = INF;
        sigma[idx] = 0.0;
        delta[idx] = 0.0;
    }
}

// ---- initSourcesKernel ----
// For each source i in [0, numSources):
//   dist[i*N + v] = 0,  sigma[i*N + v] = 1
//   add (v, i) to frontier and stack.
__global__ void initSourcesKernel(
    const int* sourceVertices, int numSources,
    int* dist, double* sigma,
    int2* frontier, int* frontierSize,
    int2* stack,    int* stackSize,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numSources) return;

    int v = sourceVertices[i];
    dist [i * N + v] = 0;
    sigma[i * N + v] = 1.0;

    int fPos = atomicAdd(frontierSize, 1);
    frontier[fPos] = make_int2(v, i);

    int sPos = atomicAdd(stackSize, 1);
    stack[sPos] = make_int2(v, i);
}

// ---- bfsForwardKernel ----
// Each thread handles one (vertex v, source batch index s) from the current
// frontier.  Neighbours are discovered/updated using per-source dist/sigma
// arrays indexed as [s*N + w].
__global__ void bfsForwardKernel(
    const int* rowPtr, const int* colIdx,
    int* dist, double* sigma,
    const int2* currFrontier, int currFrontierSize,
    int2* nextFrontier, int* nextFrontierSize,
    int2* stack,        int* stackSize,
    int currentLevel,   int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= currFrontierSize) return;

    int2 entry = currFrontier[idx];
    int v = entry.x;   // vertex
    int s = entry.y;   // source batch index

    double sigma_v  = sigma[s * N + v];
    int    nextLevel = currentLevel + 1;

    int start = rowPtr[v];
    int end   = rowPtr[v + 1];

    for (int e = start; e < end; e++) {
        int w   = colIdx[e];
        int old = atomicCAS(&dist[s * N + w], INF, nextLevel);

        if (old == INF) {
            // First discovery of w from source s
            atomicAdd(&sigma[s * N + w], sigma_v);

            int qPos = atomicAdd(nextFrontierSize, 1);
            nextFrontier[qPos] = make_int2(w, s);

            int sPos = atomicAdd(stackSize, 1);
            stack[sPos] = make_int2(w, s);

        } else if (old == nextLevel) {
            // Additional shortest path
            atomicAdd(&sigma[s * N + w], sigma_v);
        }
    }
}

// ---- dependencyBackwardKernel ----
// Process all stack entries in [stackStart, stackEnd) — these are all entries
// at BFS level L.  For each (v, s):
//   - read delta_v = delta[s*N+v]  (accumulated from deeper levels)
//   - for each predecessor u (dist[s*N+u] == dist_v - 1):
//       delta[s*N+u] += (sigma[s*N+u] / sigma[s*N+v]) * (1 + delta_v)
//   - if v != source_vertex: BC[v] += delta_v
__global__ void dependencyBackwardKernel(
    const int* rowPtr, const int* colIdx,
    const int* dist,   const double* sigma,
    double* delta,     double* BC,
    const int2* stack, int stackStart, int stackEnd,
    const int* sourceVertices, int N)
{
    int idx      = blockIdx.x * blockDim.x + threadIdx.x;
    int stackIdx = stackStart + idx;
    if (stackIdx >= stackEnd) return;

    int2 entry = stack[stackIdx];
    int v = entry.x;
    int s = entry.y;

    double sigma_v = sigma[s * N + v];
    double delta_v = delta[s * N + v];
    int    dist_v  = dist [s * N + v];
    int    pred_d  = dist_v - 1;

    int start = rowPtr[v];
    int end   = rowPtr[v + 1];

    for (int e = start; e < end; e++) {
        int u = colIdx[e];
        if (dist[s * N + u] == pred_d) {
            double coeff = (sigma[s * N + u] / sigma_v) * (1.0 + delta_v);
            atomicAdd(&delta[s * N + u], coeff);
        }
    }

    if (v != sourceVertices[s]) {
        atomicAdd(&BC[v], delta_v);
    }
}