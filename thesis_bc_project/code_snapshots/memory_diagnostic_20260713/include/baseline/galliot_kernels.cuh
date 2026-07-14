#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

// Initialize per-source dist/sigma/delta arrays for a batch.
// Total elements = N * batchSize.
__global__ void initBatchKernel(
    int* dist,         // [batchSize * N]
    double* sigma,     // [batchSize * N]
    double* delta,     // [batchSize * N]
    int N,
    int batchSize
);

// Set distance=0, sigma=1 for each source vertex, add to frontier & stack.
__global__ void initSourcesKernel(
    const int* sourceVertices,  // [numSources] actual vertex IDs
    int numSources,
    int* dist,                  // [batchSize * N]
    double* sigma,              // [batchSize * N]
    int2* frontier,             // output: initial frontier (vertex, src_batch_idx)
    int* frontierSize,
    int2* stack,                // output: stack entries for level 0
    int* stackSize,
    int N
);

// Per-source parallel BFS: each (vertex, source) in the current frontier
// independently explores neighbours using dist[s*N+v] indexing.
__global__ void bfsForwardKernel(
    const int* rowPtr,
    const int* colIdx,
    int* dist,                  // [batchSize * N]
    double* sigma,              // [batchSize * N]
    const int2* currFrontier,   // (vertex, src_batch_idx)
    int currFrontierSize,
    int2* nextFrontier,
    int* nextFrontierSize,
    int2* stack,
    int* stackSize,
    int currentLevel,
    int N
);

// Backward dependency accumulation — called once per BFS level, deepest first.
// Reads delta[s*N+v] (already accumulated from deeper levels) and propagates
// (sigma_u / sigma_v) * (1 + delta_v) to each predecessor u.
__global__ void dependencyBackwardKernel(
    const int* rowPtr,
    const int* colIdx,
    const int* dist,            // [batchSize * N]
    const double* sigma,        // [batchSize * N]
    double* delta,              // [batchSize * N]
    double* BC,                 // [N]
    const int2* stack,
    int stackStart,
    int stackEnd,
    const int* sourceVertices,  // [batchSize]
    int N
);

#endif // KERNELS_CUH