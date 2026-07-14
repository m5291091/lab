#include "galliot.h"
#include "galliot_kernels.cuh"
#include "baseline_utils.h"
#include <iostream>
#include <cstring>
#include <algorithm>

#define BLOCK_SIZE 256

// ================================================================
//  Construction / Destruction
// ================================================================

GalloitBC::GalloitBC(CSRGraph* g, int batch, bool verb)
    : graph(g), batchSize(batch), verbose(verb),
      d_BC(nullptr), d_dist(nullptr), d_sigma(nullptr), d_delta(nullptr),
      d_sourceVertices(nullptr),
      d_Qcurr(nullptr), d_Qnext(nullptr),
      d_QcurrSize(nullptr), d_QnextSize(nullptr),
      d_stack(nullptr), d_stackSize(nullptr)
{
    allocateMemory();
}

GalloitBC::~GalloitBC() {
    freeMemory();
}

// ================================================================
//  Memory management
// ================================================================

void GalloitBC::allocateMemory() {
    int N = graph->numVertices;
    size_t NxB = (size_t)N * batchSize;

    // BC scores (accumulated across all batches)
    CUDA_CHECK(cudaMallocManaged(&d_BC, (size_t)N * sizeof(double)));
    std::memset(d_BC, 0, (size_t)N * sizeof(double));

    // Per-source arrays
    CUDA_CHECK(cudaMallocManaged(&d_dist,  NxB * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_sigma, NxB * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&d_delta, NxB * sizeof(double)));

    // Source vertex IDs
    CUDA_CHECK(cudaMallocManaged(&d_sourceVertices, batchSize * sizeof(int)));

    // Frontier (double-buffered)
    CUDA_CHECK(cudaMallocManaged(&d_Qcurr, NxB * sizeof(int2)));
    CUDA_CHECK(cudaMallocManaged(&d_Qnext, NxB * sizeof(int2)));
    CUDA_CHECK(cudaMallocManaged(&d_QcurrSize, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_QnextSize, sizeof(int)));

    // Stack
    CUDA_CHECK(cudaMallocManaged(&d_stack, NxB * sizeof(int2)));
    CUDA_CHECK(cudaMallocManaged(&d_stackSize, sizeof(int)));

    if (verbose) {
        size_t totalMB = ((size_t)N * sizeof(double)
            + NxB * (sizeof(int) + sizeof(double) * 2)
            + batchSize * sizeof(int)
            + NxB * sizeof(int2) * 3) / (1024 * 1024);
        std::cout << "  > [PathMerge Alloc] ~" << totalMB << " MB" << std::endl;
    }
}

void GalloitBC::freeMemory() {
    if (d_BC)             cudaFree(d_BC);
    if (d_dist)           cudaFree(d_dist);
    if (d_sigma)          cudaFree(d_sigma);
    if (d_delta)          cudaFree(d_delta);
    if (d_sourceVertices) cudaFree(d_sourceVertices);
    if (d_Qcurr)          cudaFree(d_Qcurr);
    if (d_Qnext)          cudaFree(d_Qnext);
    if (d_QcurrSize)      cudaFree(d_QcurrSize);
    if (d_QnextSize)      cudaFree(d_QnextSize);
    if (d_stack)          cudaFree(d_stack);
    if (d_stackSize)      cudaFree(d_stackSize);
}

// ================================================================
//  Per-batch initialisation
// ================================================================

void GalloitBC::initializeBatch(const std::vector<int>& sources) {
    int N = graph->numVertices;
    int numSources = (int)sources.size();
    size_t NxB = (size_t)N * batchSize;

    // Reset per-source arrays
    int blocks = (int)((NxB + BLOCK_SIZE - 1) / BLOCK_SIZE);
    initBatchKernel<<<blocks, BLOCK_SIZE>>>(d_dist, d_sigma, d_delta, N, batchSize);
    CUDA_CHECK_LAST();

    // Copy source vertex IDs to device
    std::memcpy(d_sourceVertices, sources.data(), numSources * sizeof(int));

    // Reset counters
    *d_QcurrSize = 0;
    *d_QnextSize = 0;
    *d_stackSize = 0;

    // Initialise sources (dist=0, sigma=1, add to frontier & stack)
    int srcBlocks = (numSources + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initSourcesKernel<<<srcBlocks, BLOCK_SIZE>>>(
        d_sourceVertices, numSources,
        d_dist, d_sigma,
        d_Qcurr, d_QcurrSize,
        d_stack, d_stackSize,
        N
    );
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    levelBounds.clear();
}

// ================================================================
//  Forward BFS phase
// ================================================================

void GalloitBC::forwardPhase(const std::vector<int>& /* sources */) {
    int N = graph->numVertices;
    int currentLevel = 0;

    // Level 0 entries are already in the stack from initializeBatch
    levelBounds.push_back(0);                  // level 0 starts at offset 0
    levelBounds.push_back(*d_stackSize);       // level 1 starts here

    while (*d_QcurrSize > 0) {
        int qSize = *d_QcurrSize;
        *d_QnextSize = 0;

        int blocks = (qSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bfsForwardKernel<<<blocks, BLOCK_SIZE>>>(
            graph->rowPtr, graph->colIdx,
            d_dist, d_sigma,
            d_Qcurr, qSize,
            d_Qnext, d_QnextSize,
            d_stack, d_stackSize,
            currentLevel, N
        );
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap frontier buffers
        std::swap(d_Qcurr, d_Qnext);
        std::swap(d_QcurrSize, d_QnextSize);

        currentLevel++;
        levelBounds.push_back(*d_stackSize);   // next level starts here

        if (currentLevel > N) {
            std::cerr << "ERROR: BFS exceeded vertex count — possible cycle\n";
            break;
        }
    }
}

// ================================================================
//  Backward dependency-accumulation phase (level by level)
// ================================================================

void GalloitBC::backwardPhase(const std::vector<int>& /* sources */) {
    int N = graph->numVertices;
    int numLevels = (int)levelBounds.size() - 1;

    // Process from deepest level down to level 1 (level 0 = sources → skip)
    for (int L = numLevels - 1; L >= 1; L--) {
        int stackStart = levelBounds[L];
        int stackEnd   = levelBounds[L + 1];
        int count      = stackEnd - stackStart;
        if (count <= 0) continue;

        int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dependencyBackwardKernel<<<blocks, BLOCK_SIZE>>>(
            graph->rowPtr, graph->colIdx,
            d_dist, d_sigma, d_delta, d_BC,
            d_stack, stackStart, stackEnd,
            d_sourceVertices, N
        );
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// ================================================================
//  High-level compute (standalone convenience)
// ================================================================

std::vector<double> GalloitBC::compute() {
    int N = graph->numVertices;
    Timer timer;

    timer.startTimer();

    int numBatches = (N + batchSize - 1) / batchSize;
    for (int b = 0; b < numBatches; b++) {
        int start = b * batchSize;
        int end   = std::min(start + batchSize, N);

        std::vector<int> sources;
        sources.reserve(end - start);
        for (int i = start; i < end; i++)
            sources.push_back(i);

        if (verbose)
            printProgress(b + 1, numBatches, "Processing batches");

        initializeBatch(sources);
        forwardPhase(sources);
        backwardPhase(sources);
    }

    float elapsedMs = timer.stopTimer();
    if (verbose)
        std::cout << "Completed in " << elapsedMs / 1000.0f << " s\n";

    std::vector<double> bc(N);
    std::memcpy(bc.data(), d_BC, N * sizeof(double));
    normalize(bc, N);
    return bc;
}

void GalloitBC::normalize(std::vector<double>& bc, int numVertices) {
    if (numVertices <= 2) return;
    double norm = (double)(numVertices - 1) * (numVertices - 2) / 2.0;
    for (auto& val : bc)
        val /= norm;
}