#ifndef GALLIOT_H
#define GALLIOT_H

#include "baseline_graph.h"
#include <vector>

// Batched multi-source Brandes BC using per-source dist/sigma/delta arrays.
// Each array is indexed as [source_batch_idx * N + vertex].
class GalloitBC {
private:
    CSRGraph* graph;
    int batchSize;
    bool verbose;

    // Accumulated BC scores [N]
    double* d_BC;

    // Per-source arrays [batchSize * N]
    int*    d_dist;
    double* d_sigma;
    double* d_delta;

    // Source vertex IDs for the current batch [batchSize]
    int* d_sourceVertices;

    // Double-buffered frontier queues (vertex, src_batch_idx)
    int2* d_Qcurr;
    int2* d_Qnext;
    int*  d_QcurrSize;
    int*  d_QnextSize;

    // BFS stack (level-ordered) for backward pass
    int2* d_stack;
    int*  d_stackSize;

    // Host-side level boundaries: levelBounds[L] = start offset of level L in d_stack
    std::vector<int> levelBounds;

    void allocateMemory();
    void freeMemory();

public:
    GalloitBC(CSRGraph* g, int batch = 64, bool verb = true);
    ~GalloitBC();

    void initializeBatch(const std::vector<int>& sources);
    void forwardPhase(const std::vector<int>& sources);
    void backwardPhase(const std::vector<int>& sources);
    double* getBC() const { return d_BC; }

    std::vector<double> compute();
    static void normalize(std::vector<double>& bc, int numVertices);
};

#endif // GALLIOT_H