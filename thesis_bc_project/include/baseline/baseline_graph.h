#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <string>
#include <cstdint>

// CSR (Compressed Sparse Row) Graph representation
struct CSRGraph {
    int numVertices;
    int numEdges;
    
    // CSR format arrays
    int* rowPtr;      // Size: numVertices + 1
    int* colIdx;      // Size: numEdges
    
    // Constructor
    CSRGraph() : numVertices(0), numEdges(0), rowPtr(nullptr), colIdx(nullptr) {}
    
    // Destructor
    ~CSRGraph();
    
    // Allocate memory (unified memory for GPU)
    void allocate(int vertices, int edges);
    
    // Free memory
    void free();
};

// Edge structure for loading
struct Edge {
    int src;
    int dst;
    
    Edge(int s, int d) : src(s), dst(d) {}
    
    bool operator<(const Edge& other) const {
        if (src != other.src) return src < other.src;
        return dst < other.dst;
    }
    
    bool operator==(const Edge& other) const {
        return src == other.src && dst == other.dst;
    }
};

// Function declarations
CSRGraph* loadGraphFromTSV(const std::string& filename, bool verbose = true);
void convertToCSR(const std::vector<Edge>& edges, int numVertices, CSRGraph* graph);
void printGraphStats(const CSRGraph* graph);

#endif // GRAPH_H