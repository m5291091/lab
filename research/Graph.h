#pragma once
#include <iostream>
using namespace std;

class Graph {
public:
    Graph();
    ~Graph();

    int getNodeCount() const;
    int getEdgeCount() const;

    void readGraph();

    int* getAdjacencyList() const;
    int* getAdjacencyListPointers() const;

private:
    int nodeCount, edgeCount;
    int *adjacencyList, *adjacencyListPointers;
};
