#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <iostream>
#include <string>

class Graph {
public:
    Graph();
    ~Graph();

    int getNodeCount() const;
    int getEdgeCount() const;

    void readGraph();

    int* getAdjacencyList() const;
    int* getAdjacencyListPointers() const;

    // 元のグラフファイルパス
    void setSourcePath(const std::string& path);
    const std::string& getSourcePath() const;

private:
    int nodeCount, edgeCount;
    int *adjacencyList, *adjacencyListPointers;
    std::string sourcePath;
};

#endif // GRAPH_HPP
