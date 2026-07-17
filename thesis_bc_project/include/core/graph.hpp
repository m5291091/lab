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

    // CSR を token 単位で厳密に読み込む。ヘッダ・ptr・adj の不足/非整数 token、
    // および期待する 2m adjacency の後に残る余剰 token を拒否する。
    // trailing whitespace のみは許容する。失敗理由は getReadError() で取得する。
    bool readGraph();
    const std::string& getReadError() const;

    int* getAdjacencyList() const;
    int* getAdjacencyListPointers() const;

    // 元のグラフファイルパス
    void setSourcePath(const std::string& path);
    const std::string& getSourcePath() const;

private:
    int nodeCount, edgeCount;
    int *adjacencyList, *adjacencyListPointers;
    std::string sourcePath;
    std::string readError;
};

#endif // GRAPH_HPP
