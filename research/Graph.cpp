#include "Graph.h"

// コンストラクタ／デストラクタ
Graph::Graph() : nodeCount(0), edgeCount(0),
                 adjacencyList(nullptr), adjacencyListPointers(nullptr) {}

Graph::~Graph() {
    delete[] adjacencyList;
    delete[] adjacencyListPointers;
}

// 各種メソッド本体
int Graph::getNodeCount() const {
    return nodeCount;
}

int Graph::getEdgeCount() const {
    return edgeCount;
}

void Graph::readGraph() {
    // CSR フォーマットの読み込み
    cin >> nodeCount >> edgeCount;
    adjacencyListPointers = new int[nodeCount + 1];
    adjacencyList = new int[2 * edgeCount];
    for (int i = 0; i <= nodeCount; ++i) {
        cin >> adjacencyListPointers[i];
    }
    for (int i = 0; i < 2 * edgeCount; ++i) {
        cin >> adjacencyList[i];
    }
}

int* Graph::getAdjacencyList() const {
    return adjacencyList;
}

int* Graph::getAdjacencyListPointers() const {
    return adjacencyListPointers;
}
