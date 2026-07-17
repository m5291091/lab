#include "graph.hpp"

#include <charconv>
#include <climits>
#include <string>

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

bool Graph::readGraph() {
    // CSR フォーマットの読み込み
    readError.clear();

    auto read_int = [&](const std::string& check, const std::string& expected,
                        int& value) -> bool {
        std::string token;
        if (!(std::cin >> token)) {
            readError = "check=" + check + " expected=" + expected + " actual=EOF";
            return false;
        }
        const char* first = token.data();
        const char* last = first + token.size();
        const auto parsed = std::from_chars(first, last, value);
        if (parsed.ec != std::errc{} || parsed.ptr != last) {
            readError = "check=" + check + " expected=" + expected
                      + " actual=malformed_token('" + token + "')";
            return false;
        }
        return true;
    };

    if (!read_int("header.n", "integer", nodeCount)
        || !read_int("header.m", "integer", edgeCount))
        return false;

    // 確保前にヘッダの妥当性を確認する (不正値での巨大確保・負数確保を防ぐ)
    if (nodeCount <= 0 || edgeCount < 0) {
        readError = "check=header.range expected=n>0,m>=0 actual=n="
                  + std::to_string(nodeCount) + ",m=" + std::to_string(edgeCount);
        return false;
    }
    if (edgeCount > INT_MAX / 2) {
        readError = "check=header.adjacency_count expected=2*m<=INT_MAX actual=m="
                  + std::to_string(edgeCount);
        return false;
    }

    adjacencyListPointers = new int[nodeCount + 1];
    adjacencyList = new int[2 * edgeCount];
    for (int i = 0; i <= nodeCount; ++i) {
        if (!read_int("row_pointer[" + std::to_string(i) + "]", "integer ("
                      + std::to_string(nodeCount + 1) + " total)",
                      adjacencyListPointers[i]))
            return false;
    }
    for (int i = 0; i < 2 * edgeCount; ++i) {
        if (!read_int("adjacency[" + std::to_string(i) + "]", "integer ("
                      + std::to_string(2 * edgeCount) + " total)", adjacencyList[i]))
            return false;
    }

    // 期待した 2m 要素を読み終えた後は空白だけを許容する。追加 token は
    // 別グラフの連結や壊れた header を隠し得るため、値の種類を問わず拒否する。
    std::string extra;
    if (std::cin >> extra) {
        readError = "check=trailing_token expected=EOF_after_2m_adjacency actual='"
                  + extra + "'";
        return false;
    }
    return true;
}

const std::string& Graph::getReadError() const {
    return readError;
}

int* Graph::getAdjacencyList() const {
    return adjacencyList;
}

int* Graph::getAdjacencyListPointers() const {
    return adjacencyListPointers;
}

void Graph::setSourcePath(const std::string& path) {
    sourcePath = path;
}

const std::string& Graph::getSourcePath() const {
    return sourcePath;
}
