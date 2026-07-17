#ifndef RUNNER_HPP
#define RUNNER_HPP

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "graph.hpp"

// ============================================================
//  実験ランナー共通ヘルパ
//  全実行ファイル (run_benchmark / run_ablation / run_pathmerge_sweep) が
//  共有する、グラフパス解決・グラフ読込・計測/レポートのユーティリティ。
//
//  stdout/stderr 契約:
//    - 通常時 stdout : Impl<TAB>Graph<TAB>Time_sec<TAB>GTEPS
//    - dump_bc 時 stdout : ヘッダ行 + node_idx<TAB>bc_value
//    - stderr : フェーズ計測・最大 BC サマリ・進捗
// ============================================================

// data/ を祖先ディレクトリから探索してグラフの実パスを解決する。
// 見つからない場合は std::nullopt。
std::optional<std::string> resolve_graph_path(const std::string& graph_path);

// 読み込み済み CSR のホスト側整合性検査 (BC 計測区間の外側で実行する)。
// ptr[0]==0 / ptr 単調非減少 / ptr[n]==2m / 全頂点 ID が [0,n) を O(n+m) で検査し、
// 違反時はグラフ名・index・値を stderr に出して false を返す (補正・skip はしない)。
// 対称性・多重度・連結成分などの重い検査は tools/validate_graph_csr.py が担当する。
bool validate_graph(const Graph& graph, const std::string& path);

// 解決済みパスからグラフを読み込む (freopen(stdin) + readGraph + setSourcePath)。
// 読み込み後に validate_graph を通し、不正 CSR は GPU 実行前に弾く。成功時 true。
bool load_graph(const std::string& resolved_path, Graph& graph);

// 実装を 1 回計測し、GTEPS・最大 BC を算出して規約通り stdout/stderr に出力する。
// dump_bc=true の場合、全 BC 値を stdout にダンプ (正確性検証用)。
// 戻り値: 実行時間 [秒] (掃引での最適値探索などに利用)。
double run_brandes(const std::string& impl_name,
                   const std::string& graph_path,
                   std::function<std::vector<double>(Graph&)> brandes_func,
                   Graph& graph,
                   bool dump_bc = false);

#endif // RUNNER_HPP
