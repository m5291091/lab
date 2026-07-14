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

// 解決済みパスからグラフを読み込む (freopen(stdin) + readGraph + setSourcePath)。
// 成功時 true。
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
