#ifndef ABLATION_CONFIG_HPP
#define ABLATION_CONFIG_HPP

#include <string>
#include <vector>

#include "graph.hpp"

// ============================================================
//  アブレーション実験用の構成定義
//
//  論文の 3 つの工夫を個別に ON/OFF し、それぞれがどれだけ実行時間を
//  短縮したかを測定する。ブランチダイバージェンス等の実行時オーバーヘッドを
//  避けるため、各工夫は CUDA カーネル内の if ではなく C++ テンプレートの
//  コンパイル時分岐で切り替える (host_ablation.cu 参照)。
// ============================================================

struct AblationConfig {
    bool hybrid_bfs = true;  // ① ハイブリッド (トップダウン/ボトムアップ方向最適化) BFS
    bool warp_coop  = true;  // ② Warp 協調による依存関係蓄積 (shfl 還元)
    bool async_init = true;  // ③ 2 ストリーム非同期初期化 (ダブルバッファリング)
};

// 構成の短いラベルを返す (例: "H1_W0_A1")。ログ/TSV の識別子に使う。
std::string ablation_label(const AblationConfig& cfg);

// 人間可読な説明ラベル (例: "hybrid=ON warp=OFF async=ON")。
std::string ablation_describe(const AblationConfig& cfg);

// ランタイムで受け取った構成を、8 通りのコンパイル時テンプレート実体の
// いずれかへ振り分けて実行する。カーネル内で 3 フラグを分岐しないため、
// 各実体はブランチのない専用カーネルになる。
std::vector<double> brandes_gpu_opt_ablation(Graph& G, const AblationConfig& cfg);

#endif // ABLATION_CONFIG_HPP
