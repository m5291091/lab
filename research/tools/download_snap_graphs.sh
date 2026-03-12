#!/bin/bash
# SNAP 実グラフのダウンロードと CSR 変換スクリプト (拡張版)
#
# 取得グラフ一覧:
#
#  [medium]  ~100K-500K ノード  → omp + gpu + gpu_managed + gpu_opt
#   email-EuAll   : 265K nodes,  420K edges  (EU メール通信ネットワーク)
#   amazon0302    : 262K nodes,  1.2M edges  (Amazon 商品共購買ネットワーク)
#   web-Stanford  : 281K nodes,  2.3M edges  (Stanford Web グラフ)
#   web-NotreDame : 325K nodes,  1.5M edges  (Notre Dame Web グラフ)
#   amazon0505    : 410K nodes,  3.4M edges  (Amazon 商品共購買ネットワーク大)
#
#  [large]   ~500K-2M ノード    → gpu + gpu_managed + gpu_opt
#   web-Google    : 875K nodes,  5.1M edges  (Google Web グラフ)
#   roadNet-PA    : 1.09M nodes, 1.5M edges  (ペンシルバニア道路ネットワーク)
#   roadNet-TX    : 1.38M nodes, 1.9M edges  (テキサス道路ネットワーク)
#   roadNet-CA    : 1.97M nodes, 2.8M edges  (カリフォルニア道路ネットワーク)
#
#  [xlarge]  ~1.5M+ ノード (高密度) → gpu_opt のみ推奨
#   as-skitter    : 1.70M nodes, 11M edges   (インターネット AS トポロジー)
#   soc-Pokec     : 1.63M nodes, 30.6M edges (スロバキア SNS ソーシャルグラフ)
#
# 使用方法:
#   ./tools/download_snap_graphs.sh [options] [output_dir]
#
#   オプション:
#     --category CATEGORY  取得カテゴリ: medium / large / xlarge / all (デフォルト: medium large)
#     --dry-run            ダウンロードせず一覧表示のみ
#     --force              既存CSRファイルを上書き再変換
#
#   例:
#     ./tools/download_snap_graphs.sh                       # medium + large のみ
#     ./tools/download_snap_graphs.sh --category xlarge     # xlarge のみ
#     ./tools/download_snap_graphs.sh --category all        # 全カテゴリ
#     ./tools/download_snap_graphs.sh --dry-run             # 一覧確認
#
# 注意: Miyabi-G の compute ノードはインターネット非接続の場合があります。
#       ログインノードで実行するか、外部からデータを scp してください。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TOOL="${SCRIPT_DIR}/tools/snap_to_csr.py"

# ---- オプション解析 ----
SNAP_DIR=""
CATEGORIES=""
DRY_RUN=0
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --category)
            shift
            CATEGORIES="$1"
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        --force)
            FORCE=1
            ;;
        -*)
            echo "不明なオプション: $1" >&2
            exit 1
            ;;
        *)
            SNAP_DIR="$1"
            ;;
    esac
    shift
done

SNAP_DIR="${SNAP_DIR:-${SCRIPT_DIR}/../data/snap}"
CATEGORIES="${CATEGORIES:-medium large}"

mkdir -p "${SNAP_DIR}"

# ---- グラフ定義テーブル ----
# 形式: "カテゴリ|グラフ名|URL|有向グラフか(directed/undirected)|説明"
GRAPHS=(
    "medium|email-EuAll|https://snap.stanford.edu/data/email-EuAll.txt.gz|directed|EU研究機関メール通信 (265K/420K)"
    "medium|amazon0302|https://snap.stanford.edu/data/amazon0302.txt.gz|undirected|Amazon共購買ネット (262K/1.2M)"
    "medium|web-Stanford|https://snap.stanford.edu/data/web-Stanford.txt.gz|directed|Stanford Webグラフ (281K/2.3M)"
    "medium|web-NotreDame|https://snap.stanford.edu/data/web-NotreDame.txt.gz|directed|Notre Dame Webグラフ (325K/1.5M)"
    "medium|amazon0505|https://snap.stanford.edu/data/amazon0505.txt.gz|undirected|Amazon共購買ネット大 (410K/3.4M)"
    "large|web-Google|https://snap.stanford.edu/data/web-Google.txt.gz|directed|Google Webグラフ (875K/5.1M)"
    "large|roadNet-PA|https://snap.stanford.edu/data/roadNet-PA.txt.gz|undirected|ペンシルバニア道路 (1.09M/1.5M)"
    "large|roadNet-TX|https://snap.stanford.edu/data/roadNet-TX.txt.gz|undirected|テキサス道路 (1.38M/1.9M)"
    "large|roadNet-CA|https://snap.stanford.edu/data/roadNet-CA.txt.gz|undirected|カリフォルニア道路 (1.97M/2.8M)"
    "xlarge|as-skitter|https://snap.stanford.edu/data/as-skitter.txt.gz|undirected|インターネットASトポロジー (1.7M/11M)"
    "xlarge|soc-Pokec|https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz|directed|スロバキアSNS (1.63M/30.6M)"
)

# ---- 関数定義 ----

category_enabled() {
    local cat="$1"
    if [[ "$CATEGORIES" == "all" ]]; then return 0; fi
    if echo "$CATEGORIES" | grep -qw "$cat"; then return 0; fi
    return 1
}

download_and_convert() {
    local cat="$1"
    local name="$2"
    local url="$3"
    local graph_type="$4"

    local gz_file="${SNAP_DIR}/$(basename "$url")"
    local csr_file="${SNAP_DIR}/${name}"

    if [ "${DRY_RUN}" -eq 1 ]; then
        printf "  [%-8s] %-15s  %s\n" "$cat" "$name" "$url"
        return
    fi

    echo ""
    echo ">>> [${cat}] ${name}"

    if [ -f "${csr_file}" ] && [ "${FORCE}" -eq 0 ]; then
        local meta
        meta=$(head -1 "${csr_file}" 2>/dev/null)
        echo "  スキップ (変換済み: nodes=$(echo "$meta" | awk '{print $1}'), edges=$(echo "$meta" | awk '{print $2}'))"
        return
    fi

    # ダウンロード
    echo "  ダウンロード中: $(basename "$url")"
    if ! curl -L --fail --progress-bar -o "${gz_file}" "${url}"; then
        echo "  警告: ダウンロード失敗 (${url})" >&2
        echo "  手動取得: curl -L -o ${gz_file} ${url}" >&2
        return 1
    fi
    echo "  完了: $(du -h "${gz_file}" | cut -f1)"

    # CSR 変換 (.gz を直接読み込み → gunzip 不要)
    echo "  CSR 変換中..."
    local dir_flag=""
    if [ "${graph_type}" = "directed" ]; then
        dir_flag="--directed"
    fi

    if python3 "${TOOL}" "${gz_file}" "${csr_file}" ${dir_flag}; then
        rm -f "${gz_file}"
        local meta
        meta=$(head -1 "${csr_file}" 2>/dev/null)
        echo "  変換完了: nodes=$(echo "$meta" | awk '{print $1}'), edges=$(echo "$meta" | awk '{print $2}')"
    else
        echo "  警告: 変換失敗 (${name})" >&2
        rm -f "${gz_file}" "${csr_file}"
        return 1
    fi
}

# ---- メイン処理 ----

echo "========================================"
echo "  SNAP 実グラフ ダウンロード & CSR 変換"
echo "  出力先 : ${SNAP_DIR}"
echo "  カテゴリ: ${CATEGORIES}"
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "  モード  : DRY-RUN (ダウンロードなし)"
fi
echo "========================================"

if [ "${DRY_RUN}" -eq 1 ]; then
    echo ""
    echo "取得予定グラフ:"
fi

TOTAL=0
SKIPPED=0
FAILED=0

for entry in "${GRAPHS[@]}"; do
    IFS='|' read -r cat name url graph_type desc <<< "$entry"
    if ! category_enabled "$cat"; then
        continue
    fi
    TOTAL=$((TOTAL + 1))
    if ! download_and_convert "$cat" "$name" "$url" "$graph_type"; then
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
echo "  変換完了グラフ一覧 (${SNAP_DIR}):"
echo ""
printf "  %-20s  %10s  %12s  %8s\n" "グラフ名" "ノード数" "エッジ数" "サイズ"
printf "  %-20s  %10s  %12s  %8s\n" "--------------------" "----------" "------------" "--------"
for entry in "${GRAPHS[@]}"; do
    IFS='|' read -r cat name url graph_type desc <<< "$entry"
    if ! category_enabled "$cat"; then continue; fi
    csr="${SNAP_DIR}/${name}"
    if [ -f "$csr" ]; then
        read -r nodes edges <<< "$(head -1 "$csr")"
        sz=$(du -h "$csr" | cut -f1)
        printf "  %-20s  %10s  %12s  %8s  [%s]\n" "$name" "$nodes" "$edges" "$sz" "$cat"
    fi
done
echo ""
echo "  使用例:"
echo "    ${SCRIPT_DIR}/build_miyabi/brandes_runner gpu_opt ${SNAP_DIR}/roadNet-CA"
echo "    ${SCRIPT_DIR}/build_miyabi/brandes_runner gpu_opt ${SNAP_DIR}/roadNet-PA"
echo "========================================"
