#!/bin/bash -l
# NVLink-C2C / HBM3 帯域計測スクリプト
# GH200 固有の帯域特性を定量的に報告する
#
# 使用方法: qsub scripts/measure_bandwidth.sh
# または インタラクティブジョブ内で直接実行

#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=00:30:00
#PBS -N bc_bandwidth
#PBS -W group_list=gj17
#PBS -j oe

cd ${PBS_O_WORKDIR:-$(dirname "$(readlink -f "$0")")/../..}

BUILD_DIR="${PBS_O_WORKDIR:-$(pwd)}/build_miyabi"
RESULT_DIR="${BUILD_DIR}/result_bandwidth"
mkdir -p "${RESULT_DIR}"

RUNNER="${BUILD_DIR}/bandwidth_benchmark"
if [ ! -f "${RUNNER}" ]; then
    echo "ERROR: bandwidth_benchmark が見つかりません: ${RUNNER}"
    echo "先に cmake + make でビルドしてください"
    exit 1
fi

echo "========================================"
echo "  GH200 メモリ帯域計測"
echo "  理論値: HBM3=4020 GB/s, NVLink-C2C=900 GB/s"
echo "========================================"

# TSV ヘッダーを出力
echo "Transfer_Type	Size_GB	Time_ms	Bandwidth_GBs	Theoretical_GBs	Ratio_pct" \
    > "${RESULT_DIR}/bandwidth.tsv"

# バッファサイズを変えて複数点計測 (128 MB, 512 MB, 1 GB, 4 GB)
for SIZE_MB in 128 512 1024 4096; do
    echo ""
    echo ">>> バッファサイズ: ${SIZE_MB} MB"
    "${RUNNER}" ${SIZE_MB} 2>/dev/null >> "${RESULT_DIR}/bandwidth.tsv"
done

echo ""
echo "========================================"
echo "  計測結果:"
column -t -s $'\t' "${RESULT_DIR}/bandwidth.tsv"
echo ""
echo "  結果保存: ${RESULT_DIR}/bandwidth.tsv"
echo "========================================"

# 論文用サマリ: NVLink-C2C の帯域を確認
echo ""
echo "論文用: NVLink-C2C 実効帯域 (1 GB バッファ):"
grep "NVLink_C2C_Prefetch" "${RESULT_DIR}/bandwidth.tsv" | \
    awk -F'\t' 'NR==3 {printf "  %.1f GB/s (理論比 %s%%)\n", $4, $6}'
