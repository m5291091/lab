#!/bin/bash -l
# バッチサイズ感度分析スクリプト
# BC_BATCH_OVERRIDE 環境変数で gpu_opt のバッチサイズ上限を制御し
# 実行時間と TEPS の変化を記録する
#
# 使用方法 (インタラクティブジョブ内):
#   ./scripts/run_batchsize_sweep.sh [graph_file]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_miyabi"
RESULT_DIR="${BUILD_DIR}/result_batchsize_sweep"
RUNNER="${BUILD_DIR}/brandes_runner"

GRAPH="${1:-${SCRIPT_DIR}/../data/benchmark_11023_62184}"
GNAME="$(basename "$GRAPH")"

mkdir -p "${RESULT_DIR}"

echo "========================================"
echo "  バッチサイズ感度分析 (gpu_opt)"
echo "  グラフ: ${GNAME}"
echo "========================================"

echo "BatchSize	Graph	Time_sec	GTEPS	BFS_sec	Back_sec" \
    > "${RESULT_DIR}/batchsize_${GNAME}.tsv"

# テスト対象バッチサイズ (様々なスループット特性を網羅)
for BATCH in 32 64 128 256 512; do
    echo ""
    echo ">>> バッチサイズ: ${BATCH}"

    # CUDA イベントの位相タイミングも stdout ではなく stderr に出るので tee で両方キャプチャ
    result=$(BC_BATCH_OVERRIDE=${BATCH} "${RUNNER}" gpu_opt "${GRAPH}" 2>/tmp/batchsweep_stderr.txt)
    bfs_sec=$(grep '\[GPU Phase\]\|\[Phase\]' /tmp/batchsweep_stderr.txt | \
              grep -oP 'BFS: \K[0-9.]+' | head -1)
    back_sec=$(grep '\[GPU Phase\]\|\[Phase\]' /tmp/batchsweep_stderr.txt | \
               grep -oP 'Backward: \K[0-9.]+' | head -1)

    # stdout の TSV 行から Time_sec と GTEPS を取得
    time_sec=$(echo "${result}" | awk '{print $3}')
    gteps=$(echo "${result}" | awk '{print $4}')

    echo "  Time: ${time_sec} sec, GTEPS: ${gteps}"
    echo "${BATCH}	${GNAME}	${time_sec}	${gteps}	${bfs_sec:-N/A}	${back_sec:-N/A}" \
        >> "${RESULT_DIR}/batchsize_${GNAME}.tsv"
done

# 56K グラフでも計測 (グラフサイズの影響を確認)
GRAPH2="${SCRIPT_DIR}/../data/56438_300801"
if [ -f "${GRAPH2}" ]; then
    GNAME2="$(basename "$GRAPH2")"
    echo ""
    echo ">>> 追加グラフ: ${GNAME2}"
    echo "BatchSize	Graph	Time_sec	GTEPS	BFS_sec	Back_sec" \
        >> "${RESULT_DIR}/batchsize_${GNAME2}.tsv"
    for BATCH in 32 64 128 256 512; do
        result=$(BC_BATCH_OVERRIDE=${BATCH} "${RUNNER}" gpu_opt "${GRAPH2}" 2>/tmp/batchsweep_stderr.txt)
        bfs_sec=$(grep '\[GPU Phase\]\|\[Phase\]' /tmp/batchsweep_stderr.txt | \
                  grep -oP 'BFS: \K[0-9.]+' | head -1)
        back_sec=$(grep '\[GPU Phase\]\|\[Phase\]' /tmp/batchsweep_stderr.txt | \
                   grep -oP 'Backward: \K[0-9.]+' | head -1)
        time_sec=$(echo "${result}" | awk '{print $3}')
        gteps=$(echo "${result}" | awk '{print $4}')
        echo "  Batch=${BATCH}: ${time_sec} sec, ${gteps} GTEPS"
        echo "${BATCH}	${GNAME2}	${time_sec}	${gteps}	${bfs_sec:-N/A}	${back_sec:-N/A}" \
            >> "${RESULT_DIR}/batchsize_${GNAME2}.tsv"
    done
fi

echo ""
echo "========================================"
echo "  感度分析結果 (${GNAME}):"
column -t -s $'\t' "${RESULT_DIR}/batchsize_${GNAME}.tsv"
echo ""
echo "  最速バッチサイズ:"
sort -t $'\t' -k3 -n "${RESULT_DIR}/batchsize_${GNAME}.tsv" | grep -v "^Batch" | head -1 | \
    awk -F'\t' '{printf "  BatchSize=%s: %.4f sec, %.4f GTEPS\n", $1, $3, $4}'
echo "========================================"
