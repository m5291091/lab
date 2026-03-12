#!/bin/bash -l
#PBS -q small-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=02:00:00
#PBS -N bc_profile
#PBS -W group_list=gj17
#PBS -j oe

cd ${PBS_O_WORKDIR}

BUILD_DIR=${PBS_O_WORKDIR}/build_miyabi
GRAPH_SMALL="${PBS_O_WORKDIR}/../data/benchmark_11023_62184"
GRAPH_LARGE="${PBS_O_WORKDIR}/../data/325557_3216152"

mkdir -p ${BUILD_DIR}/result_profile

# ---------------------------------------------------------------
# nsys / ncu のパス解決
# nvidia/25.9 モジュール (batch ジョブで自動ロード) が
# compilers/bin を PATH に追加する。バッチ実行時に PATH に
# 含まれない場合は絶対パスをフォールバックとして使用する。
# ---------------------------------------------------------------
_NV_BIN="/work/opt/local/aarch64/cores/nvidia/25.9/Linux_aarch64/25.9/compilers/bin"
NSYS=$(command -v nsys 2>/dev/null || echo "${_NV_BIN}/nsys")
NCU=$(command -v ncu  2>/dev/null || echo "${_NV_BIN}/ncu")
echo "Using nsys: ${NSYS}  ($(${NSYS} --version 2>&1 | head -1))"
echo "Using ncu:  ${NCU}   ($(${NCU}  --version 2>&1 | head -1))"

# ================================================================
#  Nsight Systems プロファイリング (タイムライン + NVTX フェーズ可視化)
# ================================================================
echo "=== [1/4] Nsight Systems: gpu (11K graph) ==="
${NSYS} profile \
    --output="${BUILD_DIR}/result_profile/nsys_gpu_11023" \
    --stats=true \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    ${BUILD_DIR}/brandes_runner gpu ${GRAPH_SMALL} \
    > ${BUILD_DIR}/result_profile/nsys_gpu_11023.log 2>&1

echo "=== [2/4] Nsight Systems: gpu_opt (11K graph) ==="
${NSYS} profile \
    --output="${BUILD_DIR}/result_profile/nsys_gpu_opt_11023" \
    --stats=true \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    ${BUILD_DIR}/brandes_runner gpu_opt ${GRAPH_SMALL} \
    > ${BUILD_DIR}/result_profile/nsys_gpu_opt_11023.log 2>&1

echo "=== [3/4] Nsight Systems: gpu_opt (325K graph) ==="
${NSYS} profile \
    --output="${BUILD_DIR}/result_profile/nsys_gpu_opt_325K" \
    --stats=true \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    ${BUILD_DIR}/brandes_runner gpu_opt ${GRAPH_LARGE} \
    > ${BUILD_DIR}/result_profile/nsys_gpu_opt_325K.log 2>&1

# ================================================================
#  Nsight Compute プロファイリング (SM 占有率・L2 キャッシュ・DRAM 帯域)
#  計測指標:
#    sm__throughput.avg.pct_of_peak_sustained_elapsed  : SM 利用率
#    l1tex__t_sector_hit_rate.pct                       : L1 ヒット率
#    lts__t_sector_hit_rate.pct                         : L2 ヒット率
#    dram__bytes.sum.per_second                         : DRAM 帯域 (HBM3)
#    nvlrx__bytes.sum.per_second                        : NVLink 受信帯域 (C2C)
# ================================================================
echo "=== [4/4] Nsight Compute: gpu vs gpu_managed vs gpu_opt (11K graph) ==="

for IMPL in gpu gpu_managed gpu_readmostly gpu_opt; do
    echo "  ncu: ${IMPL}"
    ${NCU} \
        --target-processes all \
        --metrics \
"sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
dram__bytes.sum.per_second,\
nvlrx__bytes.sum.per_second,\
nvltx__bytes.sum.per_second" \
        --csv \
        --output "${BUILD_DIR}/result_profile/ncu_${IMPL}_11023.ncu-rep" \
        ${BUILD_DIR}/brandes_runner ${IMPL} ${GRAPH_SMALL} \
        > ${BUILD_DIR}/result_profile/ncu_${IMPL}_11023.csv \
        2> ${BUILD_DIR}/result_profile/ncu_${IMPL}_11023.log || true
done

echo ""
echo "=== Profile Summary ==="
ls -lh ${BUILD_DIR}/result_profile/
