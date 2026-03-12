#!/bin/bash -l
#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=24:00:00
#PBS -N bc_baseline_regular-g
#PBS -W group_list=gj17
#PBS -j oe

# NOTE: Miyabi-G では nvidia/25.9, nv-hpcx/25.9 がバッチジョブ開始時に
#       自動ロードされるため module load は原則不要。
#       明示的に指定する場合は以下のコメントを外す:
# module load nvidia/25.9
# module load nv-hpcx/25.9

cd ${PBS_O_WORKDIR}

mkdir -p build_miyabi && cd build_miyabi
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
make -j8 2>&1 | tail -10

mkdir -p result_baseline

# ============================================================
# グラフ規模ごとの実装選択ポリシー:
#
#   [small]  頂点 ≤ 11K  → 全5実装 (sequential / omp / gpu / gpu_managed / gpu_opt)
#   [medium] 頂点 11K~500K → omp + 3GPU (sequential は数時間かかるため除外)
#   [large]  頂点 500K~2M  → 3GPU のみ (omp も実用外)
#   [xlarge] 頂点 2M+      → gpu_opt のみ (最適化実装で長時間実験)
#
# SNAP 実グラフは ../data/snap/ に格納 (存在しない場合はスキップ)
# ============================================================

DATA_DIR="${PBS_O_WORKDIR}/../data"
SNAP_DIR="${PBS_O_WORKDIR}/../data/snap"

SMALL_GRAPHS=(
    "${DATA_DIR}/benchmark_7000_41459"
    "${DATA_DIR}/benchmark_11023_62184"
)

# medium: 既存合成グラフ + SNAP medium グラフ
MEDIUM_GRAPHS=(
    "${DATA_DIR}/benchmark_85830.data"
    "${DATA_DIR}/56438_300801"
    "${SNAP_DIR}/email-EuAll"
    "${SNAP_DIR}/amazon0302"
    "${SNAP_DIR}/web-Stanford"
    "${SNAP_DIR}/web-NotreDame"
    "${SNAP_DIR}/amazon0505"
)

# large: 既存最大グラフ + SNAP large グラフ
LARGE_GRAPHS=(
    "${DATA_DIR}/325557_3216152"
    "${SNAP_DIR}/web-Google"
    "${SNAP_DIR}/roadNet-PA"
    "${SNAP_DIR}/roadNet-TX"
    "${SNAP_DIR}/roadNet-CA"
)

# xlarge: 高密度大規模グラフ (gpu_opt のみ, 長時間)
XLARGE_GRAPHS=(
    "${SNAP_DIR}/as-skitter"
    "${SNAP_DIR}/soc-Pokec"
)

echo "Implementation	Graph	Time_sec	GTEPS" > result_baseline/summary.tsv
# フェーズ別時間はステレオ出力 (stderr) に記録
> result_baseline/phase_timing.log

# ---- ヘルパー: グラフ存在確認 ----
run_impl() {
    local impl="$1"
    local graph="$2"
    if [ ! -f "$graph" ]; then
        echo "  スキップ (グラフなし: $graph)"
        return
    fi
    echo "  [$impl]"
    ./brandes_runner "$impl" "$graph" \
        >> result_baseline/summary.tsv \
        2>> result_baseline/phase_timing.log
}

# ---- small: 全6実装 ----
echo "=== [small] 小規模グラフ: 全実装計測 ==="
for GRAPH in "${SMALL_GRAPHS[@]}"; do
    [ -f "$GRAPH" ] || continue
    echo "--- $(basename $GRAPH) ---"
    for IMPL in sequential omp gpu gpu_managed gpu_readmostly gpu_opt; do
        run_impl "$IMPL" "$GRAPH"
    done
done

# ---- medium: omp + 4GPU ----
echo ""
echo "=== [medium] 中規模グラフ (11K~500K): omp + GPU 実装 ==="
for GRAPH in "${MEDIUM_GRAPHS[@]}"; do
    [ -f "$GRAPH" ] || { echo "  スキップ (未取得): $(basename $GRAPH)"; continue; }
    echo "--- $(basename $GRAPH) ---"
    for IMPL in omp gpu gpu_managed gpu_readmostly gpu_opt; do
        run_impl "$IMPL" "$GRAPH"
    done
done

# ---- large: GPU 3実装 (GPU_Managed は roadNet の超長距離 BFS で数時間かかるためスキップ) ----
echo ""
echo "=== [large] 大規模グラフ (500K~2M): GPU + GPU_ReadMostly + GPU_Opt ==="
for GRAPH in "${LARGE_GRAPHS[@]}"; do
    [ -f "$GRAPH" ] || { echo "  スキップ (未取得): $(basename $GRAPH)"; continue; }
    echo "--- $(basename $GRAPH) ---"
    for IMPL in gpu gpu_readmostly gpu_opt; do
        run_impl "$IMPL" "$GRAPH"
    done
done

# ---- xlarge: gpu_opt のみ ----
echo ""
echo "=== [xlarge] 超大規模グラフ (2M+): gpu_opt のみ ==="
for GRAPH in "${XLARGE_GRAPHS[@]}"; do
    [ -f "$GRAPH" ] || { echo "  スキップ (未取得): $(basename $GRAPH)"; continue; }
    echo "--- $(basename $GRAPH) ---"
    run_impl "gpu_opt" "$GRAPH"
done

echo ""
echo "=== Results ==="
cat result_baseline/summary.tsv
echo ""
echo "=== Phase Timing Log ==="
cat result_baseline/phase_timing.log
