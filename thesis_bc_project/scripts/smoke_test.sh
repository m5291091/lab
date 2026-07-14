#!/bin/bash
# ============================================================
#  smoke_test.sh — GPU 計算ノードでの動作確認 + 正確性検証
#
#  インタラクティブノードを確保してから実行する:
#    qsub -I -q interact-g -l select=1:ncpus=72 -l walltime=02:00:00 -W group_list=gj17
#    cd thesis_bc_project
#    bash scripts/smoke_test.sh
#
#  内容:
#    1. 統合ビルド (SKIP_BUILD=1 でスキップ可)
#    2. run_benchmark スモークテスト
#    3. 正確性検証: 実装間で BC 値を diff
#    4. run_ablation (baseline / full) の動作と一致確認
#    5. run_pathmerge_sweep の短縮掃引
#
#  環境変数:
#    GRAPH       検証グラフ (既定 data/benchmark_7000_41459)
#    SKIP_BUILD  1 でビルドをスキップ
# ============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build_miyabi}"
GRAPH="${GRAPH:-${PROJECT_DIR}/data/benchmark_7000_41459}"
SKIP_BUILD="${SKIP_BUILD:-0}"

RUN_BENCH="${BUILD_DIR}/run_benchmark"
RUN_ABL="${BUILD_DIR}/run_ablation"
RUN_PM="${BUILD_DIR}/run_pathmerge_sweep"

pass=0; fail=0
ok()   { echo "  [PASS] $*"; pass=$((pass+1)); }
ng()   { echo "  [FAIL] $*"; fail=$((fail+1)); }
hr()   { echo "------------------------------------------------------------"; }

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "=== thesis_bc_project smoke test ==="
echo "Project: ${PROJECT_DIR}"
echo "Graph  : ${GRAPH}"
hr

# --- 1. Build ---
if [ "${SKIP_BUILD}" != "1" ]; then
    echo "[1] Building (統合ビルド)..."
    if AUTO_INSTALL_CMAKE="${AUTO_INSTALL_CMAKE:-1}" bash "${SCRIPT_DIR}/scripts/build_miyabi_interactive.sh"; then
        ok "build_miyabi_interactive.sh"
    else
        ng "build failed"; echo "Aborting."; exit 1
    fi
else
    echo "[1] SKIP_BUILD=1 (ビルドをスキップ)"
fi
hr

for b in "${RUN_BENCH}" "${RUN_ABL}" "${RUN_PM}"; do
    [ -x "${b}" ] && ok "exists: $(basename "${b}")" || ng "missing: ${b}"
done
hr

# --- 2. run_benchmark スモーク ---
echo "[2] run_benchmark smoke..."
if "${RUN_BENCH}" gpu_opt "${GRAPH}" > "${TMP}/bench.tsv" 2>/dev/null; then
    line="$(head -1 "${TMP}/bench.tsv")"
    [ -n "${line}" ] && ok "gpu_opt: ${line}" || ng "gpu_opt: 空の出力"
else
    ng "run_benchmark gpu_opt 失敗"
fi
hr

# --- 3. 正確性検証: gpu_opt vs sequential ---
echo "[3] 正確性検証 (BC 値 diff)..."
"${RUN_BENCH}" sequential "${GRAPH}" --dump-bc > "${TMP}/bc_seq.txt" 2>/dev/null
"${RUN_BENCH}" gpu_opt    "${GRAPH}" --dump-bc > "${TMP}/bc_opt.txt" 2>/dev/null
if diff -q "${TMP}/bc_seq.txt" "${TMP}/bc_opt.txt" >/dev/null; then
    ok "gpu_opt == sequential (BC 一致)"
else
    ng "gpu_opt != sequential (BC 不一致) — diff 先頭:"
    diff "${TMP}/bc_seq.txt" "${TMP}/bc_opt.txt" | head -6
fi
hr

# --- 4. アブレーション: baseline / full の動作と BC 一致 ---
echo "[4] run_ablation baseline/full..."
"${RUN_ABL}" "${GRAPH}" baseline --dump-bc > "${TMP}/bc_abl0.txt" 2>/dev/null && ok "ablation baseline 実行" || ng "ablation baseline 失敗"
"${RUN_ABL}" "${GRAPH}" full     --dump-bc > "${TMP}/bc_abl1.txt" 2>/dev/null && ok "ablation full 実行"     || ng "ablation full 失敗"
if diff -q "${TMP}/bc_abl0.txt" "${TMP}/bc_abl1.txt" >/dev/null; then
    ok "ablation baseline == full (全構成で BC 一致)"
else
    ng "ablation baseline != full (BC 不一致)"
fi
if diff -q "${TMP}/bc_seq.txt" "${TMP}/bc_abl1.txt" >/dev/null; then
    ok "ablation full == sequential (BC 一致)"
else
    ng "ablation full != sequential (BC 不一致)"
fi
hr

# --- 5. PathMerge バッチ掃引 (短縮) ---
echo "[5] run_pathmerge_sweep (短縮掃引 16,64,128)..."
if "${RUN_PM}" "${GRAPH}" 16,64,128 > "${TMP}/pm.tsv" 2>"${TMP}/pm.err"; then
    n="$(grep -c 'PathMerge_b' "${TMP}/pm.tsv" || true)"
    [ "${n}" -ge 1 ] && ok "pathmerge sweep: ${n} 構成計測" || ng "pathmerge sweep: 出力なし"
    grep 'BEST batch_size' "${TMP}/pm.err" || true
    # PathMerge vs sequential 正確性
    "${RUN_PM}" "${GRAPH}" 64 --dump-bc > "${TMP}/bc_pm.txt" 2>/dev/null
    if diff -q "${TMP}/bc_seq.txt" "${TMP}/bc_pm.txt" >/dev/null; then
        ok "pathmerge(b64) == sequential (BC 一致)"
    else
        ng "pathmerge(b64) != sequential (BC 不一致)"
    fi
else
    ng "run_pathmerge_sweep 失敗"
fi
hr

echo "=== Result: PASS=${pass}, FAIL=${fail} ==="
[ "${fail}" -eq 0 ] && echo "全チェック合格 ✅" || { echo "失敗あり ❌"; exit 1; }
