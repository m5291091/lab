#!/bin/bash
# submit_all.sh — PBS ジョブ一括投入スクリプト
#
# 全実験ジョブを依存関係付きで一度に投入する。
#   JOB1 (run_baseline.sh)  → 完了後
#   JOB2 (run_ablation.sh)  → 完了後
#   JOB3 (run_profile.sh)
#
# measure_bandwidth.sh は他のジョブに依存しないので並列投入する。
#
# 使用方法:
#   cd /work/gj17/j17000/m5291091/lab/research
#   bash scripts/submit_all.sh
#
# 実行後の確認:
#   qstat -u $USER      # 自分のジョブ一覧
#   qstat -f <job_id>   # 特定ジョブの詳細

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- ジョブスクリプトの存在確認 ----
for SCRIPT in run_baseline.sh run_ablation.sh run_profile.sh measure_bandwidth.sh; do
    if [ ! -f "${SCRIPT_DIR}/${SCRIPT}" ]; then
        echo "ERROR: ${SCRIPT_DIR}/${SCRIPT} が見つかりません" >&2
        exit 1
    fi
done

echo "========================================"
echo "  PBS ジョブ一括投入"
echo "  投入ディレクトリ: $(pwd)"
echo "========================================"

# ---- JOB1: ベースライン計測 (24h) ----
JOB1=$(qsub "${SCRIPT_DIR}/run_baseline.sh")
echo "JOB1 (run_baseline.sh)  → ${JOB1}"

# ---- JOB2: アブレーションスタディ (2h) — JOB1 完了後 ----
JOB2=$(qsub -W depend=afterok:"${JOB1}" "${SCRIPT_DIR}/run_ablation.sh")
echo "JOB2 (run_ablation.sh)  → ${JOB2}  [depends on: ${JOB1}]"

# ---- JOB3: Nsight プロファイリング (2h) — JOB2 完了後 ----
JOB3=$(qsub -W depend=afterok:"${JOB2}" "${SCRIPT_DIR}/run_profile.sh")
echo "JOB3 (run_profile.sh)   → ${JOB3}  [depends on: ${JOB2}]"

# ---- JOB4: 帯域計測 (30min) — 独立して並列実行 ----
JOB4=$(qsub "${SCRIPT_DIR}/measure_bandwidth.sh")
echo "JOB4 (measure_bandwidth.sh) → ${JOB4}  [independent]"

echo ""
echo "========================================"
echo "  投入完了"
echo "  実行順: ${JOB1} → ${JOB2} → ${JOB3}"
echo "  並列:   ${JOB4} (帯域計測)"
echo ""
echo "  モニタリング:"
echo "    qstat -u \$USER        # ジョブ一覧"
echo "    qstat -f ${JOB1}      # JOB1 詳細"
echo "    qdel ${JOB1} ${JOB2} ${JOB3} ${JOB4}  # 全キャンセル"
echo "========================================"
