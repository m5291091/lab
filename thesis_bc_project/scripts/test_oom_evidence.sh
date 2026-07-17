#!/bin/bash

# scripts/oom_evidence.sh の CPU fixture (Gate W7.3B2.2)。GPU も runner も要らない。
#
# 回帰の的は job 2404249 の誤判定行そのもの (NOT_OOM の先頭 fixture) である。
# 実行: bash scripts/test_oom_evidence.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/oom_evidence.sh"

PASS=0
FAIL=0

report() {
    local outcome="$1" detail="$2"
    if [ "${outcome}" = ok ]; then
        PASS=$((PASS + 1))
        printf '  ok   %s\n' "${detail}"
    else
        FAIL=$((FAIL + 1))
        printf '  FAIL %s\n' "${detail}"
    fi
}

expect_line_class() {
    local line="$1" expected="$2" actual
    actual="$(bcoom_classify_line "${line}")"
    if [ "${actual}" = "${expected}" ]; then
        report ok "class=${actual} <- ${line}"
    else
        report no "expected=${expected} actual=${actual} <- ${line}"
    fi
}

expect_status() {
    local rc="$1" class="$2" vector_state="$3" expected="$4" actual
    actual="$(bcoom_decide_status "${rc}" "${class}" "${vector_state}")"
    if [ "${actual}" = "${expected}" ]; then
        report ok "exit=${rc} evidence=${class} vector=${vector_state} -> ${actual}"
    else
        report no "exit=${rc} evidence=${class} vector=${vector_state} expected=${expected} actual=${actual}"
    fi
}

echo "== strong OOM evidence: real allocation failures =="
expect_line_class 'CUDA error: out of memory' cuda_oom
expect_line_class 'CUDA error at x: out of memory' cuda_oom
expect_line_class 'CUDA error at src/proposed/host_pure.cu:142: out of memory' cuda_oom
expect_line_class 'CUDA Error at src/baseline/galliot.cu:38 - out of memory' cuda_oom
expect_line_class 'cudaErrorMemoryAllocation' cuda_oom
expect_line_class 'CUDA_ERROR_OUT_OF_MEMORY' cuda_oom
expect_line_class 'RMM failure at: device_memory_resource.hpp: cudaErrorMemoryAllocation out of memory' cuda_oom
expect_line_class 'std::bad_alloc' host_alloc_failure
expect_line_class "terminate called after throwing an instance of 'std::bad_alloc'" host_alloc_failure
expect_line_class "terminate called after throwing an instance of 'rmm::bad_alloc'" host_alloc_failure
expect_line_class 'Out of memory: Killed process' kernel_oom_kill
expect_line_class 'Out of memory: Killed process 12345 (run_benchmark)' kernel_oom_kill
expect_line_class 'Killed process 12345 (run_benchmark) total-vm:99G, out of memory' kernel_oom_kill
expect_line_class 'oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null)' kernel_oom_kill
expect_line_class 'oom-killed process 12345' kernel_oom_kill

echo "== not OOM: the word is only mentioned =="
# job 2404249 が実際に停止した行 (回帰の的)。
expect_line_class '  > [Warn] BC_BATCH_OVERRIDE=1024 exceeds safe limit 512; may cause cudaMalloc OOM' none
expect_line_class 'may cause cudaMalloc OOM' none
expect_line_class 'OOM budget=100GB' none
expect_line_class 'OOM threshold' none
expect_line_class 'avoid OOM by chunking' none
expect_line_class 'OOM guard enabled' none
expect_line_class 'OOM-safe' none
expect_line_class 'OOM_OR_FAIL' none
expect_line_class 'expected OOM' none
expect_line_class '  > [Mem] GPU: total=102.0 GB, free_before=101.4 GB' none
expect_line_class '  > GTEPS = 16.1750 (nodes=325557, edges=3216152)' none

echo "== file scan: the real job 2404249 stderr must yield no evidence =="
FIXTURE_DIR="$(mktemp -d)"
trap 'rm -rf "${FIXTURE_DIR}"' EXIT

cat > "${FIXTURE_DIR}/false_positive.stderr.log" <<'EOF'
Running: GPU_Opt_Pure on 325557_3216152_corrected_v1...
  > [Warn] BC_BATCH_OVERRIDE=1024 exceeds safe limit 512; may cause cudaMalloc OOM
  > [Mem] GPU: total=102.0 GB, free_before=101.4 GB
  > [Mem] topology(GPU)=0.03 GB, dynamic(GPU)=21.34 GB, batch_per_stream=1024
  > [GPU Phase] BFS wall=28.6410 s (cum=56.6476 s), Backward wall=32.4988 s (cum=62.0710 s)
  > index : 272816, Maximum Betweenness Centrality ==> 39343117052.54
  > Elapse time [sec.] = 64.732001
  > GTEPS = 16.1750 (nodes=325557, edges=3216152)
EOF

if bcoom_scan "${FIXTURE_DIR}/false_positive.stderr.log"; then
    report no "job 2404249 stderr was treated as OOM evidence (class=${BCOOM_EVIDENCE_CLASS}, line ${BCOOM_LINE_NUMBER})"
else
    report ok "job 2404249 stderr yields OOMEvidenceClass=${BCOOM_EVIDENCE_CLASS}"
fi

cat > "${FIXTURE_DIR}/real_oom.stderr.log" <<'EOF'
Running: GPU_Opt_Pure on 325557_3216152_corrected_v1...
  > [Warn] BC_BATCH_OVERRIDE=8192 exceeds safe limit 512; may cause cudaMalloc OOM
  > [Mem] GPU: total=102.0 GB, free_before=101.4 GB
CUDA error at src/proposed/host_pure.cu:142: out of memory
EOF

if bcoom_scan "${FIXTURE_DIR}/real_oom.stderr.log"; then
    if [ "${BCOOM_EVIDENCE_CLASS}" = cuda_oom ] && [ "${BCOOM_LINE_NUMBER}" = 4 ]; then
        report ok "real OOM located at line ${BCOOM_LINE_NUMBER}, class=${BCOOM_EVIDENCE_CLASS} (warning line skipped)"
    else
        report no "real OOM misreported: class=${BCOOM_EVIDENCE_CLASS} line=${BCOOM_LINE_NUMBER}"
    fi
else
    report no "real OOM line was not detected"
fi

echo "== status decision =="
expect_status 0 none complete SUCCESS
expect_status 0 none not_applicable SUCCESS
expect_status 0 none invalid VECTOR_INVALID
expect_status 0 none missing VECTOR_INVALID
expect_status 0 cuda_oom complete RUNNER_SWALLOWED_OOM
expect_status 1 cuda_oom not_applicable OOM_CONFIRMED
expect_status 1 none not_applicable RUNTIME_FAILED
expect_status 124 none not_applicable RUNTIME_FAILED
expect_status 137 none not_applicable RUNTIME_FAILED
expect_status 137 kernel_oom_kill not_applicable OOM_CONFIRMED

echo "== end-to-end: exit0 + complete vector + warning line must be SUCCESS =="
bcoom_scan "${FIXTURE_DIR}/false_positive.stderr.log"
expect_status 0 "${BCOOM_EVIDENCE_CLASS}" complete SUCCESS
expect_status 0 "${BCOOM_EVIDENCE_CLASS}" invalid VECTOR_INVALID
bcoom_scan "${FIXTURE_DIR}/real_oom.stderr.log"
expect_status 1 "${BCOOM_EVIDENCE_CLASS}" not_applicable OOM_CONFIRMED

echo
printf 'passed=%d failed=%d\n' "${PASS}" "${FAIL}"
[ "${FAIL}" -eq 0 ] || exit 1
echo "ALL FIXTURES PASSED"
exit 0
