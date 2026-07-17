#!/bin/bash

# CMake ビルドディレクトリ衝突ガード (Gate W7.3B1.1)。
#
# job 2403658 は root project と cugraph_bc_mini が同一の build_miyabi/ を
# CMake binary directory として使用したため Stage 2 の configure で停止した。
# 直接原因は BUILD_DIR が env 経由で child script へ継承されたことである
# (`BUILD_DIR=x bash parent.sh` は parent 内の再代入後も export 属性を保つ)。
#
# 本ファイルは source して使う。各関数は診断を stderr へ出し 0/1 を返すのみで、
# exit は呼び出し側の規約 (abort=2 等) に委ねる。

# 存在しないパスも正規化する (readlink -f は末尾以外の実在を要求するため -m を使う)。
bcguard_canon() {
    readlink -m -- "$1"
}

# CMakeCache.txt の CMAKE_HOME_DIRECTORY を出力する。cache 不在なら空文字。
bcguard_cache_home() {
    local cache="$1/CMakeCache.txt"
    [ -f "${cache}" ] || return 0
    sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${cache}" | head -n1
}

bcguard_report_context() {
    local root_src="$1" root_build="$2" mini_src="$3" mini_build="$4"
    printf '  root   source=%s\n' "${root_src}" >&2
    printf '  root   binary=%s\n' "${root_build}" >&2
    printf '  root   cache CMAKE_HOME_DIRECTORY=%s\n' "$(bcguard_cache_home "${root_build}")" >&2
    printf '  mini   source=%s\n' "${mini_src}" >&2
    printf '  mini   binary=%s\n' "${mini_build}" >&2
    printf '  mini   cache CMAKE_HOME_DIRECTORY=%s\n' "$(bcguard_cache_home "${mini_build}")" >&2
    printf '  checkpoint_sha=%s\n' "${EXPECTED_SHA:-$(git rev-parse HEAD 2>/dev/null || echo not_recorded)}" >&2
    printf '  pbs_job_id=%s\n' "${PBS_JOBID:-not_pbs}" >&2
}

# root/mini の binary directory が分離され、既存 cache が別 source tree を
# 指していないことを確認する。違反時は診断を出して 1 を返す。
bcguard_assert_separate() {
    local root_src root_build mini_src mini_build
    root_src="$(bcguard_canon "$1")"
    root_build="$(bcguard_canon "$2")"
    mini_src="$(bcguard_canon "$3")"
    mini_build="$(bcguard_canon "$4")"

    local root_home mini_home
    root_home="$(bcguard_cache_home "${root_build}")"
    mini_home="$(bcguard_cache_home "${mini_build}")"
    [ -n "${root_home}" ] && root_home="$(bcguard_canon "${root_home}")"
    [ -n "${mini_home}" ] && mini_home="$(bcguard_canon "${mini_home}")"

    if [ "${root_build}" = "${mini_build}" ]; then
        echo "BUILD DIR COLLISION: root and cugraph_bc_mini share one CMake binary directory" >&2
        bcguard_report_context "${root_src}" "${root_build}" "${mini_src}" "${mini_build}"
        return 1
    fi

    if [ -n "${root_home}" ] && [ "${root_home}" != "${root_src}" ]; then
        echo "FOREIGN CMAKE CACHE: root binary directory was generated from another source tree" >&2
        bcguard_report_context "${root_src}" "${root_build}" "${mini_src}" "${mini_build}"
        return 1
    fi

    if [ -n "${mini_home}" ] && [ "${mini_home}" != "${mini_src}" ]; then
        echo "FOREIGN CMAKE CACHE: cugraph_bc_mini binary directory was generated from another source tree" >&2
        bcguard_report_context "${root_src}" "${root_build}" "${mini_src}" "${mini_build}"
        return 1
    fi

    return 0
}

# ビルド成功後に来歴スタンプを書く。これが無い binary は正式実行に使わない。
bcguard_write_provenance() {
    local build_dir="$1" checkpoint="$2"
    {
        printf 'checkpoint_sha=%s\n' "${checkpoint}"
        printf 'pbs_job_id=%s\n' "${PBS_JOBID:-not_pbs}"
        printf 'built_at=%s\n' "$(date -Is)"
    } > "${build_dir}/.bc_build_provenance"
}

# binary が今回の checkpoint から生成されたことを確認する。
# スタンプ不在 (= 旧 binary への fallback) も違反として扱う。
bcguard_assert_provenance() {
    local build_dir="$1" checkpoint="$2"
    local stamp="${build_dir}/.bc_build_provenance"
    if [ ! -f "${stamp}" ]; then
        echo "UNVERIFIED BINARY: no build provenance stamp in ${build_dir}" >&2
        printf '  expected stamp=%s\n' "${stamp}" >&2
        printf '  checkpoint_sha=%s\n  pbs_job_id=%s\n' "${checkpoint}" "${PBS_JOBID:-not_pbs}" >&2
        return 1
    fi
    local stamped
    stamped="$(sed -n 's/^checkpoint_sha=//p' "${stamp}" | head -n1)"
    if [ "${stamped}" != "${checkpoint}" ]; then
        echo "UNVERIFIED BINARY: build provenance checkpoint mismatch" >&2
        printf '  stamp=%s\n  stamped_checkpoint=%s\n  expected_checkpoint=%s\n  pbs_job_id=%s\n' \
            "${stamp}" "${stamped}" "${checkpoint}" "${PBS_JOBID:-not_pbs}" >&2
        return 1
    fi
    return 0
}
