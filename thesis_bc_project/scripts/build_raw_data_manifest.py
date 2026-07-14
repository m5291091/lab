#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[DEPRECATED / Gate J1] raw_data/MANIFEST.tsv + SHA256SUMS の生成器。

Gate J1 で raw_data/ が 133 ファイル（J0 の 23 + 移行 109 + small pbs_stdout）へ拡張され、
`result/EXTERNAL_ARTIFACTS.tsv` は「真に Git 外」のみへスリム化されたため、旧来の
「EXTERNAL_ARTIFACTS の期待 SHA から 23 件を生成する」方式は使用しない。

正式な自己完結生成器は `scripts/build_raw_data_index.py`（raw_data/ 自身から
MANIFEST.tsv / SHA256SUMS / RAW_DATA_INDEX.tsv を再生成; build_miyabi 非依存）。
互換のため本スクリプトはそれへ委譲する。
"""
import runpy, os
_here = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(_here, "build_raw_data_index.py"), run_name="__main__")
