# SOURCE — correctness/pathmerge_tuned（PathMerge tuned 全ベクトル一致検証）

判定基準・再現手順の詳細は同 dir の `README.md` を参照。

- **目的**: PathMerge の tuned バッチが既定 b64 と同一の BC を出すことを `--dump-bc` で全ベクトル検証
- **支える論文主張**: A（PathMerge tuned 採用の正確性根拠）
- **グラフ**: email-EuAll, roadNet-CA
- **実装**: PathMerge_BC（同一実装, batch 差のみ）
- **要求/実効バッチ**: email b64 vs b2048 / roadNet-CA b32 vs b64
- **SUB_BATCH / num_subs**: `not_applicable`（PathMerge）
- **試行数**: 1（`--dump-bc` 全ベクトル比較） / **warmup**: `not_applicable` / **集計方法**: `not_applicable`（全要素比較）
- **SourceSnapshotID**: `phase_def_block_20260710`
- **PBS job ID**: 2360074（email b64 vs b2048）, 2362965（roadNet-CA b32 vs b64）
- **入力**: `data/snap/{email-EuAll,roadNet-CA}`（巨大 BC ベクトルは非追跡, SHA256 記録）
- **出力**: `email-EuAll_b64_vs_b2048.md`, `roadNet-CA_b32_vs_b64.md`, `README.md`
- **正確性**: `full_vector_same_implementation`（email: len 265009, max_rel_err 4.9e-14; CA: max_rel_err 3.9e-13, Max BC 同一 index 1584888, 混合許容不一致 0）
- **再現コマンド**: `scripts/run_pathmerge_correctness.sh` + `scripts/compare_bc_vectors.py`
- **制約**: 同一実装の batch 間比較（独立参照ではない）。巨大 BC ベクトル本体は保持せず、SHA256 のみを記録。
- **archive-time vector 状態 (Gate W5.1 / W5.2, 2026-07-17)**: email/CA の 4 vector 本体は `currently_unavailable`。original runtime path（historical build output path = `build_miyabi/t1_correctness/`, `build_miyabi/t1_ca_correctness/`）、Git、`raw_data/` のいずれにも存在しない。保存されているのは比較 summary のみで、archive-time に vector を再解析していない。追加 parse は実施不能のため NaN/+Inf/-Inf/duplicate index は `not_recorded`。台帳登録は `result/EXTERNAL_ARTIFACTS.tsv`（`RetentionStatus=not_retained`, `Availability=currently_unavailable`, PBSJobID 2360074/2362965）。
- **PA/TX の範囲**: tuned/default とも b64 で別 full-vector comparison artifact がないため対象外。正式水準は `max_bc_only`。
