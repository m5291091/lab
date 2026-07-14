# PathMerge tuned 正確性確認 (BC ベクトル数値比較)

PathMerge の tuned バッチが既定 b64 と同一の BC を出すことを `--dump-bc` で検証する。
巨大な BC ベクトル (email ~7.5 MB / roadNet-CA ~50 MB 級) は git に追加しない。
比較サマリ (ベクトル長・欠損 index 数・最大絶対/相対誤差・Max BC index/value・許容判定・
入力 SHA256・checkpoint SHA・PBS job ID・要求/実効バッチ) のみ保存する。

## 実施済み
- `email-EuAll_b64_vs_b2048.md`: email-EuAll の b64 (既定) vs b2048 (tuned)。job 2360074。
  → 長さ一致 (265009)、欠損 0、最大相対誤差 4.9e-14、混合許容不一致 0。**総合判定 PASS**。
- `roadNet-CA_b32_vs_b64.md`: roadNet-CA の b32 (実測最適 tuned) vs b64 (既定)。job 2362965。
  → 長さ一致 (1965206)、欠損 0、最大相対誤差 3.9e-13、混合許容不一致 0、Max BC 同一 (index 1584888)。
  絶対許容 1e-3 は最大絶対誤差 3.34e-3 で WARN だが、BC 値 ~10^10 で絶対許容が不適切なだけ。
  **総合判定 PASS (absolute-only warning)**。

## 判定基準 (混合許容)
総合判定は numpy allclose 相当の混合許容 `abs_diff ≤ abs_tol + rel_tol·max(|a|,|b|)` を全要素で満たすか
で行う (不一致要素数を記録)。巨大 magnitude の BC では絶対許容単独は不適切なため、絶対許容超過は
WARN として分離し、単独の失敗判定にはしない。最大絶対/相対誤差・該当 index・両 SHA256・許容値は
そのまま保存する。

## 一時ベクトルの扱い
巨大 dump は `build_miyabi/`(gitignore) に生成し、比較サマリに **入力 SHA256** を記録した後に
削除してよい (サマリとハッシュを先に保存)。ハッシュにより後日の同一性確認が可能。

## 再現コマンド (GPU ノード, checkpoint phase_def_block_20260710, SKIP_BUILD=1)
```bash
# email-EuAll (b64 vs b2048)
GRAPH=snap/email-EuAll BATCHES="64 2048" TRIALS=1 SKIP_BUILD=1 \
  RESULT_DIR=build_miyabi/t1_correctness bash scripts/run_pathmerge_correctness.sh
python3 scripts/compare_bc_vectors.py \
  build_miyabi/t1_correctness/bc_b64.txt build_miyabi/t1_correctness/bc_b2048.txt \
  --extra "checkpoint_sha=phase_def_block_20260710..." "PBS_job=2360074" \
  --out result/correctness/pathmerge_tuned/email-EuAll_b64_vs_b2048.md

# roadNet-CA (b32 実測最適 vs b64 既定)
GRAPH=snap/roadNet-CA BATCHES="32 64" TRIALS=1 SKIP_BUILD=1 \
  RESULT_DIR=build_miyabi/t1_ca_correctness bash scripts/run_pathmerge_correctness.sh
python3 scripts/compare_bc_vectors.py \
  build_miyabi/t1_ca_correctness/bc_b32.txt build_miyabi/t1_ca_correctness/bc_b64.txt \
  --extra "checkpoint_sha=phase_def_block_20260710..." "PBS_job=2362965" \
  --out result/correctness/pathmerge_tuned/roadNet-CA_b32_vs_b64.md
```
