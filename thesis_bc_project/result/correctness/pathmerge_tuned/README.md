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
削除してよい (サマリとハッシュを先に保存)。同じ vector 本体を再取得できた場合に限り、記録済み
ハッシュとの同一性確認が可能である。

## Archive-time supplemental audit (2026-07-17, Gate W5.1 / W5.2)

比較に用いた 4 本の BC ベクトルの **original runtime path**（実験時に runner が出力した
**historical build output path**）は、email が `build_miyabi/t1_correctness/{bc_b64.txt,bc_b2048.txt}`、
roadNet-CA が `build_miyabi/t1_ca_correctness/{bc_b32.txt,bc_b64.txt}` である。これらは実験時の
出力先を指す記録であり、現在のアーカイブ内の保存場所ではない。4 本とも **currently unavailable**
であり、その path、Git 追跡下、`raw_data/` のいずれにも vector 本体は存在しない。台帳登録は
`result/EXTERNAL_ARTIFACTS.tsv`（`RetentionStatus=not_retained`, `Availability=currently_unavailable`）
に置く。

保存されているのは **比較 summary のみ** である。したがって archive-time の読み取り専用 parse は
実施できず、NaN、+Inf、-Inf、duplicate index は `not_recorded` のままである。既存サマリの
vector length、missing index、mismatch、error metrics、PASS 判定は当時記録された混合許容比較の
範囲に限定し、この追加監査で再検証した値として扱わない（**archive-time に vector を再解析していない**）。

各 vector の OriginalPath と SizeBytes は Git 追跡下の PBS stdout log
(`raw_data/tuning/pathmerge/{email-EuAll,roadNet-CA}/pathmerge_bc/job_{2360074,2362965}_20260711/pbs_stdout.log`
の `ls -l` 記録)、SHA256 は本ディレクトリの比較 summary の記録に基づく。Gate W5.2 では vector の
復元・再生成・推定を行っていない。

roadNet-PA/TX は tuned/default とも b64 であり、別の full-vector comparison artifact がないため、
このディレクトリの `full_vector_same_implementation` 対象には含めない。両グラフの正式水準は
既存の Max BC 記録に限定した `max_bc_only` である。

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
