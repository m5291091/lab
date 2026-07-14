# BC ベクトル数値比較: email-EuAll PathMerge b64 (既定; 要求64/実効64, num_batches=4141) vs email-EuAll PathMerge b2048 (tuned; 要求2048/実効2048, num_batches=130)

- 入力A: `build_miyabi/t1_correctness/bc_b64.txt`  (header: impl=PathMerge_b64 graph=email-EuAll nodes=265009)
  - SHA256: `ad9c4c3f0c9fa26f8de9b82e7d235b0b824d076959cb4c81419981a0d893fa5a`
- 入力B: `build_miyabi/t1_correctness/bc_b2048.txt`  (header: impl=PathMerge_b2048 graph=email-EuAll nodes=265009)
  - SHA256: `76a578f72f391aced3e65d3f5918510dcdef06663d85c6feb83349bd1d67b380`

- checkpoint_sha: phase_def_block_20260710
- PBS_job: 2360074
- graph: email-EuAll (n=265009)
- clamp: なし (両バッチとも実効=要求)

| 項目 | 値 |
|:--|:--|
| ベクトル長 A | 265009 |
| ベクトル長 B | 265009 |
| 共通 index 数 | 265009 |
| 欠損 index 数 (A のみ) | 0 |
| 欠損 index 数 (B のみ) | 0 |
| 最大絶対誤差 | 1.943111e-05 (index 47159) |
| 最大相対誤差 | 4.913736e-14 (index 237) |
| Max BC A | index 10, value 2384894520.796642 |
| Max BC B | index 10, value 2384894520.796650 |
| 許容値 | rel_tol=1e-06, abs_tol=0.001 |
| 絶対許容のみ (abs_diff ≤ abs_tol) | OK |
| 混合許容 abs_diff ≤ abs_tol+rel_tol·max(\|a\|,\|b\|) 不一致要素数 | 0 |
| **総合判定** | **PASS** |

