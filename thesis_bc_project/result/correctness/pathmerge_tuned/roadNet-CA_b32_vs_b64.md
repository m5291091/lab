# BC ベクトル数値比較: roadNet-CA PathMerge b32 (実測最適 tuned; 要求32/実効32, num_batches=61413) vs roadNet-CA PathMerge b64 (既定; 要求64/実効64, num_batches=30707)

- 入力A: `build_miyabi/t1_ca_correctness/bc_b32.txt`  (header: impl=PathMerge_b32 graph=roadNet-CA nodes=1965206)
  - SHA256: `27e9c5cf1eed361693cfae59529fc88cdb0cc1bc32b88e2035af40aba80cf01b`
- 入力B: `build_miyabi/t1_ca_correctness/bc_b64.txt`  (header: impl=PathMerge_b64 graph=roadNet-CA nodes=1965206)
  - SHA256: `afc97f4da0bc080a55cb267bcfc635bad129a43fc4bd338af9d4cdfe6500423e`

- checkpoint_sha: phase_def_block_20260710
- PBS_job: 2362965
- graph: roadNet-CA (n=1965206)
- clamp: なし (両バッチとも実効=要求)
- b32_time_s: 3092.33
- b64_time_s: 3492.03

| 項目 | 値 |
|:--|:--|
| ベクトル長 A | 1965206 |
| ベクトル長 B | 1965206 |
| 共通 index 数 | 1965206 |
| 欠損 index 数 (A のみ) | 0 |
| 欠損 index 数 (B のみ) | 0 |
| 最大絶対誤差 | 3.339767e-03 (index 1423587) |
| 最大相対誤差 | 3.878449e-13 (index 1423587) |
| Max BC A | index 1584888, value 686380725021.268311 |
| Max BC B | index 1584888, value 686380725021.267822 |
| 許容値 | rel_tol=1e-06, abs_tol=0.001 |
| 絶対許容のみ (abs_diff ≤ abs_tol) | WARN (超過; 巨大 magnitude で不適切な場合あり) |
| 混合許容 abs_diff ≤ abs_tol+rel_tol·max(\|a\|,\|b\|) 不一致要素数 | 0 |
| **総合判定** | **PASS (absolute-only warning)** |


## 解釈 (総合判定の補足)

- **総合判定は PASS (absolute-only warning)**: 構造 (ベクトル長・index) は完全一致で、混合許容
  `abs_diff ≤ abs_tol + rel_tol·max(|a|,|b|)` を満たさない要素は **0 件**。絶対許容 (1e-3) 単独では
  最大絶対誤差 3.34e-3 が超過し WARN となるが、これは BC 値の大きさが ~10^10 と巨大なため絶対許容が
  不適切なだけで、相対誤差 (最大 3.88e-13, 倍精度丸め水準) では一致している。単独の「絶対許容超過」を
  総合判定として扱わない。
- **原因 (断定しない)**: この差は、バッチ分割数の違い (b32=61413 バッチ / b64=30707 バッチ) に伴う
  **加算順序の違いによる浮動小数点丸め差と整合的**である。PathMerge は両バッチとも厳密 BC を計算する
  ため、アルゴリズム上の相違ではないと考えられる (断定はしない)。
- **参考値 (誤差推定式)**: 最大絶対誤差が生じた index 1423587 の BC 値 ~8.611e+09 に対し、丸め誤差の
  目安 `mag × eps × √(2·num_batches) ≈ 8.6e9 × 2.22e-16 × √(2·61413) ≈ 6.7e-4` は観測値 3.34e-3 と
  同オーダー。**これは厳密な上界ではなく参考のオーダー推定**である。
- 最終表では CA tuned = **b32** (実測最適)、vs tuned = **1.45×** を使用する。
- 入力ベクトルの SHA256 は上記に記録済み (巨大ベクトル本体は非追跡; ハッシュで後日の同一性確認が可能)。
