# ベンチマーク結果: 実行時間 (秒)

| 手法 | roadNet-PA |
|:-----|------:|
| GPU_Opt | 700.5 |
| GPU_Opt_Pure | 697.9 |
| GPU_Opt_Pure_Chunked | 693.7 |

# ベンチマーク結果: GTEPS (高いほど高速)

| 手法 | roadNet-PA |
|:-----|------:|
| GPU_Opt | 2.3951 |
| GPU_Opt_Pure | 2.4040 |
| GPU_Opt_Pure_Chunked | 2.4187 |

# スピードアップ (vs PathMerge_BC)

(ベースライン PathMerge_BC の結果がありません)
# スケーリング分析 (GTEPS vs グラフサイズ)

| グラフ | Nodes | Edges | n×m | GPU_Opt | GPU_Opt_Pure | GPU_Opt_Pure_Chunked |
|:------|------:|------:|------:|------:|------:|------:|
| roadNet-PA | 1,088,092 | 1,541,898 | 1.68e+12 | 2.3951 | 2.4040 | 2.4187 |
# 正確性検証 (max BC 値)

## roadNet-PA

| 手法 | Max BC Index | Max BC Value | 差分 (%) |
|:-----|:-----------|-------------:|---------:|
| GPU_Opt | 557532 | 151395302679.08 | 0 (基準) |
| GPU_Opt_Pure | 557532 | 151395302679.08 | 0 (基準) |
| GPU_Opt_Pure_Chunked | 557532 | 151395302679.08 | 0 (基準) |


# グラフ別詳細結果

## roadNet-PA (roadNet-PA)

| 手法 | 実行時間 (秒) | GTEPS | vs PathMerge_BC |
|:-----|----------:|------:|----------------:|
| GPU_Opt | 700.5 | 2.3951 | — |
| GPU_Opt_Pure | 697.9 | 2.4040 | — |
| GPU_Opt_Pure_Chunked | 693.7 | 2.4187 | — |

