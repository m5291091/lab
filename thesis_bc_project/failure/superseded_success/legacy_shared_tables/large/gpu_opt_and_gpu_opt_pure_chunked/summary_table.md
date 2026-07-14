# ベンチマーク結果: 実行時間 (秒)

| 手法 | roadNet-PA | roadNet-TX | roadNet-CA |
|:-----|------:|------:|------:|
| GPU_Opt | 1063.6 | 1637.9 | 3497.9 |
| GPU_Opt_Pure_Chunked | 1061.5 | 1635.1 | 3492.7 |

# ベンチマーク結果: GTEPS (高いほど高速)

| 手法 | roadNet-PA | roadNet-TX | roadNet-CA |
|:-----|------:|------:|------:|
| GPU_Opt | 1.5774 | 1.6190 | 1.5543 |
| GPU_Opt_Pure_Chunked | 1.5806 | 1.6218 | 1.5566 |

# スピードアップ (vs PathMerge_BC)

(ベースライン PathMerge_BC の結果がありません)
# スケーリング分析 (GTEPS vs グラフサイズ)

| グラフ | Nodes | Edges | n×m | GPU_Opt | GPU_Opt_Pure_Chunked |
|:------|------:|------:|------:|------:|------:|
| roadNet-PA | 1,088,092 | 1,541,898 | 1.68e+12 | 1.5774 | 1.5806 |
| roadNet-TX | 1,379,917 | 1,921,660 | 2.65e+12 | 1.6190 | 1.6218 |
| roadNet-CA | 1,965,206 | 2,766,607 | 5.44e+12 | 1.5543 | 1.5566 |
# 正確性検証 (max BC 値)

## roadNet-PA

| 手法 | Max BC Index | Max BC Value | 差分 (%) |
|:-----|:-----------|-------------:|---------:|
| GPU_Opt | 557532 | 151395302679.08 | 0 (基準) |
| GPU_Opt_Pure_Chunked | 557532 | 151395302679.08 | 0 (基準) |

## roadNet-TX

| 手法 | Max BC Index | Max BC Value | 差分 (%) |
|:-----|:-----------|-------------:|---------:|
| GPU_Opt | 400570 | 164495142042.45 | 0 (基準) |
| GPU_Opt_Pure_Chunked | 400570 | 164495142042.45 | 0 (基準) |

## roadNet-CA

| 手法 | Max BC Index | Max BC Value | 差分 (%) |
|:-----|:-----------|-------------:|---------:|
| GPU_Opt | 1584888 | 686380725021.27 | 0 (基準) |
| GPU_Opt_Pure_Chunked | 1584888 | 686380725021.27 | 0 (基準) |


# グラフ別詳細結果

## roadNet-PA (roadNet-PA)

| 手法 | 実行時間 (秒) | GTEPS | vs PathMerge_BC |
|:-----|----------:|------:|----------------:|
| GPU_Opt | 1063.6 | 1.5774 | — |
| GPU_Opt_Pure_Chunked | 1061.5 | 1.5806 | — |

## roadNet-TX (roadNet-TX)

| 手法 | 実行時間 (秒) | GTEPS | vs PathMerge_BC |
|:-----|----------:|------:|----------------:|
| GPU_Opt | 1637.9 | 1.6190 | — |
| GPU_Opt_Pure_Chunked | 1635.1 | 1.6218 | — |

## roadNet-CA (roadNet-CA)

| 手法 | 実行時間 (秒) | GTEPS | vs PathMerge_BC |
|:-----|----------:|------:|----------------:|
| GPU_Opt | 3497.9 | 1.5543 | — |
| GPU_Opt_Pure_Chunked | 3492.7 | 1.5566 | — |

