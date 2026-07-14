# ベンチマーク結果: 実行時間 (秒)

| 手法 | roadNet-TX |
|:-----|------:|
| GPU_Opt | 983.7 |
| GPU_Opt_Pure | 986.4 |
| GPU_Opt_Pure_Chunked | 988.6 |

# ベンチマーク結果: GTEPS (高いほど高速)

| 手法 | roadNet-TX |
|:-----|------:|
| GPU_Opt | 2.6959 |
| GPU_Opt_Pure | 2.6884 |
| GPU_Opt_Pure_Chunked | 2.6824 |

# スピードアップ (vs PathMerge_BC)

(ベースライン PathMerge_BC の結果がありません)
# スケーリング分析 (GTEPS vs グラフサイズ)

| グラフ | Nodes | Edges | n×m | GPU_Opt | GPU_Opt_Pure | GPU_Opt_Pure_Chunked |
|:------|------:|------:|------:|------:|------:|------:|
| roadNet-TX | 1,379,917 | 1,921,660 | 2.65e+12 | 2.6959 | 2.6884 | 2.6824 |
# 正確性検証 (max BC 値)

## roadNet-TX

| 手法 | Max BC Index | Max BC Value | 差分 (%) |
|:-----|:-----------|-------------:|---------:|
| GPU_Opt | 400570 | 164495142042.45 | 0 (基準) |
| GPU_Opt_Pure | 400570 | 164495142042.45 | 0 (基準) |
| GPU_Opt_Pure_Chunked | 400570 | 164495142042.45 | 0 (基準) |


# グラフ別詳細結果

## roadNet-TX (roadNet-TX)

| 手法 | 実行時間 (秒) | GTEPS | vs PathMerge_BC |
|:-----|----------:|------:|----------------:|
| GPU_Opt | 983.7 | 2.6959 | — |
| GPU_Opt_Pure | 986.4 | 2.6884 | — |
| GPU_Opt_Pure_Chunked | 988.6 | 2.6824 | — |

