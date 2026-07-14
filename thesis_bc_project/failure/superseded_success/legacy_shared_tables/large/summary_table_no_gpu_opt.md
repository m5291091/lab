# ベンチマーク結果: 実行時間 (秒)

| 手法 | roadNet-PA | roadNet-TX | roadNet-CA |
|:-----|------:|------:|------:|
| GPU_Opt_Pure | 1062.5 | 1636.1 | 3495.1 |
| PathMerge_BC | 923.3 | 1493.5 | 3494.3 |

# ベンチマーク結果: GTEPS (高いほど高速)

| 手法 | roadNet-PA | roadNet-TX | roadNet-CA |
|:-----|------:|------:|------:|
| GPU_Opt_Pure | 1.5790 | 1.6208 | 1.5556 |
| PathMerge_BC | 1.8173 | 1.7759 | 1.5560 |

# スピードアップ (vs PathMerge_BC)

| 手法 | roadNet-PA | roadNet-TX | roadNet-CA |
|:-----|------:|------:|------:|
| GPU_Opt_Pure | 0.87× | 0.91× | 1.00× |
| PathMerge_BC | 1.00× | 1.00× | 1.00× |
# スケーリング分析 (GTEPS vs グラフサイズ)

| グラフ | Nodes | Edges | n×m | GPU_Opt_Pure | PathMerge_BC |
|:------|------:|------:|------:|------:|------:|
| roadNet-PA | 1,088,092 | 1,541,898 | 1.68e+12 | 1.5790 | 1.8173 |
| roadNet-TX | 1,379,917 | 1,921,660 | 2.65e+12 | 1.6208 | 1.7759 |
| roadNet-CA | 1,965,206 | 2,766,607 | 5.44e+12 | 1.5556 | 1.5560 |
# フェーズ別タイミング (BFS / Backward)

| 手法 | roadNet-PA (BFS) | roadNet-PA (Bwd) | roadNet-TX (BFS) | roadNet-TX (Bwd) | roadNet-CA (BFS) | roadNet-CA (Bwd) |
|:-----|------:|------:|------:|------:|------:|------:|
| GPU_Opt_Pure | — | — | — | — | — | — |
| PathMerge_BC | 606.1096 | 305.9308 | 974.9525 | 497.3338 | 2522.7253 | 955.6193 |
# 正確性検証 (max BC 値)

## roadNet-PA

| 手法 | Max BC Index | Max BC Value | 差分 (%) |
|:-----|:-----------|-------------:|---------:|
| GPU_Opt_Pure | 557532 | 151395302679.08 | 0 (基準) |
| PathMerge_BC | 557532 | 151395302679.08 | 0 (基準) |

## roadNet-TX

| 手法 | Max BC Index | Max BC Value | 差分 (%) |
|:-----|:-----------|-------------:|---------:|
| GPU_Opt_Pure | 400570 | 164495142042.45 | 0 (基準) |
| PathMerge_BC | 400570 | 164495142042.45 | 0 (基準) |

## roadNet-CA

| 手法 | Max BC Index | Max BC Value | 差分 (%) |
|:-----|:-----------|-------------:|---------:|
| GPU_Opt_Pure | 1584888 | 686380725021.27 | 0 (基準) |
| PathMerge_BC | 1584888 | 686380725021.27 | 0 (基準) |


# グラフ別詳細結果

## roadNet-PA (roadNet-PA)

| 手法 | 実行時間 (秒) | GTEPS | vs PathMerge_BC |
|:-----|----------:|------:|----------------:|
| GPU_Opt_Pure | 1062.5 | 1.5790 | 0.87× |
| PathMerge_BC | 923.3 | 1.8173 | 1.00× |

## roadNet-TX (roadNet-TX)

| 手法 | 実行時間 (秒) | GTEPS | vs PathMerge_BC |
|:-----|----------:|------:|----------------:|
| GPU_Opt_Pure | 1636.1 | 1.6208 | 0.91× |
| PathMerge_BC | 1493.5 | 1.7759 | 1.00× |

## roadNet-CA (roadNet-CA)

| 手法 | 実行時間 (秒) | GTEPS | vs PathMerge_BC |
|:-----|----------:|------:|----------------:|
| GPU_Opt_Pure | 3495.1 | 1.5556 | 1.00× |
| PathMerge_BC | 3494.3 | 1.5560 | 1.00× |

