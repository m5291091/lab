# ベンチマーク結果: 実行時間 (秒)

| 手法 | roadNet-CA |
|:-----|------:|
| GPU_Opt | 2127.4 |
| GPU_Opt_Pure | 2119.3 |
| GPU_Opt_Pure_Chunked | 2123.2 |

# ベンチマーク結果: GTEPS (高いほど高速)

| 手法 | roadNet-CA |
|:-----|------:|
| GPU_Opt | 2.5557 |
| GPU_Opt_Pure | 2.5655 |
| GPU_Opt_Pure_Chunked | 2.5608 |

# スピードアップ (vs PathMerge_BC)

(ベースライン PathMerge_BC の結果がありません)
# スケーリング分析 (GTEPS vs グラフサイズ)

| グラフ | Nodes | Edges | n×m | GPU_Opt | GPU_Opt_Pure | GPU_Opt_Pure_Chunked |
|:------|------:|------:|------:|------:|------:|------:|
| roadNet-CA | 1,965,206 | 2,766,607 | 5.44e+12 | 2.5557 | 2.5655 | 2.5608 |
# 正確性検証 (max BC 値)

## roadNet-CA

| 手法 | Max BC Index | Max BC Value | 差分 (%) |
|:-----|:-----------|-------------:|---------:|
| GPU_Opt | 1584888 | 686380725021.27 | 0 (基準) |
| GPU_Opt_Pure | 1584888 | 686380725021.27 | 0 (基準) |
| GPU_Opt_Pure_Chunked | 1584888 | 686380725021.27 | 0 (基準) |


# グラフ別詳細結果

## roadNet-CA (roadNet-CA)

| 手法 | 実行時間 (秒) | GTEPS | vs PathMerge_BC |
|:-----|----------:|------:|----------------:|
| GPU_Opt | 2127.4 | 2.5557 | — |
| GPU_Opt_Pure | 2119.3 | 2.5655 | — |
| GPU_Opt_Pure_Chunked | 2123.2 | 2.5608 | — |

