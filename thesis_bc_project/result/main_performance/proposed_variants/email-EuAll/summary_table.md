# ベンチマーク結果: 実行時間 (秒)

| 手法 | email-EuAll |
|:-----|------:|
| GPU_Opt | 30.81 |
| GPU_Opt_Pure | 30.71 |
| GPU_Opt_Pure_Chunked | 30.75 |

# ベンチマーク結果: GTEPS (高いほど高速)

| 手法 | email-EuAll |
|:-----|------:|
| GPU_Opt | 3.1349 |
| GPU_Opt_Pure | 3.1450 |
| GPU_Opt_Pure_Chunked | 3.1410 |

# スピードアップ (vs PathMerge_BC)

(ベースライン PathMerge_BC の結果がありません)
# スケーリング分析 (GTEPS vs グラフサイズ)

| グラフ | Nodes | Edges | n×m | GPU_Opt | GPU_Opt_Pure | GPU_Opt_Pure_Chunked |
|:------|------:|------:|------:|------:|------:|------:|
| email-EuAll | 265,009 | 364,481 | 9.66e+10 | 3.1349 | 3.1450 | 3.1410 |
# 正確性検証 (max BC 値)

## email-EuAll

| 手法 | Max BC Index | Max BC Value | 差分 (%) |
|:-----|:-----------|-------------:|---------:|
| GPU_Opt | 10 | 2384894520.80 | 0 (基準) |
| GPU_Opt_Pure | 10 | 2384894520.80 | 0 (基準) |
| GPU_Opt_Pure_Chunked | 10 | 2384894520.80 | 0 (基準) |


# グラフ別詳細結果

## email-EuAll (email-EuAll)

| 手法 | 実行時間 (秒) | GTEPS | vs PathMerge_BC |
|:-----|----------:|------:|----------------:|
| GPU_Opt | 30.81 | 3.1349 | — |
| GPU_Opt_Pure | 30.71 | 3.1450 | — |
| GPU_Opt_Pure_Chunked | 30.75 | 3.1410 | — |

