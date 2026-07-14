# BFS カーネル 2×2 比較（forced shared / forced block・選択則非依存）

shared / block を **強制実行**した直接比較の実測（中央値・標本標準偏差・速度向上・Max BC 一致）。
自動選択則には依存しない。

## 実測 (中央値・標本標準偏差)

| グラフ | shared 中央値 (s) | block 中央値 (s) | shared 標本SD | block 標本SD | n(shared) | n(block) | 速い側 | 速度向上 (遅/速) | Max BC 一致 |
|:-----|------:|------:|------:|------:|--:|--:|:----:|------:|:----:|
| roadNet-PA | 1063.7 | 701.6 | 0.0595 | 3.5740 | 3 | 3 | block | 1.52× | ✓ |

## Max BC (shared / block)

| グラフ | shared Index | shared Value | block Index | block Value | 一致 |
|:-----|--:|------:|--:|------:|:----:|
| roadNet-PA | 557532 | 151395302679.0800 | 557532 | 151395302679.0800 | ✓ |

## 事実 (forced 比較の結果)

- roadNet-PA: shared≈1063.7s / block≈701.6s → **block が 1.52倍高速** (n(shared)=3, n(block)=3)
- shared と block の Max BC index/value は一致 (roadNet-PA)。
- 本結果は測定した強制比較グラフに限定し、**未測定グラフへ一般化しない**。

> 注: 現行実装は BFS カーネルを常に block で実行する。旧実装には平均次数に基づく自動選択則が存在したが、現在は使用していない。
