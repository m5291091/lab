# BFS カーネル 2×2 比較（forced shared / forced block・選択則非依存）

shared / block を **強制実行**した直接比較の実測（中央値・標本標準偏差・速度向上・Max BC 一致）。
自動選択則には依存しない。

## 実測 (中央値・標本標準偏差)

| グラフ | shared 中央値 (s) | block 中央値 (s) | shared 標本SD | block 標本SD | n(shared) | n(block) | 速い側 | 速度向上 (遅/速) | Max BC 一致 |
|:-----|------:|------:|------:|------:|--:|--:|:----:|------:|:----:|
| roadNet-TX | 1639.2 | 984.6 | 0.2840 | 7.2604 | 3 | 3 | block | 1.66× | ✓ |

## Max BC (shared / block)

| グラフ | shared Index | shared Value | block Index | block Value | 一致 |
|:-----|--:|------:|--:|------:|:----:|
| roadNet-TX | 400570 | 164495142042.4500 | 400570 | 164495142042.4500 | ✓ |

## 事実 (forced 比較の結果)

- roadNet-TX: shared≈1639.2s / block≈984.6s → **block が 1.66倍高速** (n(shared)=3, n(block)=3)
- shared と block の Max BC index/value は一致 (roadNet-TX)。
- 本結果は測定した強制比較グラフに限定し、**未測定グラフへ一般化しない**。

> 注: 現行実装は BFS カーネルを常に block で実行する。旧実装には平均次数に基づく自動選択則が存在したが、現在は使用していない。
