# 08 メモリ容量評価（RQ3）

**入力グラフの容量と、GPU の working set 容量を明確に分ける。** 本章の主張は、修正版
325557（`325557_3216152_corrected_v1`, job 2404743, checkpoint `45352a3`）上の
**targeted feasibility boundary validation**（各構成 n=1）に基づく。実行時間は方式間の
正式な性能比較ではなく、feasibility の付随情報である。

## 8.0 容量問題の所在：入力ファイルではなく batch 依存 working set

修正版 325557 の静的サイズは小さく、いずれも HBM3（公称 96 GB）を超えない。

| 量 | 定義 | 値 |
|:--|:--|--:|
| 入力ファイル | ディスク上の CSR テキスト（`stat`） | 45,348,105 bytes ≈ 45.35 MB / 43.25 MiB |
| CSR topology | `((n+1)+2m)×4`（int32） | 27,031,448 bytes ≈ 27.03 MB / 25.78 MiB |
| BC 出力ベクトル | `n×8`（double） | 2,604,456 bytes ≈ 2.60 MB / 2.48 MiB |
| per-source state | `32n + 4·D_est + 8`（`D_est=256`, `host_pure.cu:141-157`） | 10,418,856 bytes ≈ 10.42 MB |

容量を消費するのは **始点ごとの状態配列を複数始点について同時に保持する batch 依存
working set** である。code 由来の allocation 見積りは

```
Working-set estimate = EffectiveNS × EffectiveBatch × PerSourceStateBytes
```

（Chunked は同時 resident を `SUB_BATCH` に制限）。これは allocation 見積りであり、
**実測 process RSS・実測 HBM residency・実測 migration bytes とは区別する**（未取得）。
なお batch はグラフを分割した近似ではなく、全始点を複数回の batch / sub-batch に分けて
**厳密に**処理する実行単位であり、BC を近似・省略しない。

## 8.1 GPU_Opt(UM) / Pure / Chunked の違い（設計）

UM・Pure・Chunked は独立した 3 手法ではなく、**共通 GPU 実行基盤のメモリ管理方式**である
（[04](04_method_design.md) §4.7）。要点：
- **Pure**（`host_pure.cu`）：`cudaMalloc` で working set を device に直接確保。device メモリを
  超えると **CUDA device OOM**。
- **UM / GPU_Opt**（`host_um.cu`）：`cudaMallocManaged` による managed allocation と migration で、
  device メモリを超え得る working set を扱う（入力グラフの分割格納ではない）。
- **Chunked**（`host_chunked.cu`）：source batch を `SUB_BATCH` 単位の sub-batch へ分割し、
  同時 resident working set を制限する。

## 8.2 実行可能バッチ境界（targeted validation, job 2404743, n=1）

`result/memory_scalability/corrected_325557/feasibility_boundary.tsv` /
`raw_data/corrected_325557/job_2404743/{feasibility_results,oom_evidence,implementation_manifest}.tsv`。
失敗は 0 秒ではなくマーカーで表す。**CUDA device OOM と cgroup host-memory OOM kill を区別する**。

| 構成 | 方式 | Batch | working-set 見積り | 結果 | Runtime |
|:--|:--|--:|--:|:--|--:|
| pure_b4096 | GPU_Opt_Pure | 4096 | ≈ 85.35 GB（NS=2） | SUCCESS | 65.89 s |
| pure_b8192 | GPU_Opt_Pure | 8192 | ≈ 170.70 GB（NS=2） | **CUDA out-of-memory**（device, `host_pure.cu:144`, exit 1） | — |
| um_b10240 | GPU_Opt(UM) | 10240 | ≈ 106.69 GB（NS=1） | SUCCESS | 238.67 s |
| um_b12288 | GPU_Opt(UM) | 12288 | ≈ 128.03 GB（NS=1） | **cgroup host-memory OOM kill**（exit 137, `oom_evidence=none`） | — |
| chunked_b16384 | GPU_Opt_Pure_Chunked | 16384 | resident ≈ 68.72 GB（`SUB_BATCH=6596`, num_subs=3） | SUCCESS | 66.60 s |

- **最大成功バッチ（試験範囲内）**：Pure = **b4096** < UM = **b10240** < Chunked = **b16384**。
- **Pure b8192 は CUDA device OOM**：working set 見積り ≈ 170.70 GB が free HBM（約 101.4 GB）を
  超え、`host_pure.cu:144` で `out of memory` を記録（`oom_evidence=cuda_oom`, exit 1）。
- **UM b10240 の成功は入力ファイルが大きいからではない**：batch 依存 managed allocation
  （≈ 106.69 GB）が free HBM を超える領域へ達した条件での成功であり、migration により継続できた。
- **UM b12288 は CUDA/HBM OOM ではなく cgroup host-memory OOM kill**（exit 137, SIGKILL;
  `oom_evidence=none`＝CUDA OOM 文字列なし）。UM も無制限ではない。

## 8.3 Chunked の sub-batch と binding constraint

- Chunked b16384：`SUB_BATCH=6596`, `num_subs=3`（`implementation_manifest.tsv`）。同時 resident は
  `EffectiveNS(1) × SUB_BATCH(6596) × PerSourceStateBytes ≈ 68.72 GB` に制限される。
- `SUB_BATCH=6596` は HBM 予算のみで決まった値ではない。`host_chunked.cu:136` の
  `safe_sub_batch = INT_MAX / n = 6596` という **index-safety 上限**と、HBM 予算由来の上限
  （保存値から整数除算すると 7783 始点）の小さい方が採られ、修正版 325557 では
  **index-safety 上限（6596）が binding constraint** だった（6596 < 7783）。
- batch を sub-batch に分割しても、`num_subs` 回のループで**全始点を処理**し BC を近似・省略しない。

## 8.4 実行時間の位置づけ（性能比較ではない）

上表の runtime は各構成 **n=1** の feasibility 付随情報であり、方式間の正式な性能比較に用いない
（cross-trial 集計なし）。in-capacity 領域では 3 方式の速度は同程度で、UM b10240 が相対的に
長い（238.67 s）のは managed working set が free HBM を超える領域での migration コストと解釈できる
が、これは **単一実行の観測**であり headline 性能値ではない。physical residency / RSS / migration
bytes は直接計測していない。

## 8.5 まとめ（RQ3 回答）

- 容量問題を作るのは入力グラフファイル（≈ 45.35 MB）や CSR（≈ 27.03 MB）ではなく、
  **batch × per-source state** の working set である。
- feasibility（到達バッチ）：Pure（≤b4096）< UM（≤b10240）< Chunked（≤b16384）。
- UM の目的は入力グラフの分割格納ではなく、**device メモリを超え得る working set を managed
  allocation と migration で扱う容量拡張**にある。ただし UM も b12288 で cgroup host-memory OOM
  kill になり無制限ではない。
- Chunked の目的は **同時 resident working set の制御と実行可能バッチの拡大**（sub-batch 分割）。
- いずれの方式も全始点を処理し BC を近似しない（`SUPPORTED_WITH_LIMITATIONS`）。
- **書かないこと**：「96 GB を超えるグラフを格納」「入力グラフが HBM3 を超えた」「batch で
  グラフを分割」「一部の始点だけ計算」「UM が OOM を完全回避」「Chunked は常に OOM を回避」。
  旧 tree / 旧 malformed 325557 の feasibility（b12288 OOM_OR_FAIL 等）は historical として保持し
  現行境界ではない。migration byte 量・実測 residency・実測 RSS は未計測（[11](11_limitations.md)）。
