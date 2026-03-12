# Miyabi-G における Betweenness Centrality 高速化実装
## 実行・テスト手順書

**対象システム**: Miyabi-G (NVIDIA GH200 Grace Hopper Superchip)  
**研究**: 修士論文「Miyabi GH200 のアーキテクチャ特性を活用した BC 計算の高速化」

> 初めてこのリポジトリを触る場合は、先に **[README.md](../README.md)** で研究背景・GH200 アーキテクチャ・提案手法を把握してから本手順書を読んでください。

---

## 実験パイプライン全体像

```
ログインノード                       GPU 計算ノード
     |                                    |
     |--- qsub -I (インタラクティブ) ----->|
     |                              [ビルド] → brandes_runner
     |                              [動作確認] → 全実装でテスト
     |                              [正確性検証] → PASS: 5 FAIL: 0
     |
     |--- qsub run_baseline.sh ---------->| (24h バッチ)
     |                              [全グラフ × 全実装 計測]
     |                              → result_baseline/summary.tsv
     |
     |--- qsub run_ablation.sh ---------->| (2h バッチ)
     |                              [アブレーション 4グラフ × 4実装]
     |                              → result_ablation/ablation_summary.tsv
     |
     |--- qsub measure_bandwidth.sh ----->| (30min バッチ)
     |                              [HBM3 / NVLink-C2C 帯域計測]
     |                              → result_bandwidth/bandwidth.tsv
     |
     |<-- 全ジョブ完了後 ------------------------------------------
     |
     |--- python3 analysis/analyze_all.py (ログインノードで実行可)
                                    → figures/ に PDF/PNG
                                    → tables/ に LaTeX/TSV
```

**実験結果の数値まとめは [RESULT.md](../RESULT.md) を参照。**

---

## 目次

1. [ファイル一覧と研究上の役割](#1-ファイル一覧と研究上の役割)
2. [ディレクトリ構成](#2-ディレクトリ構成)
3. [STEP 1: インタラクティブジョブでビルドと動作確認](#3-step-1-インタラクティブジョブ)
4. [STEP 2: 正確性検証](#4-step-2-正確性検証)
5. [STEP 3: ベースライン計測](#5-step-3-ベースライン計測)
6. [STEP 4: アブレーションスタディ (提案手法の定量評価)](#6-step-4-アブレーションスタディ)
7. [STEP 5: Nsight Systems プロファイリング](#7-step-5-nsight-プロファイリング)
8. [STEP 6: 帯域計測](#8-step-6-帯域計測)
9. [STEP 7: バッチサイズ感度分析](#9-step-7-バッチサイズ感度分析)
10. [結果の分析・可視化](#10-結果の分析可視化)
11. [ジョブ管理コマンド一覧](#11-ジョブ管理コマンド)
12. [トラブルシューティング](#12-トラブルシューティング)
13. [グラフデータセット構成と取得方法](#13-グラフデータセット構成と取得方法)

---

## 1. ファイル一覧と研究上の役割

### 1.1 実装ファイル

| ファイル | 役割 |
|---------|------|
| `brandes_sequential.cpp` | 逐次版 Brandes アルゴリズム (CPU シングルスレッド, 参照実装) |
| `brandes_omp.cpp` | OpenMP スレッド並列版 (CPU 全コア利用) |
| `brandes_gpu.cu` | CUDA GPU 版 — `cudaMalloc` + バッチ処理 (GPU ベースライン) |
| `brandes_gpu_managed.cu` | Unified Memory 版 — CPU LPDDR5X 配置、NVLink-C2C 経由アクセス |
| `brandes_gpu_readmostly.cu` | **提案手法 1** — SetReadMostly + 適応型 Prefetch によるメモリ配置最適化 |
| `brandes_gpu_opt.cu` | **提案手法 1 + 2** — ReadMostly + 2-stream 非同期パイプライン |
| `GraphManaged.h` / `GraphManaged.cpp` | `cudaMallocManaged` 対応グラフクラス |
| `brandes.h` | 全実装の関数プロトタイプ宣言 |
| `common.h` | 共通ヘッダ (CUDA エラーチェック等) |
| `Graph.cpp` / `Graph.h` | CSR 形式グラフ読み込みクラス |
| `main.cpp` | 単体実行エントリポイント (`sequential` / `omp` / `gpu` / `gpu_managed` / `gpu_readmostly` / `gpu_opt` / `all` 対応) |
| `CMakeLists.txt` | CMake ビルド設定 |

### 1.2 スクリプト・ツールファイル

| ファイル | 役割 |
|---------|------|
| `scripts/verify_correctness.sh` | 全実装の BC 値が sequential と一致するかを検証 (5 実装) |
| `scripts/compare_bc.py` | 2 つの BC 出力ファイルの数値一致検証ツール |
| `scripts/run_baseline.sh` | PBS バッチジョブ: 全グラフ・全実装のベースライン計測 |
| `scripts/run_ablation.sh` | PBS バッチジョブ: アブレーションスタディ (4 実装 × 4 グラフ) |
| `scripts/run_profile.sh` | PBS バッチジョブ: Nsight Systems プロファイリング |
| `scripts/run_batchsize_sweep.sh` | バッチサイズ感度分析 (インタラクティブジョブ内) |
| `scripts/measure_bandwidth.sh` | PBS バッチジョブ: HBM3 / NVLink-C2C 実効帯域計測 |
| `tools/bandwidth_benchmark.cu` | HBM3 / NVLink-C2C 実効帯域を直接計測するベンチマーク |
| `tools/download_snap_graphs.sh` | SNAP データセットをダウンロードして CSR 形式に変換 |
| `tools/snap_to_csr.py` | エッジリスト形式グラフを CSR 形式に変換 |
| `tools/gen_graph.py` | 任意スケールの合成グラフ (BA・ER・グリッド) を生成 |
| `analysis/analyze_all.py` | 全実験結果を読み込んで図 (PDF/PNG) と表 (LaTeX) を生成 |

### 1.3 ビルド成果物 (git 管理外・`.gitignore` で除外)

| ディレクトリ | 内容 |
|------------|------|
| `build_miyabi/` | ビルド成果物 (`brandes_runner`, `bandwidth_benchmark`) |
| `build_miyabi/result_baseline/` | ベースライン実験結果 TSV |
| `build_miyabi/result_ablation/` | アブレーションスタディ結果 TSV |
| `build_miyabi/result_bandwidth/` | NVLink-C2C / HBM3 実効帯域測定結果 |
| `build_miyabi/result_profile/` | Nsight Systems プロファイルデータ |
| `build_miyabi/result_batchsize_sweep/` | バッチサイズ感度分析結果 |
| `build_miyabi/result_verify/` | 正確性検証ファイル |

---

## 2. ディレクトリ構成

```
lab/
├── research/
│   ├── brandes_sequential.cpp
│   ├── brandes_omp.cpp
│   ├── brandes_gpu.cu
│   ├── brandes_gpu_managed.cu           <- Unified Memory 版
│   ├── brandes_gpu_readmostly.cu        <- 提案手法 1: ReadMostly + Prefetch
│   ├── brandes_gpu_opt.cu               <- 提案手法 1+2: + 2-stream 非同期
│   ├── GraphManaged.cpp / GraphManaged.h
│   ├── brandes.h / common.h
│   ├── Graph.cpp / Graph.h
│   ├── main.cpp
│   ├── CMakeLists.txt
│   ├── HOWTO.md
│   ├── .gitignore
│   │
│   ├── scripts/
│   │   ├── verify_correctness.sh
│   │   ├── compare_bc.py
│   │   ├── run_baseline.sh
│   │   ├── run_ablation.sh
│   │   ├── run_profile.sh
│   │   ├── run_batchsize_sweep.sh
│   │   └── measure_bandwidth.sh
│   │
│   ├── tools/
│   │   ├── bandwidth_benchmark.cu
│   │   ├── download_snap_graphs.sh
│   │   ├── snap_to_csr.py
│   │   └── gen_graph.py
│   │
│   └── analysis/
│       ├── analyze_all.py
│       ├── figures/
│       └── tables/
│
└── data/
    ├── benchmark_7000_41459        <- 7K/41K   全実装比較・正確性検証用
    ├── benchmark_11023_62184       <- 11K/62K  全実装比較用
    ├── benchmark_85830.data        <- 85K/241K
    ├── 56438_300801                <- 56K/300K
    ├── 325557_3216152              <- 325K/3.2M  大規模 GPU ベンチマーク
    └── snap/                       <- SNAP 実世界グラフ (9 グラフ)
```

---

## 3. STEP 1: インタラクティブジョブ

**目的**: GPU 計算ノード上でビルドと全実装の動作確認。

```bash
# インタラクティブジョブ取得 (単一ノード, 最大 2 時間)
qsub -I -q interact-g -l select=1:ncpus=72 -l walltime=02:00:00 -W group_list=gj17
```

`qsub: job XXXXXX.opbs ready` と表示されたら計算ノードへ接続完了。

> **"Job violates resource limits"** が出る場合:  
> `qstat --rscuse` で `interact-g` の使用率を確認。混雑時はバッチジョブを使う。

### ビルド

```bash
cd /work/gj17/j17000/m5291091/lab/research
mkdir -p build_miyabi && cd build_miyabi
rm -f CMakeCache.txt                         # 再ビルド時はキャッシュを削除
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
# 生成バイナリ: brandes_runner, bandwidth_benchmark
```

### 動作確認

```bash
cd /work/gj17/j17000/m5291091/lab/research/build_miyabi

# 全実装を小グラフで一括テスト
./brandes_runner all ../../data/benchmark_7000_41459

# 個別実装テスト
./brandes_runner gpu_readmostly  ../../data/benchmark_7000_41459
./brandes_runner gpu_opt         ../../data/325557_3216152
```

**出力フォーマット** (タブ区切り TSV):
```
GPU_Opt    325557_3216152    2.341    8.822
```
`実装名  グラフ名  実行時間(秒)  GTEPS`

### 実装名一覧

| 引数 | 実装 |
|------|------|
| `sequential` | 逐次版 (CPU シングルスレッド) |
| `omp` | OpenMP 並列版 (CPU 全コア) |
| `gpu` | CUDA GPU 版 (cudaMalloc) |
| `gpu_managed` | Unified Memory 版 |
| `gpu_readmostly` | **提案手法 1** (SetReadMostly + 適応型 Prefetch) |
| `gpu_opt` | **提案手法 1+2** (ReadMostly + 2-stream 非同期) |
| `all` | 全実装を一括実行 |

---

## 4. STEP 2: 正確性検証

**目的**: 全実装の BC 値が sequential と一致することを定量的に証明 (論文の正確性保証)。  
**実行場所**: インタラクティブジョブ内 (GPU ノード必須)

```bash
cd /work/gj17/j17000/m5291091/lab/research
bash scripts/verify_correctness.sh                          # デフォルト: benchmark_7000_41459
bash scripts/verify_correctness.sh ../../data/56438_300801  # グラフ指定も可
```

**期待される出力** (全 5 実装が PASS):
```
PASS: All BC values match within tolerance.   # OpenMP
PASS: All BC values match within tolerance.   # GPU
PASS: All BC values match within tolerance.   # GPU_Managed
PASS: All BC values match within tolerance.   # GPU_ReadMostly  <- 提案手法 1
PASS: All BC values match within tolerance.   # GPU_Opt
========================================
  PASS: 5  FAIL: 0
========================================
```

---

## 5. STEP 3: ベースライン計測

**目的**: 全グラフ・全実装の実行時間計測。Sequential → OpenMP → GPU → GPU_Managed → GPU_ReadMostly → GPU_Opt の段階的高速化を記録する。  
**使用キュー**: `regular-g`  **walltime**: 24 時間

```bash
cd /work/gj17/j17000/m5291091/lab/research
qsub scripts/run_baseline.sh
# -> 結果: build_miyabi/result_baseline/summary.tsv
```

**グラフサイズ別の計測対象**:

| グラフ規模 | 対象グラフ | 計測対象実装 |
|-----------|---------|-----------|
| small (7K, 11K) | benchmark_7000_41459, benchmark_11023_62184 | sequential, omp, gpu, gpu_managed, gpu_readmostly, gpu_opt |
| medium (56K〜410K) | snap 実世界グラフ 7 本 | omp, gpu, gpu_managed, gpu_readmostly, gpu_opt |
| large (800K〜2M) | roadNet-*, web-Google | gpu, gpu_readmostly, gpu_opt |

---

## 6. STEP 4: アブレーションスタディ

**目的**: 提案手法の独立した寄与量を定量化する。本研究の核心実験。

| ステップ | 比較 | 測定する効果 |
|---------|------|------|
| 1 | GPU vs GPU_Managed | Unified Memory の基本コスト (GH200 では NVLink-C2C がボトルネックとなり意図的に大幅劣化) |
| 2 | GPU_Managed vs GPU_ReadMostly | **提案手法 1**: SetReadMostly + 適応型 Prefetch 効果 |
| 3 | GPU_ReadMostly vs GPU_Opt | **提案手法 2**: 2-stream 非同期パイプライン効果 |

```bash
# バッチジョブとして投入 (推奨, select=1, walltime=2h)
cd /work/gj17/j17000/m5291091/lab/research
qsub scripts/run_ablation.sh
# -> 結果: build_miyabi/result_ablation/ablation_summary.tsv

# または インタラクティブジョブ内で直接実行 (~20 分)
bash scripts/run_ablation.sh
```

**対象グラフ**: benchmark_7000_41459 (7K), benchmark_11023_62184 (11K), 56438_300801 (56K), 325557_3216152 (325K)

### GPU_ReadMostly の技術的意義

GH200 では Unified Memory を使用すると、グラフ構造データがデフォルトで CPU 側 LPDDR5X に配置される。  
GPU がこのデータにアクセスする際は NVLink-C2C (実効帯域 156 GB/s) を経由するため、  
HBM3 直接アクセス (実効帯域 1488 GB/s) と比べて約 9.6 倍のレイテンシが生じる。

`GPU_ReadMostly` は `cudaMemAdviseSetReadMostly` と `cudaMemPrefetchAsync` を用い、  
グラフのトポロジデータを HBM3 に積極的にキャッシュすることで NVLink-C2C ボトルネックを解消する。

> **注意**: テストグラフ (最大 27 MB) はすべて HBM3 配置閾値 (HBM3 容量の 35%) を下回るため、  
> 全グラフで HBM3 配置が選択される。結果として `GPU ≈ GPU_ReadMostly` の性能となるが、  
> これは意図通りの動作であり、「手法 1 が GPU 性能を完全回復した」ことの証拠である。  
> 重要な比較は **GPU_Managed (5 倍遅い) → GPU_ReadMostly (GPU 同等)** である。

---

## 7. STEP 5: Nsight プロファイリング

**目的**: GPU カーネルの詳細プロファイル (SM 稼働率・メモリ帯域・レイテンシ)。  
**使用キュー**: `regular-g`  **walltime**: 1 時間

```bash
cd /work/gj17/j17000/m5291091/lab/research
qsub scripts/run_profile.sh
# -> 結果: build_miyabi/result_profile/*.nsys-rep
```

---

## 8. STEP 6: 帯域計測

**目的**: GH200 の HBM3 / NVLink-C2C 実効帯域を定量化する。  
**使用キュー**: `regular-g`  **walltime**: 30 分

```bash
cd /work/gj17/j17000/m5291091/lab/research
qsub scripts/measure_bandwidth.sh
# -> 結果: build_miyabi/result_bandwidth/bandwidth.tsv
```

**計測済み参考値**:

| 転送種別 | 実効帯域 | 理論値 | 達成率 |
|---------|---------|-------|--------|
| HBM3 DtoD | 1488 GB/s | 4020 GB/s | 37% |
| Pinned HtoD | 425 GB/s | 900 GB/s | 47% |
| NVLink-C2C Prefetch | 156 GB/s | 900 GB/s | 17% |

NVLink-C2C の実効帯域 (156 GB/s) が GPU_Managed 劣化の直接的な原因であり、  
HBM3 と比較して 9.6 倍の帯域差が BC 実行時間に直接反映される。

---

## 9. STEP 7: バッチサイズ感度分析

**目的**: GPU カーネルのバッチサイズ (並列ソース頂点数) の最適値を決定する。  
**実行場所**: インタラクティブジョブ内 (GPU ノード必須)

```bash
cd /work/gj17/j17000/m5291091/lab/research
bash scripts/run_batchsize_sweep.sh
# -> 結果: build_miyabi/result_batchsize_sweep/
```

**計測済み参考値** (benchmark_11023_62184):

| バッチサイズ | 実行時間 | GTEPS |
|------------|---------|-------|
| 32 | 0.708 s | 0.97 |
| 64 | 0.440 s | 1.56 |
| **128** | **0.364 s** | **1.88** |
| 256 | 0.387 s | 1.77 |
| 512 | 0.371 s | 1.85 |

BatchSize=128 が最速。デフォルト値として採用済み。

---

## 10. 結果の分析・可視化

全実験データを収集して図・表を生成する:

```bash
cd /work/gj17/j17000/m5291091/lab/research/analysis
python3 analyze_all.py
# -> figures/ に PDF/PNG、tables/ に TSV/LaTeX を出力
```

**主要な出力図**:

| ファイル | 内容 |
|---------|------|
| `exec_time_vs_graphsize.pdf` | グラフサイズ vs 実行時間 (全実装) |
| `gteps_comparison.pdf` | GTEPS 比較 |
| `phase2_memory_comparison.pdf` | メモリ配置戦略の比較 |
| `ablation_study.pdf` | **アブレーション: 手法 1 / 手法 2 の独立寄与** |
| `bandwidth_comparison.pdf` | HBM3 / NVLink-C2C 帯域比較 |
| `batchsize_sensitivity.pdf` | バッチサイズ感度分析 |

**主要な出力表**:

| ファイル | 内容 |
|---------|------|
| `exec_time_table.tex` | 実行時間表 (LaTeX) |
| `gteps_table.tex` | GTEPS 表 (LaTeX) |
| `ablation_table.tex` | アブレーション表 (LaTeX) |
| `bandwidth_table.tex` | 帯域表 (LaTeX) |

---

## 11. ジョブ管理コマンド

```bash
# ジョブ一覧確認
qstat

# ジョブ詳細 (開始時刻・待ち理由等)
qstat -v

# ノード空き状況
qstat --rscuse

# ジョブ削除
qdel <JobID>
```

---

## 12. トラブルシューティング

### cmake に失敗する / CUDA コンパイラが見つからない
- インタラクティブジョブ (`interact-g`) または `regular-g` キュー上で実行すること
- ログインノードでは CUDA コンパイラが利用不可
- 再ビルド時は `rm -f build_miyabi/CMakeCache.txt` を先に実行

### "Job violates resource limits" (qsub エラー)
- `qstat --rscuse` で `interact-g` / `regular-g` の使用率を確認
- トークン予算超過: `select=` を減らすか、別キューを使う

### GPU 実装が遅い / GTEPS が期待値より低い
- バッチサイズを確認: `run_batchsize_sweep.sh` で最適値を計測
- グラフが小さすぎる場合: GPU の立ち上がりコストが支配的になる
- `gpu_managed` は NVLink-C2C 帯域 (156 GB/s) がボトルネックのため 56K+ グラフで意図的に低速

### Graph クラスの API

```cpp
graph.getAdjacencyListPointers()  // CSR 行オフセット配列 R
graph.getAdjacencyList()          // CSR 列インデックス配列 C
graph.getNodeCount()              // 頂点数
graph.getEdgeCount()              // 有向辺数 (edge_size = 2 × getEdgeCount())
```

---

## 13. グラフデータセット構成と取得方法

### 同梱グラフ

```
data/
├── benchmark_7000_41459     7,000 頂点  /  41,459 辺
├── benchmark_11023_62184   11,023 頂点  /  62,184 辺
├── benchmark_85830.data    85,830 頂点  / 241,167 辺
├── 56438_300801            56,438 頂点  / 300,801 辺
├── 325557_3216152         325,557 頂点  / 3,216,152 辺
└── random                   中規模ランダムグラフ (~32K 頂点)
```

### SNAP 実世界グラフの取得

```bash
bash research/tools/download_snap_graphs.sh
```

取得後は `data/snap/` 以下に CSR 形式で配置される:

```
data/snap/
├── email-EuAll    265K / 420K 辺
├── amazon0302     262K / 1.2M 辺
├── web-Stanford   281K / 2.3M 辺
├── web-NotreDame  325K / 1.5M 辺
├── amazon0505     410K / 2.4M 辺
├── web-Google     875K / 5.1M 辺
├── roadNet-PA    1.09M / 1.5M 辺
├── roadNet-TX    1.38M / 1.9M 辺
└── roadNet-CA    1.97M / 2.8M 辺
```

### 独自グラフの生成

```bash
# Barabasi-Albert モデル (スケールフリーグラフ)
python3 research/tools/gen_graph.py --model ba --nodes 50000 --m 3 -o data/ba_50k

# Erdos-Renyi モデル (ランダムグラフ)
python3 research/tools/gen_graph.py --model er --nodes 50000 --p 0.0001 -o data/er_50k
```

