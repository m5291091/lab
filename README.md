# GH200 における Betweenness Centrality 高速化実装

> **修士論文**「Miyabi GH200 のアーキテクチャ特性を活用した Betweenness Centrality 計算の高速化」  
> 対象システム: **Miyabi-G (NVIDIA GH200 Grace Hopper Superchip)**

---

## このリポジトリについて

本リポジトリは、スーパーコンピュータ Miyabi-G に搭載された NVIDIA GH200 Grace Hopper Superchip 上で、  
グラフ解析アルゴリズムの一種である **Betweenness Centrality (BC)** を高速に計算するための実装と実験コードをまとめたものです。

実験結果の詳細は **[RESULT.md](./RESULT.md)** を参照してください。

---

## 背景と研究課題

### GH200 のメモリ構成

GH200 は、CPU と GPU が一体化した特殊なチップ構成を持ちます。

```
  ┌─────────────────────────────────────────────────────┐
  │                  GH200 Grace Hopper                 │
  │                                                     │
  │  ┌──────────────┐   NVLink-C2C    ┌──────────────┐  │
  │  │  Grace CPU   │  <== 900 GB/s ==>  Hopper GPU  │  │
  │  │  (72 cores)  │                │   (132 SM)    │  │
  │  │              │                │               │  │
  │  │  LPDDR5X     │                │  HBM3         │  │
  │  │  480 GB      │                │  96 GB        │  │
  │  │  実効: ~170 GB/s│                │  実効: 1488 GB/s│  │
  │  └──────────────┘                └──────────────┘  │
  └─────────────────────────────────────────────────────┘
```

GH200 は **Unified Memory** (CPU/GPU が同一アドレス空間でメモリを共有する仕組み) をサポートしており、  
プログラマが明示的にデータ転送を書かなくてもよい便利な機能です。  
しかし、**デフォルトでは CPU 側の LPDDR5X にデータが配置され**、  
GPU からのアクセスは低速な NVLink-C2C 経由となります。

```
  GPU からデータアクセスした場合の帯域比較:

  [HBM3 直接アクセス]       : 1488 GB/s  (非常に高速)
  [NVLink-C2C 経由 Prefetch]: ~156 GB/s  (約 9.6 倍遅い)
  [LPDDR5X ランダムアクセス]: さらに低速
```

### 研究課題

BC 計算は各頂点から BFS (幅優先探索) を繰り返し実行するアルゴリズムで、  
グラフ構造データ (CSR 形式の隣接リスト) への大量のランダムアクセスが発生します。

Unified Memory を使用した実装 (`GPU_Managed`) は、  
コードがシンプルになる一方で、グラフデータが CPU 側に配置されるため  
**GPU ベースライン比で最大 5 倍以上の性能低下**が生じます。

本研究では、GH200 の **NVLink-C2C ボトルネックを解消**しつつ、  
Unified Memory の利便性を維持する 2 つの最適化手法を提案します。

---

## 提案手法

| 手法 | 概要 | 実装ファイル |
|------|------|------------|
| **手法 1**: ReadMostly + Prefetch | グラフ構造データに `cudaMemAdviseSetReadMostly` を適用し、GPU の HBM3 に積極的にキャッシュすることで NVLink-C2C ボトルネックを解消 | `brandes_gpu_readmostly.cu` |
| **手法 2**: 2-stream 非同期パイプライン | 2 本の CUDA ストリームが交互にバッチを処理することで、`cudaMemsetAsync` による初期化を GPU カーネル実行と並列化し、待機時間をゼロにする | `brandes_gpu_opt.cu` |

**手法 1 + 2 の組み合わせ** (`GPU_Opt`) が最終提案実装です。

---

## 実装一覧

6 種類の実装を用意し、段階的な性能向上を比較します。

| 実装名 | ソースファイル | 概要 | 位置づけ |
|--------|-------------|------|---------|
| `Sequential` | `brandes_sequential.cpp` | CPU シングルスレッド版 | 最遅の参照実装 |
| `OpenMP` | `brandes_omp.cpp` | CPU 全コア (最大 72 スレッド) 並列化 | CPU ベースライン |
| `GPU` | `brandes_gpu.cu` | CUDA + `cudaMalloc` でデータを HBM3 に明示確保 | GPU ベースライン |
| `GPU_Managed` | `brandes_gpu_managed.cu` | `cudaMallocManaged` で Unified Memory を使用 | ナイーブ UM 実装 (意図的に遅い) |
| `GPU_ReadMostly` | `brandes_gpu_readmostly.cu` | **手法 1**: ReadMostly + 適応型 Prefetch | アブレーション用 中間実装 |
| `GPU_Opt` | `brandes_gpu_opt.cu` | **手法 1 + 2**: ReadMostly + 2-stream 非同期 | **最終提案実装** |

> **アブレーションの流れ**:  
> `GPU_Managed` (5倍遅) → `GPU_ReadMostly` (GPU と同等: 手法1の効果) → `GPU_Opt` (さらに高速化: 手法2の効果)

---

## ディレクトリ構成

```
lab/
├── README.md               ← このファイル
├── RESULT.md               ← 実験結果まとめ (論文執筆用)
│
├── research/               ← ソースコード・スクリプト・実験環境
│   ├── brandes_*.cpp/.cu   ← 各実装のソースコード
│   ├── CMakeLists.txt      ← CMake ビルド設定
│   ├── HOWTO.md            ← 詳細な実験手順書
│   │
│   ├── scripts/            ← PBS バッチジョブスクリプト
│   ├── tools/              ← グラフ取得・変換・生成ユーティリティ
│   │
│   ├── build_miyabi/       ← ビルド成果物・実験結果 (git 管理外)
│   │   ├── result_baseline/      → ベースライン計測結果
│   │   ├── result_ablation/      → アブレーション実験結果
│   │   ├── result_bandwidth/     → 帯域計測結果
│   │   ├── result_batchsize_sweep/ → バッチサイズ感度分析
│   │   └── result_profile/       → Nsight プロファイルデータ
│   │
│   └── analysis/           ← 結果分析・可視化スクリプト
│       ├── analyze_all.py  ← 全実験データを読み込んで図・表を生成
│       ├── figures/        ← 生成された図 (PDF/PNG)
│       └── tables/         ← 生成された表 (TSV/LaTeX)
│
└── data/                   ← グラフデータ (CSR 形式)
    ├── benchmark_*/        ← 合成ベンチマークグラフ
    ├── 325557_3216152      ← 大規模グラフ (325K 頂点 / 3.2M 辺)
    └── snap/               ← SNAP 実世界グラフ (9 本: 265K〜1.97M ノード)
```

---

## クイックスタート

### 1. ビルド (GPU 計算ノード上で実行すること)

```bash
# インタラクティブジョブを取得
qsub -I -q interact-g -l select=1:ncpus=72 -l walltime=02:00:00 -W group_list=gj17

# ビルド
cd /work/gj17/j17000/m5291091/lab/research
mkdir -p build_miyabi && cd build_miyabi
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

> **注意**: ログインノードでは CUDA コンパイラが利用できません。必ず GPU 計算ノード上でビルドしてください。

### 2. 動作確認

```bash
cd /work/gj17/j17000/m5291091/lab/research/build_miyabi

# 全実装を小グラフで一括テスト
./brandes_runner all ../../data/benchmark_7000_41459

# 個別実行例
./brandes_runner gpu_opt ../../data/325557_3216152
```

出力形式 (タブ区切り): `実装名  グラフ名  実行時間(秒)  GTEPS`

### 3. 本番実験 (PBS バッチジョブ)

```bash
cd /work/gj17/j17000/m5291091/lab/research

qsub scripts/run_baseline.sh    # ベースライン全計測 (walltime: 24h)
qsub scripts/run_ablation.sh    # アブレーションスタディ (walltime: 2h)
qsub scripts/measure_bandwidth.sh  # 帯域計測
```

### 4. 結果の可視化

```bash
cd /work/gj17/j17000/m5291091/lab/research/analysis
python3 analyze_all.py
# -> figures/ に PDF/PNG、tables/ に TSV/LaTeX が生成されます
```

---

## 主要な実験結果 (概要)

詳細は **[RESULT.md](./RESULT.md)** を参照。

| 比較 | 代表スピードアップ | 意味 |
|------|-----------------|------|
| Sequential vs GPU_Opt | 最大 **27.9 倍** | GPU 並列化の効果 |
| OpenMP vs GPU_Opt | 最大 **44.1 倍** (web-NotreDame) | GPU の優位性 |
| GPU_Managed vs GPU_ReadMostly | 平均 **4.9 倍** 改善 | 手法 1 の効果 (NVLink-C2C ボトルネック解消) |
| GPU_ReadMostly vs GPU_Opt | 最大 **1.3 倍** 改善 | 手法 2 の効果 (2-stream 非同期) |
| GPU vs GPU_Opt | 最大 **1.16 倍** 改善 | 提案手法全体の GPU 超越 |

最高スループット: **43.9 GTEPS** (web-NotreDame, GPU_Opt, 325K nodes / 1.5M edges)

---

## グラフデータセット

実験に使用したグラフデータの詳細は **[data/README.md](./data/README.md)** を参照。

| 規模 | グラフ数 | 頂点数の範囲 | 代表例 |
|------|---------|-----------|------|
| small | 2 | 7K〜11K | benchmark_7000_41459 |
| medium | 7 | 56K〜410K | web-Stanford, amazon0505 |
| large | 5 | 875K〜1.97M | web-Google, roadNet-CA |

SNAP グラフの追加取得:
```bash
bash research/tools/download_snap_graphs.sh
```

---

## 詳細ドキュメント

| ドキュメント | 内容 |
|-----------|------|
| [RESULT.md](./RESULT.md) | 実験結果の詳細・図の解説・考察 |
| [research/HOWTO.md](./research/HOWTO.md) | 実験手順の詳細 (ビルド〜実行〜分析) |
| [data/README.md](./data/README.md) | グラフデータセットの仕様・取得方法 |
