# 実験手順書

**対象システム**: Miyabi-G (NVIDIA GH200 Grace Hopper Superchip)  
**研究**: 修士論文「Miyabi GH200 のアーキテクチャ特性を活用した BC 計算の高速化」

詳細な手順 (ビルド〜正確性検証〜個別スクリプト実行) は **[HOWTO.md](./HOWTO.md)** に記載している。  
本文書はビルドから本番実験まで、証拠を一括取得する手順をまとめたものである。

---

## 目次

1. [前提環境](#1-前提環境)
2. [リポジトリのセットアップ](#2-リポジトリのセットアップ)
3. [ビルド手順](#3-ビルド手順)
4. [インタラクティブテスト（動作確認）](#4-インタラクティブテスト動作確認)
5. [個別実験スクリプトの実行](#5-個別実験スクリプトの実行)
6. [本番実験（全証拠を一括取得）](#6-本番実験全証拠を一括取得)
7. [結果の確認と図の生成](#7-結果の確認と図の生成)
8. [5段階アブレーション設計の早見表](#8-5段階アブレーション設計の早見表)
9. [よくあるトラブルと対処法](#9-よくあるトラブルと対処法)

---

## 1. 前提環境

Miyabi-G（NVIDIA GH200 Grace Hopper Superchip 搭載）での実行を前提とする。
sm_90 以外の環境では `CMake` の `DCMAKE_CUDA_ARCHITECTURES` を適宜変更すること（[§9参照](#9-よくあるトラブルと対処法)）。

**必要なモジュール**（Miyabi-G 標準環境）:

| モジュール | バージョン |
|-----------|---------|
| CUDA | 12.x |
| GCC | 11 以上 |
| CMake | 3.20 以上 |

**インタラクティブジョブの取得**（ビルド・動作確認に使用）:

```bash
qsub -I -q interact-g -l select=1:ncpus=72 -l walltime=02:00:00 -W group_list=gj17
```

`qsub: job XXXXXX.opbs ready` と表示されたら GPU 計算ノードへ接続完了。  
ログインノードでは CUDA コンパイラが使えないため、ビルドは必ずこのノード上で行う。

---

## 2. リポジトリのセットアップ

```bash
git clone https://github.com/m5291091/lab.git
cd lab
git checkout copilot/improve-existing-research-code
```

---

## 3. ビルド手順

```bash
cd research
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j$(nproc)
# 生成される実行ファイル: build/brandes_runner, build/bandwidth_benchmark
```

実際の Miyabi-G 環境では `build_miyabi/` を使う慣例になっている:

```bash
mkdir -p build_miyabi && cd build_miyabi
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

> 再ビルド時は `rm -f CMakeCache.txt` でキャッシュを削除してから cmake を再実行する。

---

## 4. インタラクティブテスト（動作確認）

各 Stage の単体テストを小規模グラフで行う。

```bash
# グラフデータの場所を確認
ls ../../data/
```

### Stage 別の実行例

```bash
# Stage 0: HBM3専用（シングルストリーム）
./brandes_runner gpu ../../data/benchmark_7000_41459

# Stage 1: HBM3専用（ダブルバッファ）
./brandes_runner gpu_stream ../../data/benchmark_7000_41459

# Stage 2: UVM素朴実装
./brandes_runner gpu_managed ../../data/benchmark_7000_41459

# Stage 3: UVM + ReadMostly（デフォルト閾値 0.35）
./brandes_runner gpu_readmostly ../../data/benchmark_7000_41459

# Stage 3: 閾値を下げて topo_on_gpu=false を強制実行（テスト用）
./brandes_runner gpu_readmostly ../../data/benchmark_7000_41459 --topo-threshold 0.001

# Stage 4: UVM + ReadMostly + ダブルバッファ（提案手法）
./brandes_runner gpu_opt ../../data/benchmark_7000_41459
```

全 Stage を一括実行する場合:

```bash
./brandes_runner all ../../data/benchmark_7000_41459
```

### 正解確認（Sequential との比較）

```bash
./brandes_runner sequential ../../data/benchmark_7000_41459 --dump-bc > out_seq.txt
./brandes_runner gpu_opt    ../../data/benchmark_7000_41459 --dump-bc > out_gpu.txt
diff out_seq.txt out_gpu.txt  # 差分がなければ正解
```

スクリプトで全実装を一括検証する場合:

```bash
bash scripts/verify_correctness.sh
# 期待出力: PASS: 6  FAIL: 0
```

---

## 5. 個別実験スクリプトの実行

各スクリプトは単独でも実行できる。本番実験（§6）では `run_all_experiments.sh` が自動的に呼び出す。

```bash
# メモリ帯域計測（~30 分）
bash scripts/measure_bandwidth.sh
# -> build_miyabi/result_bandwidth/bandwidth.tsv

# 全実装 × 全グラフ（~24 時間、バッチジョブ推奨）
qsub scripts/run_baseline.sh
# -> build_miyabi/result_baseline/summary.tsv

# アブレーション（4実装 × 4グラフ、~2 時間）
qsub scripts/run_ablation.sh
# -> build_miyabi/result_ablation/ablation_summary.tsv

# バッチサイズ感度分析（インタラクティブジョブ内）
bash scripts/run_batchsize_sweep.sh
# -> build_miyabi/result_batchsize_sweep/

# Nsight プロファイル（代表3グラフ、~2 時間）
qsub scripts/run_profile.sh
# -> build_miyabi/result_profile/*.nsys-rep
```

### 閾値感度実験（`--topo-threshold`）

`run_all_experiments.sh` の Step 3 として自動実行されるが、単独で試す場合:

```bash
for T in 0.001 0.01 0.1 0.35; do
    echo "=== threshold=${T} ==="
    ./build_miyabi/brandes_runner gpu_readmostly \
        ../../data/snap/web-Google --topo-threshold ${T}
done
```

---

## 6. 本番実験（全証拠を一括取得）

`run_all_experiments.sh` を1回実行するだけで、帯域計測→全グラフ実行→閾値感度→Nsight プロファイル→可視化 が順に実行される。

### インタラクティブジョブ上での実行

```bash
bash scripts/run_all_experiments.sh 2>&1 | tee logs/run_$(date +%Y%m%d).log
```

### PBS バッチジョブとして投入（推奨）

`jobs/` ディレクトリにジョブスクリプトを置いて `qsub` で投入する。
以下は Miyabi-G での標準設定例:

```bash
#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=72:ngpus=1:mem=480gb
#PBS -l walltime=24:00:00
#PBS -N bc_all_experiments

cd $PBS_O_WORKDIR/research
bash scripts/run_all_experiments.sh
```

```bash
# 投入
qsub jobs/run_all.sh

# 状況確認
qstat -v
```

### 実行順序と所要時間の目安

| Step | 内容 | 時間目安 |
|------|------|---------|
| 0 | ビルド | ~2 min |
| 1 | メモリ帯域計測 | ~30 min |
| 2 | 全6実装 × 全グラフ 計測 | ~24 h |
| 3 | 閾値感度実験（0.001/0.01/0.1/0.35） | ~1 h |
| 4 | Nsight プロファイル（3グラフ） | ~2 h |
| 5 | analyze_all.py で可視化 | ~5 min |

各ステップが失敗しても次のステップを続行する（`run_or_warn` による制御）。

### 出力先

```
research/
├── logs/
│   └── run_all_YYYYMMDD_HHMMSS.log   実行ログ（タイムスタンプ付き）
└── data/
    ├── bandwidth/bandwidth_*.tsv      帯域計測結果
    ├── timing/timing_*.tsv            実行時間（全実装×全グラフ）
    ├── threshold/threshold_*.tsv      閾値感度実験結果
    └── profile/                       Nsight プロファイルデータ
```

---

## 7. 結果の確認と図の生成

```bash
# 自動で図・表が生成される
python3 analysis/analyze_all.py

# 生成先
ls analysis/figures/   # PDF/PNG ファイル群
ls analysis/tables/    # TSV/LaTeX ファイル群
```

### 生成される図

| ファイル | 内容 |
|---------|------|
| `exec_time_vs_graphsize.pdf` | グラフサイズ vs 実行時間（全実装比較） |
| `gteps_comparison.pdf` | GTEPS スループット比較 |
| `ablation_study.pdf` | アブレーション: Stage 0〜4 の段階的効果 |
| `bandwidth_comparison.pdf` | HBM3 / NVLink-C2C 実効帯域比較 |
| `batchsize_sensitivity.pdf` | バッチサイズ感度分析 |
| `phase2_memory_comparison.pdf` | メモリ配置戦略（HBM3 vs CPU）の比較 |
| `speedup_vs_Sequential.pdf` | Sequential に対するスピードアップ率 |
| `speedup_vs_OpenMP.pdf` | OpenMP に対するスピードアップ率 |

### 生成される TSV テーブル

| ファイル | 内容 |
|---------|------|
| `exec_time_table.tsv` | 全実装・全グラフの実行時間（秒） |
| `gteps_table.tsv` | 全実装・全グラフの GTEPS スループット |
| `ablation_speedup_vs_gpu.tsv` | GPU ベースライン比のアブレーションスピードアップ率 |
| `ablation_gteps_table.tsv` | アブレーション対象グラフの GTEPS 比較 |
| `speedup_table_vs_Sequential.tsv` | Sequential 比スピードアップ（全グラフ） |
| `speedup_table_vs_OpenMP.tsv` | OpenMP 比スピードアップ（全グラフ） |
| `bandwidth_table.tsv` | HBM3 / NVLink-C2C 帯域計測値 |

---

## 8. 5段階アブレーション設計の早見表

```
Stage 0  brandes_gpu             HBM3専用・シングルストリーム
Stage 1  brandes_gpu_stream      HBM3専用・ダブルバッファ          ← 新規追加
Stage 2  brandes_gpu_managed     UVM・シングルストリーム
Stage 3  brandes_gpu_readmostly  UVM・ReadMostly（手法1）
Stage 4  brandes_gpu_opt         UVM・ReadMostly＋ダブルバッファ（手法1+2）
```

各 Stage が答える研究問い:

| 比較 | 研究問い |
|------|---------|
| Stage 0 → Stage 1 | ダブルバッファ単体は HBM3 専用でどれだけ効くか？ |
| Stage 0 → Stage 2 | UVM 導入だけで何倍遅くなるか（NVLink-C2C ボトルネック）？ |
| Stage 2 → Stage 3 | SetReadMostly（手法1）はどれだけボトルネックを解消するか？ |
| Stage 3 → Stage 4 | cudaMemsetAsync + 2ストリーム（手法2）で何%追加改善するか？ |
| Stage 1 → Stage 4 | UVM + 手法1+2 は HBM3 専用ダブルバッファと互角か？ |
| Stage 0 → Stage 4 | 提案手法全体で元の GPU ベースラインを超えるか？ |

---

## 9. よくあるトラブルと対処法

### CUDA out of memory エラー

大規模グラフ（roadNet-CA 等）で `cudaMalloc` が失敗する場合、バッチサイズを下げる:

```bash
# CMakeLists.txt の BATCH_SIZE_DEFAULT を小さくして再ビルドするか、
# run_batchsize_sweep.sh で実際の最適値を確認する
bash scripts/run_batchsize_sweep.sh
```

Stage 0 (`brandes_gpu`) はすべてのデータを HBM3 に確保するため、96 GB を超えるグラフでは OOM になる。
その場合は Stage 3/4 の UVM 実装を使う。

### `topo_on_gpu = false` が発動しない場合

閾値（デフォルト 0.35）は HBM3 容量（96 GB）の 35%、つまり約 33.6 GB が上限。
実験グラフのトポロジサイズは最大でも約 52 MB であり、閾値を大幅に下回るため、
通常は常に `topo_on_gpu = true`（HBM3 配置）が選択される。

`topo_on_gpu = false` パスを強制的にテストしたい場合は閾値を小さくする:

```bash
# 閾値 0.001 = 96 MB 以上のグラフで CPU 配置に切り替わる
./build_miyabi/brandes_runner gpu_readmostly \
    ../../data/snap/web-Google --topo-threshold 0.001
```

### Nsight が使えない環境での代替計測

`nsys` コマンドが存在しない場合、`run_all_experiments.sh` は自動的にプロファイルステップをスキップする。
代替として CUDA イベントによる簡易計測が `brandes_runner` の出力に含まれる実行時間（秒）を参照する。

L2 キャッシュヒット率などの詳細指標が必要な場合は `nv-nsight-cu-cli` を使う:

```bash
nv-nsight-cu-cli --metrics l2_read_hit_rate \
    ./build_miyabi/brandes_runner gpu_opt ../../data/56438_300801
```

### ビルドエラー（sm_90 以外の環境）

GH200 以外の GPU でビルドする場合は `DCMAKE_CUDA_ARCHITECTURES` を変更する:

```bash
# A100 (sm_80) の場合
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80

# V100 (sm_70) の場合
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70
```

HBM3 専用ダブルバッファ（Stage 1）の性能特性は GH200 以外では異なる可能性がある。

### cmake 失敗 / CUDA コンパイラが見つからない

ログインノードでは CUDA コンパイラが使えない。  
必ず `qsub -I` でインタラクティブジョブを取得してから実行する。  
再ビルド時は `rm -f build_miyabi/CMakeCache.txt` でキャッシュを削除する。
