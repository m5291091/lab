# thesis_bc_project — GH200 向け厳密媒介中心性 (Exact Betweenness Centrality) ベンチマーク

NVIDIA GH200 (Grace Hopper) / Miyabi-G 環境上で、複数の**厳密 BC (Exact Betweenness
Centrality)** 実装を共通 CLI から計測・比較するための**自己完結型**プロジェクトです。
本リポジトリ (`thesis_bc_project/`) 単体でビルド・実行でき、他ディレクトリへの依存は
ありません（cuGraph は `third_party/cugraph/` に同梱）。

- **提案手法** (`proposed`): GH200 特化 CUDA カーネル (Unified Memory / 純 GPU メモリ / チャンク)
- **ベースライン** (`baseline`): CPU 逐次 / OpenMP / 素朴 GPU / RAPIDS cuGraph / Galliot path-merging
- **実験ランナー** (`experiments`): 総合ベンチ / アブレーション / PathMerge バッチ探索

---

## Repository Access and Third-Party Code Notice

本repositoryは、修士研究の記録、再現性確認、および研究審査のために限定共有するprivate repositoryです。
一般公開および第三者への再配布を目的としていません。
本研究で評価したPathMergeは、原著論文著者による公式実装ではなく、第三者実装を基にしたadapter派生実装です。上流実装には明示的なライセンスが確認できていないため、当該ソースコードの公開再配布権は未確認です。
本論文の性能結果は、評価に使用したこの第三者実装との比較に限定され、PathMergeアルゴリズム一般または原著者の公式実装に対する性能優位を主張するものではありません。

## ディレクトリ構成

```
thesis_bc_project/
├── CMakeLists.txt          # トップレベルビルド定義
├── include/                # ヘッダ (ホスト API + カーネルテンプレート)
│   ├── core/               #   graph.hpp(データロード), common.hpp, runner.hpp
│   ├── proposed/           #   brandes_gpu.hpp, ablation_config.hpp, brandes_kernels.cuh(カーネル)
│   └── baseline/           #   brandes_baseline.hpp, galliot 系ヘッダ
├── src/                    # 実装
│   ├── core/               #   graph.cpp(CSR読込), runner.cpp(計測/レポート)
│   ├── proposed/           #   host_um / host_pure / host_chunked / host_ablation (.cu)
│   └── baseline/           #   sequential / omp / gpu_unopt / cugraph_bc / galliot / pathmerge
├── experiments/            # 実行ファイル (ランナー)
│   ├── run_benchmark.cu        # 全実装ベンチ + 正確性検証 (--dump-bc)
│   ├── run_ablation.cu         # アブレーション実験
│   └── run_pathmerge_sweep.cu  # PathMerge バッチサイズ探索
├── third_party/cugraph/    # 同梱 cuGraph サブセット (BC 関連)
├── cugraph_bc_mini/        # BC 専用ミニ cuGraph ビルド (libcugraph_bc_mini.a 生成)
├── scripts/                # ビルド / ベンチ / 解析スクリプト (PBS ジョブ含む)
├── tools/                  # 帯域計測 / グラフ生成 / SNAP ダウンローダ
└── data/                   # 小〜中規模グラフを同梱 (snap は tools で取得)
```

**3 層分離 (HPC/CUDA ベストプラクティス)**
- データロード部: `src/core/graph.cpp` (+ `include/core/graph.hpp`)
- ホスト側制御部: `src/{proposed,baseline}/*.cu` (メモリ管理・カーネル起動・計測)
- CUDA カーネル部: `include/proposed/brandes_kernels.cuh` (テンプレート化 device/global 関数)

すべての実装は共通シグネチャ `std::vector<double>(Graph&)` を持ち、`run_brandes()` が
統一的に計測します。

---

## 実行環境 (Miyabi-G)

ビルド・実行は必ず **GPU 計算ノード**で行ってください（ログインノード不可）。
インタラクティブノードは以下で確保します:

```bash
qsub -I -q interact-g -l select=1:ncpus=72 -l walltime=02:00:00 -W group_list=gj17
```

> 補足: 構文チェック（コンパイルのみ）はログインノードでも可能ですが、実行 (GPU) と
> `cugraph_bc_mini` の初回ビルド（RMM/RAFT/CCCL 等を CPM でネットワーク取得）は
> 計算ノードで行ってください。

---

## テスト環境での実行手順

### 1. ビルド

GPU ノードを確保したら、プロジェクトルートで統合ビルドスクリプトを実行します。
CMake (3.30.4以上) が必要ですが、システムの既定バージョンが古いため、初回は最新の CMake を導入します。

**【推奨】uv を使用する場合（高速・環境を汚さない）**
```bash
cd thesis_bc_project

# 1. uv のインストール (未導入の場合のみ)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. uv を使って CMake をインストール (独立したツールとして導入)
# ※既に pip 等で cmake が入っていてエラーになる場合は --force を付けて上書きしてください
uv tool install cmake

# 3. ビルド実行
bash scripts/build_miyabi_interactive.sh
```

**pip を使用する場合（従来の自動インストール）**
```bash
cd thesis_bc_project

# 初回ビルド (pip で新しい CMake を ~/.local/bin 等に導入)
AUTO_INSTALL_CMAKE=1 bash scripts/build_miyabi_interactive.sh
```

**2回目以降のビルド（共通）**
```bash
# 増分ビルド
bash scripts/build_miyabi_interactive.sh

# CMake キャッシュが壊れた場合
CLEAN_CACHE=1 bash scripts/build_miyabi_interactive.sh
```

Stage 1 で `cugraph_bc_mini/build/libcugraph_bc_mini.a`（初回のみ約 10 分）、
Stage 2 で `build_miyabi/` 以下に各ランナーを生成します。

生成物:
- `build_miyabi/run_benchmark` — 全実装ベンチ / 正確性検証
- `build_miyabi/run_ablation` — アブレーション実験
- `build_miyabi/run_pathmerge_sweep` — PathMerge バッチサイズ探索
- `build_miyabi/bandwidth_benchmark` — 帯域計測

> `run_ablation` と `run_pathmerge_sweep` は cuGraph に依存しません。これらだけを
> 素早くビルドしたい場合は、対応する PBS スクリプト（後述）が対象ターゲットのみを
> ビルドするため `cugraph_bc_mini` のビルドを省略できます。

### 2. スモークテスト

```bash
cd build_miyabi

# 最小グラフで動作確認
./run_benchmark sequential ../data/chain_200
./run_benchmark gpu_opt    ../data/benchmark_7000_41459

# 全実装を一括実行
./run_benchmark all        ../data/benchmark_7000_41459
```

stdout はタブ区切り 1 行 `Impl<TAB>Graph<TAB>Time_sec<TAB>GTEPS`、
フェーズ計測や最大 BC は stderr に出力されます。

### 3. 正確性検証 (実装間の BC 値 diff)

`--dump-bc` で全頂点の BC 値を stdout にダンプし、実装間で比較します。

```bash
cd build_miyabi
./run_benchmark gpu_opt    ../data/benchmark_7000_41459 --dump-bc > bc_gpu.txt
./run_benchmark cugraph_bc ../data/benchmark_7000_41459 --dump-bc > bc_cugraph.txt
diff bc_gpu.txt bc_cugraph.txt   # 一致すれば正しい
```

小規模3グラフの独立参照 full-vector 検証は `result/correctness/small_full_vector/`、GH200 メモリ経路(UM/Pure/Chunked)・大バッチの正確性/診断アーカイブ（325557限定, 同一batch mismatch=0 は非byte一致, stress差は未解決, formal `CORE_FAIL` 保存）は `result/correctness/memory_paths/` を参照。後者の分析 TSV/Markdown は `scripts/analyze_memory_correctness.py` で raw vector から再生成できる。

### 4. アブレーション実験（提案手法 3 工夫の寄与測定）

3 つの工夫を**コンパイル時テンプレート**で個別に ON/OFF し、寄与を測定します
（カーネル内 if ではないためブランチダイバージェンスのオーバーヘッドなし）:

| フラグ | 工夫 |
|--------|------|
| `H` | ① ハイブリッド (トップダウン/ボトムアップ方向最適化) BFS |
| `W` | ② Warp 協調による依存関係蓄積 (shfl 還元) |
| `A` | ③ 2 ストリーム非同期初期化 (ダブルバッファリング) |

```bash
cd build_miyabi
# 8 構成すべてを実行 (フル要因ablation)
./run_ablation ../data/benchmark_7000_41459 all

# 個別構成 (例: ハイブリッドのみ ON)
./run_ablation ../data/benchmark_7000_41459 H1W0A0
./run_ablation ../data/benchmark_7000_41459 baseline   # = H0W0A0 (全 OFF)
./run_ablation ../data/benchmark_7000_41459 full       # = H1W1A1 (全 ON)
```

stdout の各行は `Ablation_H<b>_W<b>_A<b><TAB>Graph<TAB>Time_sec<TAB>GTEPS`。
`baseline` (000) と各単一工夫 (100/010/001)、`full` (111) を比べることで
各工夫の短縮効果が分かります。

> **注: 3 工夫の寄与はグラフ依存**。とくに **W (Warp 協調蓄積) の効果はグラフ構造に依存**し、
> 高次数・ハブありグラフでは有効だが、低次数 (道路網など) では中立〜わずかに不利になり得る。
> 「H・W・A の 3 工夫すべてが全グラフで常に高速化する」とは限らないため、各工夫の寄与は
> グラフごとに個別 ablation で確認すること。

### 5. PathMerge バッチサイズ探索

ベースライン PathMerge (Galliot) のバッチサイズ（同時処理ソース数）を可変にし、
最適サイズを探索します。現行実装は `int2` フロンティア + per-source 配列方式のため
**旧来の 64 上限はなく**、HBM3 メモリの許す限り任意サイズを測定できます
（超過分は自動的にクランプ）。

```bash
cd build_miyabi
# 既定リストを掃引 (1,2,4,8,...,256)
./run_pathmerge_sweep ../data/benchmark_7000_41459

# 任意のリストを指定
./run_pathmerge_sweep ../data/benchmark_7000_41459 32,64,128,256,512
```

各行は `PathMerge_b<N><TAB>Graph<TAB>Time_sec<TAB>GTEPS`、最後に最速バッチサイズを
stderr に報告します。

---

## 本番ジョブの実装方法 (PBS)

本番実験は `scripts/` 以下の PBS ジョブを `qsub` で投入します
（`#PBS -q regular-g`, `group_list=gj17`）。プロジェクトルートから投入してください。

```bash
cd thesis_bc_project

# --- 総合ベンチマーク (規模別・並列投入可) ---
qsub scripts/run_benchmark_small.sh
qsub scripts/run_benchmark_medium.sh
qsub scripts/run_benchmark_large.sh

# --- アブレーション実験 (step 4) ---
qsub scripts/run_ablation.sh
#   例: 特定グラフ/試行回数を指定
#   qsub -v GRAPHS_STR="benchmark_7000_41459 56438_300801",TRIALS=3 scripts/run_ablation.sh

# --- PathMerge バッチサイズ探索 (step 5) ---
qsub scripts/run_pathmerge_sweep.sh
#   例: バッチリストを指定
#   qsub -v BATCH_LIST="32,64,128,256,512" scripts/run_pathmerge_sweep.sh

# --- UM オーバーサブスクリプション実験 ---
qsub scripts/run_um_oversubscribe_experiment.sh
qsub scripts/run_um_oversubscribe_gpu_opt.sh

# --- 投入前にコマンドだけ確認 (ビルド/実行しない) ---
DRY_RUN=1 SKIP_BUILD=1 bash scripts/run_benchmark_small.sh
DRY_RUN=1 SKIP_BUILD=1 bash scripts/run_ablation.sh
```

各ジョブは `build_miyabi/result_*` にタイムスタンプ付きの結果ディレクトリを作成し、
`results.tsv` 等を出力します。PBS の stdout/stderr は投入ディレクトリに
`bc_*.oNNNNNN` として保存されます。

**PBS 操作**: `qstat -u $USER`（状態）, `qstat -f <JOB_ID>`（詳細）, `qdel <JOB_ID>`（取消）。

---

## 大規模グラフ (SNAP) の取得

小〜中規模グラフは `data/` に同梱済みです。大規模な SNAP グラフはリポジトリに
含めず、以下で取得します（`data/snap/` は `.gitignore` 済み）。

```bash
bash tools/download_snap_graphs.sh
```

---

## 解析・図表生成

ジョブ完了後、統計解析と論文用図表を生成します（scipy + matplotlib が必要）。

**【推奨】uv を使用する場合（仮想環境の事前構築不要で直接実行）**
```bash
uv run --with scipy --with matplotlib python scripts/statistical_analysis.py \
    --results build_miyabi/result_benchmark_*/results.tsv \
    --phases  build_miyabi/result_benchmark_*/phase_timing.log \
    --oversubscribe build_miyabi/result_um_oversubscribe/oversubscribe_results.tsv \
    --outdir ./thesis_figures
```

**従来の方法（システム Python や pip の仮想環境を使用）**
```bash
python3 scripts/statistical_analysis.py \
    --results build_miyabi/result_benchmark_*/results.tsv \
    --phases  build_miyabi/result_benchmark_*/phase_timing.log \
    --oversubscribe build_miyabi/result_um_oversubscribe/oversubscribe_results.tsv \
    --outdir ./thesis_figures
```

---

## 実行時チューニング用環境変数

| 環境変数 | 対象 | 説明 |
|----------|------|------|
| `BC_BATCH_OVERRIDE` | `gpu_opt` / `gpu_opt_pure` / ablation | 1 ストリームあたりのバッチサイズを上書き |
| `PATHMERGE_BC_BATCH_SIZE` | `pathmerge_bc` | PathMerge のバッチサイズ (既定 64, 上限なし) |
| `CUGRAPH_BC_MAX_SOURCES_PER_BATCH` | `cugraph_bc` | cuGraph のバッチサイズ |
| `JOBS` | ビルド | 並列コンパイルジョブ数 (既定 8) |
| `SKIP_BUILD` | PBS ジョブ | 1 でビルドをスキップ |
| `DRY_RUN` | PBS ジョブ | 1 でコマンド表示のみ |

---

## グラフデータ形式 (CSR テキスト)

`data/` のグラフは 3 行の CSR テキストです（`Graph::readGraph()` が読込）:

1. `n_nodes n_edges`
2. `ptr[0..n_nodes]`         （CSR オフセット, 長さ n_nodes+1）
3. `adj[0..2*n_edges-1]`     （隣接配列, 無向のため有向 nnz = 2*n_edges）

すべての実装は**無向・非重み**グラフの BC を出力し、無向グラフの二重計上を
補正するため最終 BC を 1/2 しています。
