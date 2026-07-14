# Miyabi-G ベンチマーク・実験 実行ガイド

本ディレクトリに配置されたスクリプトを利用して、論文に必要な全実験の実施、および統計検定やグラフの生成を自動で行うことができます。

## 1. 必要なコマンドと実行手順（本番用）

本番実験を行うために必要なコマンドは以下の通りです。
これらを順番に（あるいは同時に）ジョブ投入し、すべての実行が完了したのちに Python スクリプトで集計・図表化を行います。

```bash
# 1. 作業ディレクトリへ移動
cd /work/gj17/j17000/m5291091/lab/thesis_bc_project

# 2. 各規模のグラフに対するベンチマークのジョブを投入
# （並列に投入可能です。PBSが空きリソースに応じて順次実行します）
qsub scripts/run_benchmark_small.sh
qsub scripts/run_benchmark_medium.sh
qsub scripts/run_benchmark_large.sh

# gpu_opt / gpu_opt_pure_chunked を各規模で再計測したい場合
qsub scripts/run_benchmark_small_gpu_opt_compare.sh
qsub scripts/run_benchmark_medium_gpu_opt_compare.sh
qsub scripts/run_benchmark_large_gpu_opt_compare.sh

# 3. UMのメモリ・オーバーサブスクライブ性能評価（容量vs性能）のジョブ投入
qsub scripts/run_um_oversubscribe_experiment.sh

# 3'. gpu_opt / gpu_opt_pure_chunked を再計測したい場合（HBM3 streaming 比較用）
qsub scripts/run_um_oversubscribe_gpu_opt.sh

# 3''. gpu_opt_pure_chunked だけを再計測したい場合
qsub scripts/run_um_oversubscribe_gpu_opt_pure_chunked.sh

# 4. 全ジョブの完了を待機 (qstat -u $USER などで確認)
# ...

# 5. 統計検定と論文用図表の生成
# ⚠️ 注意: SciPy と Matplotlib がインストールされた Python 環境を有効化（アクティベート）してください。
python3 scripts/statistical_analysis.py \
    --results build_miyabi/result_benchmark_*/results.tsv \
    --phases build_miyabi/result_benchmark_*/phase_timing.log \
    --oversubscribe build_miyabi/result_um_oversubscribe/oversubscribe_results.tsv \
    --outdir ./thesis_figures
```

> **Note:**
> 上記の `python3 scripts/statistical_analysis.py ...` のコマンド内にある `build_miyabi/result_benchmark_*/results.tsv` のようなアスタリスク（`*`）を用いたワイルドカード指定により、`small`, `medium`, `large` の各ジョブが出力した複数の結果ディレクトリから自動的にすべての結果を収集し、一つの表やグラフに統合して出力します。

## 2. 生成されるデータ (thesis_figures 配下)

`statistical_analysis.py` の実行が成功すると、指定した `--outdir` (デフォルトでは `./thesis_figures`) 内に以下のファイルが生成されます。
これらはそのまま論文の図表として利用可能です。

1. **`statistical_test.md`**
   各グラフにおける `gpu_opt` (UM版) と `gpu_opt_pure` (手動メモリ管理版) の実行時間の 平均 ± 標準偏差 と、**Wilcoxon符号順位検定のp値** を出力します。email-EuAll は p=0.125 で、**n=5 では有意差を検出できませんでした**（これは同等性の証明ではなく、この試行数では差を検出できなかったという意味です。roadNet は n=3 のため検定不可）。UM オーバーヘッドが小さいことの傍証にはなりますが、統計的同等性を主張するものではありません。
2. **`phase_breakdown.pdf`**
   実行時間のうち **BFS / Backward の計測成分** を積み上げ棒グラフにしたものです。現行の UM 版ログは BFS/Backward (と Prefetch) を計測する一方、H2D/D2H 転送や初期化は個別計測していないため、**end-to-end 総時間との差は `Other (未計測)` として明示**しています。したがって本図は転送込みの完全な end-to-end 内訳ではなく、カーネル計測成分＋未計測分の内訳です。
3. **`batch_scalability.pdf`**
   中規模グラフ (`325557_3216152` など) を用いて `BC_BATCH_OVERRIDE` を増加させていった場合の実行時間をプロットした折れ線グラフです。HBM3 の上限（約 96GB）付近で `pure` 版がクラッシュ（OOM）する一方で、`UM` 版が LPDDR5X にスピルしながら動作を継続する様子を示します。

## 3. ジョブ管理の Tips

* **状態確認**: `qstat -u $USER`
* **詳細確認**: `qstat -f <JOB_ID>`
* **キャンセル**: `qdel <JOB_ID>`
* 出力された標準出力/エラーは、ジョブ投入したディレクトリに `bc_bench_small.oXXXXXX` のような形式で保存されます。実行時エラーが起きた場合はこのファイルを確認してください。

## 4. アブレーション寄与測定と BFS カーネル選択の寄与測定

提案手法 `gpu_opt` の各最適化の寄与を、査読に耐える形で定量化するためのワークフローです。
2 つの実験に分かれます:

* **アブレーション**: 3 工夫（H=ハイブリッド BFS / W=Warp 協調蓄積 / A=2 ストリーム非同期初期化）を
  コンパイル時テンプレートで ON/OFF する 8 構成フル要因実験（`run_ablation`）。
* **BFS カーネル選択 2×2**: 「適切なカーネル選択」（shared-frontier vs block-per-source）はアブレーションの枠外
  （`run_ablation` の BFS は常に block-per-source 固定）。`gpu_opt` は `avg_deg < 5` で実行時にカーネルを選ぶため、
  環境変数 `BC_FORCE_BFS_KERNEL`（`shared` / `block` / `auto`）で強制し、閾値をまたぐグラフで
  「正しい選択 vs 誤った選択」の速度差＝選択機構の寄与を測る（`run_kernel_selection`）。

### 4.1 アブレーション実験

```bash
cd /work/gj17/j17000/m5291091/lab/thesis_bc_project

# 中〜高次数（既定4グラフ, 5試行, 公平性のためバッチ固定）
qsub -v 'GRAPHS_STR=benchmark_7000_41459 benchmark_11023_62184 56438_300801 325557_3216152,TRIALS=5,BC_BATCH_OVERRIDE=512' scripts/run_ablation.sh

# 低次数（別ジョブ。roadNet 系は 1 回が長いので TRIALS を絞り walltime を延長した別ジョブに）
qsub -v 'GRAPHS_STR=snap/email-EuAll,TRIALS=3,BC_BATCH_OVERRIDE=512' scripts/run_ablation.sh
```

ジョブ完了時に `summarize_ablation.py` が自動実行され、結果ディレクトリ
`build_miyabi/result_ablation_<TS>/` に以下が生成されます（手動実行も可）:

```bash
python3 scripts/summarize_ablation.py \
    build_miyabi/result_ablation_<TS>/ablation_results.tsv \
    build_miyabi/result_ablation_<TS>
```

* `ablation_summary.md` — 中央値の生値表、**単独寄与 (add-one)** `T(H0W0A0)/T(単独ON)`、
  **除外寄与 (leave-one-out)** `T(full-1)/T(H1W1A1)`、主効果、**交互作用チェック**（両者の乖離を検出）、
  および `ablation.log` からの **フェーズ帰属**（H→BFS / W→Backward / A→wall−(BFS+Backward)）。
* `ablation_contributions.tsv` — 機械可読な寄与テーブル。

> **論文での断り書き**: 低次数グラフのアブレーションは「block-per-source BFS 内での H/W/A 寄与」であり、
> 本番 `gpu_opt` のカーネル構成（shared 選択）とは異なります。この差分こそ 4.2 の 2×2 実験で埋めます。

### 4.2 BFS カーネル選択 2×2

```bash
cd /work/gj17/j17000/m5291091/lab/thesis_bc_project

# 閾値 avg_deg=5 をまたぐ組（email-EuAll≈2.75 が shared 側, benchmark_7000≈11.85 / 56438≈10.66 が block 側）
qsub -v 'GRAPHS_STR=snap/email-EuAll benchmark_7000_41459 56438_300801,TRIALS=5' scripts/run_kernel_selection.sh

# roadNet 系を加える場合は 1 回が長いので別ジョブ（walltime 延長, TRIALS を絞る）
qsub -v 'GRAPHS_STR=snap/roadNet-PA,TRIALS=3,TIMEOUT_SEC=18000' scripts/run_kernel_selection.sh
```

各グラフに対し `BC_FORCE_BFS_KERNEL=shared` と `=block` を強制計測します
（`BC_BATCH_OVERRIDE` は既定 512 で両カーネル統一）。ジョブ完了時に `summarize_kernel_selection.py` が
自動実行され、`build_miyabi/result_kernel_selection_<TS>/` に以下が生成されます:

```bash
python3 scripts/summarize_kernel_selection.py \
    build_miyabi/result_kernel_selection_<TS>/kernel_selection_results.tsv \
    build_miyabi/result_kernel_selection_<TS>
```

* `kernel_selection_summary.md` — グラフごとの shared/block 中央値時間、`avg_deg<5` ヒューリスティクスの選択、
  「速い側を選べているか」の正誤、**選択機構の寄与**（誤選択時に失う倍率＝遅い側/速い側）、
  および shared/block の Max BC 一致（正確性サニティ）。
* `kernel_selection_contributions.tsv` — 機械可読テーブル。

> **注意**: `run_kernel_selection.sh` は `gpu_opt`（`run_benchmark`）を使うため cuGraph mini ライブラリに依存します。
> 初回は `bash scripts/build_miyabi_interactive.sh` で Stage1+2 をビルドしてください（`SKIP_BUILD=1` でスキップ可）。

### 4.3 正確性の確認（推奨・小グラフで 1 回）

```bash
cd build_miyabi
# 8 構成の BC 値一致（小グラフは全 dump を diff, H/W で演算順が変わる浮動小数微差もここで確認）
./run_ablation ../data/benchmark_7000_41459 baseline --dump-bc > /tmp/bc_base.txt
./run_ablation ../data/benchmark_7000_41459 full     --dump-bc > /tmp/bc_full.txt
diff /tmp/bc_base.txt /tmp/bc_full.txt
```

BFS カーネル選択（4.2）の shared/block も、強制計測の結果ディレクトリに残る Max BC は
`%0.2lf`（小数 2 桁）の粗い比較なので、厳密な一致確認を小グラフで 1 回だけ行っておくと万全です:

```bash
cd build_miyabi
# shared 強制 vs block 強制 の全 dump を diff（2 桁の Max BC では拾えない微差もここで確認）
BC_FORCE_BFS_KERNEL=shared ./run_benchmark gpu_opt ../data/benchmark_7000_41459 --dump-bc > /tmp/bc_shared.txt
BC_FORCE_BFS_KERNEL=block  ./run_benchmark gpu_opt ../data/benchmark_7000_41459 --dump-bc > /tmp/bc_block.txt
diff /tmp/bc_shared.txt /tmp/bc_block.txt
```
