# 改善提案レポート

**修士論文「Miyabi GH200 のアーキテクチャ特性を活用した Betweenness Centrality 計算の高速化」**

> 作成日: 2026-03-19  
> 対象リポジトリ: `m5291091/lab`  
> 目的: 論文の主張を実験で完全に証明するための改善提案

---

## 目次

1. [現状の問題点まとめ](#1-現状の問題点まとめ)
2. [改善提案1: 実装の再設計（公平な比較のための5段階アブレーション）](#2-改善提案1-実装の再設計公平な比較のための5段階アブレーション)
3. [改善提案2: 35%閾値を跨ぐ実験の追加](#3-改善提案2-35閾値を跨ぐ実験の追加)
4. [改善提案3: 合成大規模グラフの追加（35%超シナリオの実験）](#4-改善提案3-合成大規模グラフの追加35超シナリオの実験)
5. [改善提案4: Nsight による中間指標の計測](#5-改善提案4-nsight-による中間指標の計測)
6. [改善提案5: 統一実験スクリプトの設計](#6-改善提案5-統一実験スクリプトの設計)
7. [改善提案6: `brandes_gpu.cu` にダブルバッファを追加すべきか](#7-改善提案6-brandes_gpucuにダブルバッファを追加すべきか)
8. [改善提案7: 論文主張の再設計](#8-改善提案7-論文主張の再設計)
9. [改善後の論文強化比較表](#9-改善後の論文強化比較表)
10. [優先順位サマリ](#10-優先順位サマリ)

---

## 1. 現状の問題点まとめ

| # | 問題 | 影響 |
|---|------|------|
| P1 | **35%閾値の `false` パスが未実験** — 現在の全15グラフはトポロジ最大52MB、35%閾値（33.6GB）に一度も到達していない。`topo_on_gpu = false` コードパスは実行されていない。 | `brandes_gpu_opt.cu` の大規模グラフ対応機能が「論理的に実行できる」という主張にとどまり、実験的根拠がない |
| P2 | **96GB超グラフでの評価未実施** — HBM3専用（`brandes_gpu.cu`）は96GB超で動作不可。UVMの「大容量対応」という差別化が実験で示せていない。 | 論文の最大の差別化ポイントが未証明 |
| P3 | **実装間の「不公平な比較」** — `brandes_gpu.cu`（HBM3専用）には手法2（ダブルバッファ）がなく、`brandes_gpu_opt.cu`（手法1+手法2）との比較は「手法2なし vs 手法2あり」であり、手法1・手法2の効果が混在する。 | アブレーション分析の信頼性が低下 |
| P4 | **GPU_ReadMostly で一部グラフが未計測** — `random (32K)` グラフで `GPU_ReadMostly` の値が「—」（未計測）。 | アブレーション表の穴が審査委員の疑問を招く |
| P5 | **中間指標の欠如** — SetReadMostly がL2キャッシュヒット率を上げていることの直接証拠がない。Nsight Systemsのプロファイルタイムラインの活用が不十分。 | 「なぜ速くなるのか」のメカニズム説明が弱い |

---

## 2. 改善提案1: 実装の再設計（公平な比較のための5段階アブレーション）

### なぜ必要か

現在の4段階アブレーション（GPU / GPU_Managed / GPU_ReadMostly / GPU_Opt）では、「HBM3専用 vs GPU_Opt」の比較が「手法2なし vs 手法2あり」という不公平な状態になっている。これでは手法1の効果と手法2の効果を正確に分離できず、論文の主張の根拠が弱くなる。

### 何が変わるか

現在の4段階を以下の**5段階アブレーション**に再設計する：

```
Stage 0: brandes_gpu.cu          ← HBM3専用 + シングルストリーム（変更なし）
Stage 1: brandes_gpu_stream.cu   ← HBM3専用 + 2ストリームダブルバッファ（新規追加）
Stage 2: brandes_gpu_managed.cu  ← UVM素朴実装（変更なし）
Stage 3: brandes_gpu_readmostly.cu ← 手法1のみ（変更なし）
Stage 4: brandes_gpu_opt.cu      ← 手法1+手法2（変更なし）
```

各段階の比較が明確に分離される：

| 比較軸 | Stage A | Stage B | 測定する効果 |
|--------|---------|---------|------------|
| 手法2単体（HBM3環境） | Stage 0 | Stage 1 | ダブルバッファリングの純粋な効果 |
| HBM3専用+手法2 vs UVM+手法1+2 | Stage 1 | Stage 4 | UVMのオーバーヘッドがダブルバッファで補えるか |
| UVM導入コスト | Stage 0 | Stage 2 | NVLink-C2Cボトルネックの定量化 |
| 手法1（SetReadMostly）単体 | Stage 2 | Stage 3 | NVLink-C2Cボトルネック解消の効果 |
| 手法2（ダブルバッファ）単体（UVM環境） | Stage 3 | Stage 4 | UVM環境での初期化コスト削減効果 |

また、「GPU_Opt が HBM3専用+ダブルバッファを超えるか」という**新しい研究問い**が生まれ、GH200のUVM最適化の限界と可能性を論じる強い根拠になる。

### 実装難易度

**中**。`brandes_gpu.cu` の逐次初期化ループを `cudaMemsetAsync` に置き換え、2本のストリームで交互に処理する実装を `brandes_gpu_stream.cu` として追加する。コアロジック（BFS・バックワード）の変更は不要で、`brandes_gpu_opt.cu` の2ストリーム部分をHBM3専用の `cudaMalloc` 版に移植するだけで済む。

### 優先順位

**P1**（アブレーション分析の信頼性に直結する最重要改善）

---

## 3. 改善提案2: 35%閾値を跨ぐ実験の追加

### なぜ必要か

現在の全グラフではトポロジサイズが最大52MB（web-Google, roadNet-CAでもHBM3の0.05%程度）であり、`topo_on_gpu = false` コードパスが一度も実行されていない。このため「グラフサイズ適応型配置」という主張は「論理的には動くはず」という記述にとどまり、実験的根拠がない。審査委員から「実際に動作することを示せるか」と問われた際に答えられない状況にある。

### 何が変わるか

**閾値を環境変数または引数化**することで、既存グラフ（例: web-Google 44MBトポロジ）を用いて `topo_on_gpu = true` vs `false` の性能比較を実施できる：

```bash
# 実験1: 現在の設定（全グラフがtrue）
./brandes_runner gpu_opt graph.mtx --topo-threshold 0.35

# 実験2: 閾値を下げてfalseパスを強制的に実行
./brandes_runner gpu_opt graph.mtx --topo-threshold 0.001

# 実験3: 閾値境界付近の探索
./brandes_runner gpu_opt graph.mtx --topo-threshold 0.001
./brandes_runner gpu_opt graph.mtx --topo-threshold 0.01
./brandes_runner gpu_opt graph.mtx --topo-threshold 0.1
./brandes_runner gpu_opt graph.mtx --topo-threshold 0.35
./brandes_runner gpu_opt graph.mtx --topo-threshold 0.5
```

例えば `web-Google`（トポロジ44MB）で `topo_on_gpu = false` を強制するには、閾値を `44MB / 96GB ≒ 0.00046` 以下（例: 0.0004）に下げる必要がある。HBM3 96GB × 0.0004 = 38.4MB < 44MB となるため `topo_on_gpu = false` が強制的に実行される。実際には「グラフのトポロジが閾値×GPU総メモリを超える」ケースをシミュレートできる。

この実験により：
- `topo_on_gpu = false` パスの**動作検証**（バグがないことの確認）
- 同一グラフで `true` vs `false` の性能差の定量化（何倍遅くなるか）
- 閾値35%の妥当性の根拠データ取得

### 実装難易度

**低**。`brandes_gpu_opt.cu` の `topo_on_gpu = (topo_bytes < (size_t)(prop.totalGlobalMem * 0.35))` の `0.35` を `main.cpp` から渡せるコマンドライン引数に変更するだけで実現できる。

### 優先順位

**P2**（合成グラフ生成（改善提案3）の前段として先に実施すべき）

---

## 4. 改善提案3: 合成大規模グラフの追加（35%超シナリオの実験）

### なぜ必要か

改善提案2では「閾値を下げる」ことで `false` パスを強制実行できるが、これはあくまで「小グラフを大グラフに見せかける」代替手段である。本来の `false` パスは「グラフが本当に大きくて35GBを超える」シナリオを想定している。

また、「HBM3専用（`brandes_gpu.cu`）では動作不可、UVM版は動作する」という論文の最大の差別化ポイントを実験で示すには、96GBを超える実際の大規模グラフが必要。

### 何が変わるか

Barabási–AlbertモデルまたはRMATモデルで以下の合成グラフを生成し、実験を追加する：

| 合成グラフ | 頂点数 | 平均次数 | トポロジサイズ（概算） | `topo_on_gpu` |
|----------|--------|--------|----------------------|--------------|
| `synth_100M_d20` | 1億 | 20 | ~7.5 GB | `true`（35%以下） |
| `synth_500M_d20` | 5億 | 20 | ~37 GB | `false`（35%超） |
| `synth_1B_d20` | 10億 | 20 | ~74 GB | `false`（35%超） |

> **トポロジサイズの計算式**: `(n_nodes + 1) × 4B (R配列) + n_edges × 4B (C配列)` ≒ `n_nodes × (1 + 平均次数) × 4B`（無向グラフでは `n_edges` が既に両方向の辺数を含む）

これにより実現できる実験：

1. **`synth_100M_d20`（7.5GB）**:
   - `brandes_gpu.cu`（HBM3専用）は動作可能 → Stage 0〜4 全比較
   - `topo_on_gpu = true` パス（HBM3直接配置）の確認

2. **`synth_500M_d20`（37GB）**:
   - `brandes_gpu.cu`（HBM3専用）は動作可能（37GB + 動的データでギリギリ）
   - `topo_on_gpu = false` パス（CPU配置 + NVLink-C2C）の性能計測
   - Stage 3 vs Stage 4 でダブルバッファの効果が顕著になる想定

3. **`synth_1B_d20`（74GB）**:
   - `brandes_gpu.cu`（HBM3専用）は**動作不可**（OOMエラー）
   - `brandes_gpu_opt.cu`（UVM版）は動作可能 → 差別化の実証
   - 「大規模グラフでHBM3専用を超える唯一の方法」という強い主張が可能

### 合成グラフの生成方法

```python
# /tmp/gen_synth_graph.py (一例)
import networkx as nx

def gen_ba_graph(n, m, output_path):
    """Barabási–Albert モデルでグラフ生成し MTX 形式で保存"""
    G = nx.barabasi_albert_graph(n, m, seed=42)
    with open(output_path, 'w') as f:
        f.write(f"%%MatrixMarket matrix coordinate pattern symmetric\n")
        f.write(f"{n} {n} {G.number_of_edges()}\n")
        for u, v in G.edges():
            f.write(f"{u+1} {v+1}\n")
```

または `graph500` リファレンス実装の RMAT グラフ生成ツールを使用する（スケールフリー性が高く実グラフに近い特性）。

### 実装難易度

**中**（合成グラフの生成自体は容易だが、10億頂点グラフの生成には時間とストレージ（数百GB）が必要。Miyabi-G のストレージ容量を事前確認すること）。

### 優先順位

**P1**（論文の差別化ポイントを実験で示すため最重要）

---

## 5. 改善提案4: Nsight による中間指標の計測

### なぜ必要か

現在の論文は「`SetReadMostly` によりNVLink-C2CボトルネックをL2キャッシュで補う」という主張をしているが、L2キャッシュヒット率の直接計測がない。また、ダブルバッファリングの「memsetとBFSのオーバーラップ」が実際に起きているかの検証もない。これらの中間指標がないと「なぜ速くなるのか」のメカニズム説明が実行時間の比較だけに依存し、説得力が弱くなる。

### 何が変わるか

以下の指標を `run_profile.sh` に追加して自動取得する：

| 指標 | Nsight Compute メトリクス名 | 意味 | 証明する主張 |
|------|--------------------------|------|------------|
| **L2 hit rate（%）** | `l2_tex_hit_rate` | L2キャッシュヒット率 | `SetReadMostly` の効果の直接証拠 |
| **NVLink BW utilization（%）** | `nvlink_bandwidth_util` | NVLink帯域使用率 | NVLinkボトルネックの緩和を定量化 |
| **SM occupancy（%）** | `sm_active_cycles_avg` | SMの利用率 | 手法2による初期化コスト削減の証明 |
| **memset overlap time（ms）** | Nsight Systemsタイムライン | memsetとBFSのオーバーラップ時間 | ダブルバッファリングの有効性 |

### `run_profile.sh` への追記例

```bash
# 代表グラフ3つ（小・中・大）に対してプロファイル取得
PROFILE_GRAPHS=("benchmark_7000_41459" "56438_300801" "325557_3216152")

for GNAME in "${PROFILE_GRAPHS[@]}"; do
    # Nsight Compute: L2ヒット率・NVLink帯域・SM稼働率
    ncu --metrics l2_tex_hit_rate,nvlink_bandwidth_util,sm_active_cycles_avg \
        --csv \
        "${RUNNER}" gpu_opt "${DATA_DIR}/${GNAME}" \
        > "${RESULT_DIR}/ncu_${GNAME}.csv"

    # Nsight Systems: タイムラインでmemset-BFSオーバーラップを可視化
    nsys profile --stats=true \
        --output "${RESULT_DIR}/nsys_${GNAME}" \
        "${RUNNER}" gpu_opt "${DATA_DIR}/${GNAME}"
done
```

### 期待される証拠の整理

```
【主張】SetReadMostly がNVLink-C2Cボトルネックを解消する
  → 証拠: GPU_Managed の L2 hit rate < GPU_ReadMostly の L2 hit rate

【主張】ダブルバッファリングで初期化コストが隠蔽される
  → 証拠: Nsight Systemsタイムラインで memset と BFS カーネルが重なっている

【主張】手法1+2 で SM 稼働率が向上する
  → 証拠: GPU_Managed の SM occupancy < GPU_Opt の SM occupancy
```

### 実装難易度

**低**（既存の `run_profile.sh` への追記のみ。Nsight が Miyabi-G にインストール済みであることを確認）。

### 優先順位

**P2**（実験結果を補強する証拠として論文の説得力を大幅に向上させる）

---

## 6. 改善提案5: 統一実験スクリプトの設計

### なぜ必要か

現在のスクリプト構成（`run_baseline.sh`, `run_ablation.sh`, `run_profile.sh`, `measure_bandwidth.sh`）は個別に実行する設計であり、「全証拠データの取得漏れ」が発生しやすい。また、`random (32K)` での `GPU_ReadMostly` 未計測（問題P4）のように、実験スクリプトのカバレッジが不完全な箇所がある。

### 何が変わるか

「1回の実行で全証拠が揃う」統合スクリプト `run_all_experiments.sh` を作成する：

```
run_all_experiments.sh の構成
│
├── Step 1: メモリ帯域ベンチマーク
│   └── measure_bandwidth.sh
│       → HBM3 D→D, NVLink-C2C, Pinned H→D, D→H の帯域を記録
│
├── Step 2: 5段階アブレーション（全グラフ × Stage 0〜4）
│   └── run_ablation.sh（Stage 1: brandes_gpu_stream を追加）
│       → ablation_summary.tsv（全グラフ × 5実装の実行時間・GTEPS）
│
├── Step 3: 閾値感度実験（--topo-threshold パラメタスイープ）
│   └── run_threshold_sensitivity.sh（新規）
│       → 閾値 = 0.001, 0.01, 0.1, 0.35, 0.5 × 代表グラフ3つ
│       → threshold_sensitivity.tsv
│
├── Step 4: 合成大規模グラフ実験（新規）
│   └── run_synth_graphs.sh（新規）
│       → synth_100M（true/falseパス両方）
│       → synth_500M（falseパス）
│       → synth_1B（HBM3専用がOOM → UVM版のみ動作）
│       → synth_results.tsv
│
├── Step 5: Nsight プロファイル（代表グラフ3つ × 主要実装3つ）
│   └── run_profile.sh（拡張版）
│       → L2ヒット率、NVLink帯域、SM稼働率のCSV
│       → Nsight Systemsタイムライン（オーバーラップ確認）
│
└── Step 6: 全結果の自動集計・図生成
    └── analyze_all.py（拡張）
        → 全TSV/CSVを読み込みグラフ・表を自動生成
        → ablation_study.pdf（5段階版）
        → threshold_sensitivity.pdf（閾値感度）
        → synth_scaling.pdf（合成グラフのスケーリング）
```

### `run_all_experiments.sh` のスケルトン

```bash
#!/bin/bash
# run_all_experiments.sh
# 使用方法: bash run_all_experiments.sh
# 所要時間目安: 約6〜12時間（合成グラフ生成含む）

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_ROOT="${SCRIPT_DIR}/build_miyabi/results_all"
mkdir -p "${RESULT_ROOT}"

echo "[Step 1/6] メモリ帯域ベンチマーク"
bash "${SCRIPT_DIR}/scripts/measure_bandwidth.sh" \
    2>&1 | tee "${RESULT_ROOT}/step1_bandwidth.log"

echo "[Step 2/6] 5段階アブレーション（全グラフ）"
bash "${SCRIPT_DIR}/scripts/run_ablation.sh" \
    2>&1 | tee "${RESULT_ROOT}/step2_ablation.log"

echo "[Step 3/6] 閾値感度実験"
bash "${SCRIPT_DIR}/scripts/run_threshold_sensitivity.sh" \
    2>&1 | tee "${RESULT_ROOT}/step3_threshold.log"

echo "[Step 4/6] 合成大規模グラフ実験"
bash "${SCRIPT_DIR}/scripts/run_synth_graphs.sh" \
    2>&1 | tee "${RESULT_ROOT}/step4_synth.log"

echo "[Step 5/6] Nsight プロファイル"
bash "${SCRIPT_DIR}/scripts/run_profile.sh" \
    2>&1 | tee "${RESULT_ROOT}/step5_profile.log"

echo "[Step 6/6] 結果集計・図生成"
python3 "${SCRIPT_DIR}/analysis/analyze_all.py" \
    --result-dir "${RESULT_ROOT}" \
    --output-dir "${SCRIPT_DIR}/analysis/figures" \
    2>&1 | tee "${RESULT_ROOT}/step6_analyze.log"

echo "完了: 全証拠データが ${RESULT_ROOT} に出力されました"
```

### 実装難易度

**中**（各ステップのスクリプトは既存のものを流用・拡張できる。新規作成は `run_threshold_sensitivity.sh` と `run_synth_graphs.sh` の2ファイル）。

### 優先順位

**P3**（他の改善提案が揃ってから最後に統合する）

---

## 7. 改善提案6: `brandes_gpu.cu` にダブルバッファを追加すべきか

### 問題の背景

現在のアブレーション設計では：

- `brandes_gpu.cu`（Stage 0）: HBM3専用 + **シングルストリーム**
- `brandes_gpu_opt.cu`（Stage 4）: UVM + 手法1 + **手法2（ダブルバッファ）**

「Stage 0 vs Stage 4」の比較は「手法2なし vs 手法2あり」であり、手法1と手法2の効果が混在している。

### 追加すべき理由（Yes の根拠）

1. **公平な比較軸の確立**: Stage 0 vs Stage 1（HBM3専用のシングル vs ダブルバッファ）の比較により、「手法2単体の効果」をHBM3専用環境で純粋に測定できる。これはアブレーション分析の信頼性を根本的に高める。

2. **強い研究問いの生成**: 「UVM + 手法1 + 手法2（Stage 4）が HBM3専用 + ダブルバッファ（Stage 1）を超えるか」というのは非常に強い研究問いである。もしStage 4 ≥ Stage 1 であれば「GH200のUVM最適化によりHBM3直接アクセスと同等以上の性能を達成できる」という強い主張が可能になる。

3. **手法2の普遍性の実証**: ダブルバッファリングの有効性がUVM環境に限らず、HBM3専用環境でも有効であることを示すことで、手法2の汎用性を論じることができる。

4. **現在の論文の弱点を補完**: 現状では「手法2はUVM環境で初期化コストを隠蔽する」という主張しかできないが、「HBM3専用環境でも同様の効果が得られ、その効果量は（HBM3専用: X%、UVM: Y%）である」という比較が可能になる。

### 追加しない理由（No の根拠）

1. **現在のアブレーション設計の完結性**: Stage 0（HBM3専用）→ Stage 2（UVM素朴）→ Stage 3（手法1）→ Stage 4（手法1+2）という流れは「UVM最適化の段階的改善」として完結しており、HBM3専用に手法2を追加するとこの物語の流れが複雑になる。

2. **比較軸の増加**: Stage 1を追加すると比較の組み合わせが増え、論文のメッセージが分散するリスクがある。

3. **HBM3専用は「超えるべきベースライン」として機能する**: 現在の設計でも「GPU_ReadMostly ≈ GPU（ほぼ同等）」「GPU_Opt > GPU（手法1+2でわずかに上回る）」という主張は成立する。HBM3専用を手法2なしのベースラインとして位置づけることで「同じ手法2がないのにUVM版がほぼ同等の性能」という主張が強調できる見方もある。

### 最終推奨: **追加すべき（Yes）**

理由: 論文の審査において「なぜHBM3専用に手法2を適用しなかったのか」という当然の疑問が生じる。この疑問に対して「追加したがStage 4がStage 1を上回った（または同等だった）ため、UVM最適化の有効性が実証された」と答える方が、「追加していないため比較できない」と答えるより遥かに説得力がある。

Stage 1の実装は技術的に容易（`brandes_gpu_opt.cu` の2ストリーム部分を `cudaMalloc` 版に移植するだけ）であり、追加コストに対してリターンが大きい。

---

## 8. 改善提案7: 論文主張の再設計

### 現在の論文の主張（改善前）

```
1. GH200のUVM環境ではNVLink-C2Cが9.6倍の帯域差によりBC計算に
   最大5.1倍の性能低下をもたらす（実験済み）

2. SetReadMostly + グラフサイズ適応型配置（手法1）により、
   GPU_ReadMostly ≈ GPU（HBM3専用）の性能を達成（実験済み）

3. cudaMemsetAsync + 2ストリームダブルバッファ（手法2）により、
   GPU_Opt が GPU（HBM3専用）を一部グラフで上回る（実験済み）

4. topo_on_gpu = false パスにより96GB超グラフにも対応できる
   （論理的に実行可能だが、実験未実施）
```

### 改善後の論文の主張

```
1. GH200のUVM環境ではNVLink-C2Cが9.6倍の帯域差によりBC計算に
   最大5.1倍の性能低下をもたらすことを実証した（既存）

2. SetReadMostly + グラフサイズ適応型配置により、96GB以下では
   HBM3専用+ダブルバッファと同等以上の性能を達成した（新規:
   Stage 1追加後に実証可能）

3. topo_on_gpu = false パスにより、96GB超（X億頂点、Yエッジ）の
   グラフでも正しく動作することを実験で実証した（新規:
   合成グラフ追加後に実証可能）

4. 手法2（ダブルバッファ）はHBM3専用・UVM両環境で有効であり、
   UVM環境での効果（X%削減）はHBM3専用環境（Y%削減）と
   同等/以上である（新規: Stage 1追加後に実証可能）
```

### 主張の強化ポイント

| 現在の主張 | 改善後の主張 | 強化の根拠 |
|-----------|------------|-----------|
| 「96GB超に対応できる（論理的）」 | 「X億頂点でも動作を実証した（実験的）」 | 合成グラフ（改善提案3）で実証 |
| 「手法1+2でHBM3専用をわずかに上回る」 | 「手法1+2でHBM3専用+手法2と同等以上」 | Stage 1追加（改善提案1）で比較 |
| 「手法2が有効」（UVM環境のみ） | 「手法2はHBM3専用・UVM両環境で有効（効果量も比較）」 | Stage 0 vs 1, Stage 3 vs 4の比較 |
| 「SetReadMostlyがL2ヒット率を改善（推定）」 | 「SetReadMostlyがL2ヒット率をZ%向上（Nsight計測）」 | Nsight計測（改善提案4）で直接証明 |

---

## 9. 改善後の論文強化比較表

| 評価軸 | 改善前 | 改善後 | 改善に必要な作業 |
|--------|--------|--------|----------------|
| **アブレーション設計の公平性** | ❌ 不公平（手法2の有無が混在） | ✅ 公平（5段階で効果を完全分離） | Stage 1 (`brandes_gpu_stream.cu`) 追加 |
| **`topo_on_gpu = false` の検証** | ❌ 実験未実施（論理的のみ） | ✅ 実験で動作確認 | 閾値引数化 + 合成グラフ追加 |
| **96GB超グラフでの差別化実証** | ❌ 未実証 | ✅ synth_1B で実証（HBM3専用OOM, UVM版動作） | 合成グラフ生成 + 実験 |
| **中間指標の提示** | ❌ 実行時間のみ | ✅ L2ヒット率・NVLink帯域・SM稼働率 | Nsight プロファイル追加 |
| **GPU_ReadMostly の計測完全性** | ❌ random(32K)が未計測 | ✅ 全グラフ計測 | `run_ablation.sh` の修正 |
| **手法2の普遍性** | ❌ UVM環境のみ示す | ✅ HBM3専用・UVM両環境で示す | Stage 1追加 + 比較実験 |
| **「1回の実行で全証拠が揃う」** | ❌ 複数スクリプトを個別実行 | ✅ `run_all_experiments.sh` で統合 | 統合スクリプト作成 |

---

## 10. 優先順位サマリ

| 優先度 | 改善提案 | 実装難易度 | 論文への影響 | 推定工数 |
|--------|---------|----------|------------|--------|
| **P1** | 改善提案1: Stage 1 (`brandes_gpu_stream.cu`) の追加 | 中 | ◎ アブレーション全体の信頼性向上 | 2〜3日 |
| **P1** | 改善提案3: 合成大規模グラフの追加（synth_1B） | 中 | ◎ 論文最大の差別化ポイントを実証 | 2〜4日（グラフ生成含む） |
| **P2** | 改善提案2: 35%閾値の引数化 + 閾値感度実験 | 低 | ○ `false` パスの動作検証 | 0.5日 |
| **P2** | 改善提案4: Nsight 中間指標の計測 | 低 | ○ メカニズム説明の強化 | 1日 |
| **P3** | 改善提案5: 統一実験スクリプト `run_all_experiments.sh` | 中 | △ 再現性・完全性の向上 | 1日 |
| **P3** | 改善提案7: 論文主張の再設計 | — | ◎ 全改善完了後に自動的に強化 | 0.5日（文章修正） |

### 推奨実施順序

```
Week 1:
  Day 1-2: 改善提案2（閾値引数化） + 既存グラフでの false パス確認
  Day 3-4: 改善提案1（Stage 1: brandes_gpu_stream.cu の実装）
  Day 5:   改善提案4（Nsight プロファイル追加）

Week 2:
  Day 1-3: 改善提案3（合成グラフ生成 + synth_100M / 500M / 1B 実験）
  Day 4:   改善提案5（run_all_experiments.sh 統合）
  Day 5:   改善提案7（論文主張の書き直し）
```

---

## 付録: `random (32K)` グラフの GPU_ReadMostly 未計測について

`research/analysis/tables/exec_time_table.tsv` を確認すると、`random (32K)` グラフの `GPU_ReadMostly` 列が空欄（未計測）になっている。この穴を埋めるには `run_ablation.sh` または `run_baseline.sh` でこのグラフを `gpu_readmostly` 実装で実行するよう追加するだけで解決できる（実装難易度: **低**）。これは上記の優先順位P1〜P3の実施とは独立して今すぐ修正できる。

```bash
# 追加するだけでよい行（run_baseline.sh の medium グラフリストに追加）
"${DATA_DIR}/random"  # 既存グラフファイルのパス確認が必要
```

この修正はアブレーション表の完全性を確保し、審査委員への印象を改善する小さいが重要な修正である。
