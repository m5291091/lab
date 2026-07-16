# Master's Thesis Writing Plan

## 1. Purpose

本計画は、修士論文

> **Design and Evaluation of a Batch-Based GPU Execution Framework for Betweenness Centrality on GH200**

を執筆するための章構成、各章の目的、節構成、使用する図表、根拠ファイル、執筆順序および検証規則を定める。

論文の最終版は英語とする。ただし、初稿は`writing/japanese/`以下に日本語で作成し、内容確定後に同一構成で`writing/english/`へ英訳する。

タイトル、Chapter名、Section名、Figure caption、Table caption、図表内部の文字は英語とする。`writing/japanese/`内の本文のみ日本語で記述する。

---

## 2. Working Title

### English Title

**Design and Evaluation of a Batch-Based GPU Execution Framework for Betweenness Centrality on GH200**

### Japanese Reference Title

GH200上の媒介中心性計算に向けたバッチ型GPU実行基盤の設計と評価

日本語タイトルは内容確認用であり、論文タイトルページには大学指定に従って英語タイトルを使用する。

---

## 3. Directory Structure

次のファイルを作成する。

```text
thesis_bc_project/docs/thesis/writing/
├── plan.md
├── japanese/
│   ├── 00_abstract.md
│   ├── 01_introduction.md
│   ├── 02_background.md
│   ├── 03_related_work.md
│   ├── 04_proposed_gpu_execution_framework.md
│   ├── 05_experimental_methodology.md
│   ├── 06_performance_evaluation.md
│   ├── 07_ablation_and_kernel_analysis.md
│   ├── 08_memory_scalability.md
│   ├── 09_correctness_and_numerical_behavior.md
│   ├── 10_discussion.md
│   ├── 11_conclusion.md
│   ├── appendix_a_experimental_parameters.md
│   ├── appendix_b_pathmerge_batch_sweeps.md
│   ├── appendix_c_complete_ablation_results.md
│   ├── appendix_d_correctness_details.md
│   ├── appendix_e_supplementary_baselines.md
│   └── appendix_f_reproducibility.md
└── english/
    ├── 00_abstract.md
    ├── 01_introduction.md
    ├── 02_background.md
    ├── 03_related_work.md
    ├── 04_proposed_gpu_execution_framework.md
    ├── 05_experimental_methodology.md
    ├── 06_performance_evaluation.md
    ├── 07_ablation_and_kernel_analysis.md
    ├── 08_memory_scalability.md
    ├── 09_correctness_and_numerical_behavior.md
    ├── 10_discussion.md
    ├── 11_conclusion.md
    ├── appendix_a_experimental_parameters.md
    ├── appendix_b_pathmerge_batch_sweeps.md
    ├── appendix_c_complete_ablation_results.md
    ├── appendix_d_correctness_details.md
    ├── appendix_e_supplementary_baselines.md
    └── appendix_f_reproducibility.md
```

最初は`japanese/`のみを執筆する。`english/`は日本語版の内容と数値が確定するまで翻訳を開始しない。

---

## 4. Authoritative Sources

執筆時は次を正式な根拠として使用する。

```text
thesis_bc_project/docs/thesis/evidence_matrix.tsv
thesis_bc_project/docs/thesis/thesis_values.tsv
thesis_bc_project/docs/thesis/references.bib
thesis_bc_project/docs/thesis/SOURCE_AUDIT.tsv
thesis_bc_project/result/CLAIMS.md
thesis_bc_project/result/COVERAGE.md
thesis_bc_project/result/TABLES_AND_FIGURES.md
thesis_bc_project/result/figures/thesis/
thesis_bc_project/result/tables/thesis/
thesis_bc_project/raw_data/
```

`raw_data/`は数値の再確認に使用する。論文中の値を、丸め済みspeedupから逆算してはならない。

次の文書は歴史的資料であり、正式な結果の根拠には使用しない。

```text
lab_evaluation_v2.md
pre_experiment_final_check.md
```

---

## 5. Writing Rules

### 5.1 Language

`writing/japanese/`では以下の規則を使用する。

```markdown
# Chapter 1 Introduction

## 1.1 Motivation

ここに日本語本文を書く。
```

- Chapter名とSection名は英語
- 本文は日本語
- Figure captionとTable captionは英語
- 図表内部の文字は英語
- 数式中の変数と記号は英語表記
- 実装名、製品名、API名、論文名は原表記
- 最終英訳を考慮し、一文を過度に長くしない

### 5.2 Citations

引用は`references.bib`のBibTeX keyを使用する。

```text
[@brandes2001]
[@beamer2012]
```

一次資料が確認できていない主張を追加しない。検索結果、ブログ、Wikipediaを最終根拠として引用しない。

### 5.3 Numerical Values

- 主値はmedian
- 補助値はmeanとsample standard deviation
- speedupはmedian/median
- OOMを0秒として扱わない
- warmupを本試行へ含めない
- 単一最速試行を代表値にしない
- GTEPSの式を統一する
- n、aggregation、batch sizeを併記する

### 5.4 Claim Scope

使用可能な中心主張は次に限定する。

> 固定b512のblock-based GPU_Optは、email-EuAllおよびroadNet-PA/TX/CAにおいて、グラフごとに調整した評価対象の第三者実装PathMergeより1.31～3.17倍高速だった。

本文の英訳時には次の表現を使用する。

> On the four evaluated graphs, the fixed-batch block-based GPU_Opt implementation was 1.31x to 3.17x faster than the tuned third-party PathMerge implementation evaluated in this study.

次を主張しない。

- あらゆるグラフで高速
- 常にPathMergeより高速
- 最速のBC実装
- PathMerge原著者の公式実装より高速
- PathMergeアルゴリズム一般より高速
- 他GPUでも同じ結果になる
- UMまたはChunkedが無制限にOOMを回避する
- 大規模stress条件を含めて完全な数値一致を証明した
- Hybrid BFS、Warp協調、2ストリームを個別に新規発明した

### 5.5 PathMerge Description

PathMergeの初出では必ず次を明記する。

- 評価対象は第三者実装
- 原著論文著者の公式実装ではない
- 上流は`gobardhanm/path-merging-bc`
- 上流ライセンスは確認できていない
- external comparatorでありground truthではない
- 結果をPathMerge一般へ一般化しない

### 5.6 Correctness Description

- 小規模3グラフのSequential independent full-vector比較は`Pass`
- same-batch comparisonとindependent reference comparisonを区別する
- byte-identicalでない結果を「完全一致」と書かない
- memory-pathの`Core Fail`を隠さない
- toleranceを変更して`Pass`へ変更しない
- 原因未確定の差を浮動小数点誤差と断定しない

---

## 6. Research Questions

### RQ1 Performance

評価した4グラフにおいて、固定b512のblock-based GPU_Optは、グラフごとに調整した第三者実装PathMergeより高速か。

### RQ2 Optimization Contributions

Hybrid BFS、Warp-Cooperative Accumulation、Dual-Stream ExecutionおよびBlock Kernelは、観測された性能にどの程度寄与するか。

### RQ3 Memory Scalability

GPU_Opt、GPU_Opt_Pure、GPU_Opt_Pure_Chunkedのメモリ管理方式は、実行可能なbatch sizeと性能にどのような影響を与えるか。

### RQ4 Correctness and Numerical Behavior

提案実装のBCベクトルは独立参照および異なるメモリ経路とどこまで一致し、どの条件で未解決の差が残るか。

---

## 7. Contributions

論文の貢献は次の4件に限定する。

1. GH200向けバッチ型GPU BC実行基盤の設計と実装
2. tuned third-party PathMergeに対する4グラフでの性能評価
3. Hybrid BFS、Warp-Cooperative Accumulation、Dual Streamsのアブレーション
4. UM、Pure、Chunkedの容量特性および数値的限界の評価

---

# 8. Chapter Plan

## 8.1 `00_abstract.md`

### Heading

```markdown
# Abstract
```

### Purpose

研究背景、課題、提案、評価方法、主要結果、結論を独立して理解できる形で要約する。

### Japanese Draft Structure

1. BC計算の重要性と計算コスト
2. GH200上のバッチ型GPU実行基盤
3. Hybrid BFS、block-based source processing、warp cooperation、dual streams
4. tuned third-party PathMergeとの比較
5. 主要性能値1.31～3.17倍
6. アブレーション結果
7. メモリ容量評価
8. 小規模correctnessとstress条件の制約
9. 評価範囲に限定した結論

### Length

日本語初稿は約900～1,200文字を目安とする。英語版は大学指定の300～400 wordsに調整する。

### Timing

Abstractは最後に執筆する。

---

## 8.2 `01_introduction.md`

### Heading

```markdown
# Chapter 1 Introduction
```

### Sections

```markdown
## 1.1 Motivation
## 1.2 Problem Statement
## 1.3 Research Questions
## 1.4 Contributions
## 1.5 Scope and Limitations
## 1.6 Thesis Organization
```

### Required Content

#### 1.1 Motivation

- グラフ解析の利用分野
- Betweenness Centralityの役割
- Brandes法の計算量
- GPUによる高速化の必要性
- GH200のHBM3、Grace CPU memory、NVLink-C2C、Unified Memory

#### 1.2 Problem Statement

- 全始点BFSの反復
- BFSとBackward phaseの不規則性
- グラフ構造に依存する並列性
- GPUメモリ容量
- 既存GPU実装との公平な比較

#### 1.3 Research Questions

RQ1～RQ4を記載する。

#### 1.4 Contributions

4件の貢献を記載する。

#### 1.5 Scope and Limitations

- GH200のみ
- 主性能は4グラフ
- memory scalabilityは325557限定
- PathMergeは第三者実装
- correctnessの未解決条件

#### 1.6 Thesis Organization

Chapter 2～11を各1～2文で説明する。

### Target Length

4～5ページ。

### Write After

Chapter 5～10の内容確定後に執筆する。

---

## 8.3 `02_background.md`

### Heading

```markdown
# Chapter 2 Background
```

### Sections

```markdown
## 2.1 Graphs and Betweenness Centrality
## 2.2 Brandes Algorithm
## 2.3 Parallelism in BC Computation
## 2.4 GPU Execution Model
## 2.5 GH200 Memory Architecture
## 2.6 Challenges Addressed in This Thesis
```

### Required Content

#### 2.1 Graphs and Betweenness Centrality

- $G=(V,E)$
- directed/undirected graph
- shortest path
- $\sigma_{st}$
- $\sigma_{st}(v)$
- BCの定義
- normalizationとendpointsの条件

#### 2.2 Brandes Algorithm

- BFS
- distance
- predecessor
- shortest-path count
- dependency accumulation
- BC accumulation
- 非重みグラフの計算量

#### 2.3 Parallelism in BC Computation

- source-level parallelism
- vertex-level parallelism
- edge-level parallelism
- batch processing

#### 2.4 GPU Execution Model

- thread
- warp
- block
- grid
- global memory
- shared memory
- atomic operation
- CUDA streams

#### 2.5 GH200 Memory Architecture

- HBM3
- LPDDR5X
- NVLink-C2C
- Unified Memory
- migration
- prefetch
- oversubscription

#### 2.6 Challenges

本研究が扱う技術課題を、提案手法を先取りしすぎない範囲で整理する。

### Planned Figures

- Brandes Algorithm Flow
- GH200 Memory Hierarchy

図内文字は英語とする。

### Target Length

6～8ページ。

---

## 8.4 `03_related_work.md`

### Heading

```markdown
# Chapter 3 Related Work
```

### Sections

```markdown
## 3.1 Exact Betweenness Centrality
## 3.2 GPU-Based BC Computation
## 3.3 Direction-Optimizing BFS
## 3.4 PathMerge and GPU Baselines
## 3.5 Unified Memory and Out-of-Core Processing
## 3.6 Positioning of This Work
```

### Required Content

#### 3.1 Exact Betweenness Centrality

- Brandes
- CPU並列化
- 分散処理

#### 3.2 GPU-Based BC Computation

- McLaughlin and Bader
- Sarıyüceら
- source-level/fine-grained parallelism

#### 3.3 Direction-Optimizing BFS

- top-down
- bottom-up
- Beamerら
- $\alpha$、$\beta$の出典

#### 3.4 PathMerge and GPU Baselines

- Galliot/PathMerge
- cuGraph
- evaluated third-party implementation
- official implementationではないこと
- external comparatorとしての位置づけ

#### 3.5 Unified Memory and Out-of-Core Processing

- Unified Memory
- prefetch
- oversubscription
- out-of-core graph processing
- Subwayなど

#### 3.6 Positioning

既存要素技術と本研究の差分を説明する。

### Citation Source

```text
docs/thesis/references.bib
docs/thesis/SOURCE_AUDIT.tsv
docs/thesis/12_related_work_gap.md
```

### Target Length

5～7ページ。

---

## 8.5 `04_proposed_gpu_execution_framework.md`

### Heading

```markdown
# Chapter 4 Proposed GPU Execution Framework
```

### Sections

```markdown
## 4.1 Framework Overview
## 4.2 Batch-Based Source Processing
## 4.3 Block-Based Source Assignment
## 4.4 Hybrid BFS
## 4.5 Dependency Accumulation
## 4.6 Dual-Stream Execution
## 4.7 Memory Management Variants
### 4.7.1 GPU_Opt
### 4.7.2 GPU_Opt_Pure
### 4.7.3 GPU_Opt_Pure_Chunked
## 4.8 Expected Effects and Trade-Offs
```

### Required Content

#### 4.1 Framework Overview

入力グラフからBC出力までの実行フローを示す。

#### 4.2 Batch-Based Source Processing

- source batch
- batch index
- fixed b512
- 1 block = 1 source
- streamごとのbatch

#### 4.3 Block-Based Source Assignment

- block kernel
- shared kernelとの差
- 現行実装が常時blockであること

#### 4.4 Hybrid BFS

- top-down
- bottom-up
- switching condition
- frontier representation

#### 4.5 Dependency Accumulation

- backward phase
- thread-per-vertex
- warp-cooperative accumulation
- atomic operations

#### 4.6 Dual-Stream Execution

- stream 0/1
- initialization
- computation overlap
- synchronization boundaries

#### 4.7 Memory Variants

3つの独立提案ではなく、共通実行基盤のメモリ管理方式として説明する。

#### 4.8 Trade-Offs

期待される利点とコストを説明する。実測結果はChapter 6～9へ分離する。

### Planned Figures

- Overall Framework
- Batch-to-Source Mapping
- Hybrid BFS State Transition
- Dual-Stream Timeline
- Memory Management Variants

図内文字は英語とする。

### Target Length

8～10ページ。

---

## 8.6 `05_experimental_methodology.md`

### Heading

```markdown
# Chapter 5 Experimental Methodology
```

### Sections

```markdown
## 5.1 Research Questions
## 5.2 Hardware and Software Environment
## 5.3 Graph Datasets
## 5.4 Evaluated Implementations
## 5.5 Parameter Settings
## 5.6 Timing and Statistical Method
## 5.7 Performance Comparison and PathMerge Tuning Procedure
## 5.8 Ablation and Kernel Analysis Method
## 5.9 Memory Scalability Protocol
## 5.10 Correctness Validation
## 5.11 Reproducibility and Data Provenance
## 5.12 Scope and Methodological Limitations
```

### Required Tables

- T1 Graph Metadata
- T6 Experimental Environment

### Required Content

#### 5.1 Research Questions

RQ1～RQ4と各実験項目の対応、主指標、集計方法、試行数を示す。性能評価・要因分析・容量評価・正確性検証を独立した観点として分離する。

#### 5.2 Hardware and Software Environment

- GH200（sm_90）、NVLink-C2C、UM前提
- GPUメモリは公称HBM3容量と実行環境の記録値を区別する
- 単位系（GiB / 10進GB）と取得方法の違いを混同しない
- free memoryは総容量ではなくメモリ予算の基準として扱う
- ソフトウェア版数
- PBS batch system、group
- queue名は保存ログから独立に確定できないため正式な実験条件に含めない
- memory-path実験のホストメモリ制限はresource configurationとして記述する

#### 5.3 Graph Datasets

- 実データと合成グラフ
- provenanceとSHA256
- 各グラフの選択目的とRQ対応

#### 5.4 Evaluated Implementations

- Sequential
- OpenMP
- cuGraph
- evaluated third-party PathMerge implementation
- GPU_Opt
- GPU_Opt_Pure
- GPU_Opt_Pure_Chunked

主比較と補助比較を区別する。3つのメモリ管理方式を独立提案として書かない。

#### 5.5 Parameter Settings

- GPU_Optの固定b512
- PathMergeのグラフ別調整バッチ
- 要求バッチと実効バッチ、clamp、SUB_BATCH、num_subs、NS_eff
- warmupの有無と`not_recorded`
- 性能測定とcorrectness-only実行の区別
- 実行時ノブ

#### 5.6 Timing and Statistical Method

- timing scope
- trials
- warmup
- median
- sample standard deviation
- GTEPS
- speedup
- OOM/TIMEOUT/FAILを0秒扱いしない規約

#### 5.7 Performance Comparison and PathMerge Tuning Procedure

RQ1の主要性能比較とPathMerge tuning方法を述べる。

- 主性能比較の対象4グラフと分子・分母の測定条件
- screening
- confirmation
- requested/effective batch
- clamp
- PA/TXのsweep値とmain baseline値の差
- 分子と分母のcheckpoint相違

#### 5.8 Ablation and Kernel Analysis Method

RQ2のH/W/A ablationとPA/TX forced shared/block比較の方法を述べる。

- H/W/Aの8構成と対象グラフ、試行数
- 主効果の定義
- forced shared/blockによるkernel比較（roadNet-PA/TX限定）
- profilingは部分トレース
- 未測定グラフへ一般化しない

#### 5.9 Memory Scalability Protocol

RQ3のUM/Pure/Chunkedの容量評価方法を述べる。

- 325557_3216152限定
- 評価対象はfeasibility（SUCCESS/OOM、最大成功バッチ）であり最速時間ではない
- legacy系とmemory-path系の2測定系とOOM境界の非可比性
- UM/Chunkedの無制限な容量拡張を主張しない

#### 5.10 Correctness Validation

RQ4の正確性水準を定義し、各比較の水準を区別する。

- full_vector_independent_reference
- full_vector_same_implementation
- max_bc_only
- none
- vector length
- missing index
- NaN/Inf
- mixed tolerance
- SHA256
- independent reference
- same implementation
- memory-pathのCore Failを隠さない

#### 5.11 Reproducibility and Data Provenance

- SourceSnapshotIDが正式参照（commit SHAではない）
- raw_data / code_snapshots / result / failure の分離
- 実験群とcheckpointの対応
- `result/`全体は単一checkpointに対応しない

#### 5.12 Scope and Methodological Limitations

評価範囲と方法上の制約を要約する。詳細な妥当性への影響はChapter 10で論じる。

- 対象範囲（4グラフ / 325557限定 / 小規模3グラフ）
- 比較対象としてのPathMergeの限定
- 設定の非対称性
- 未解決事項（stress正確性、CORE_FAIL）
- legacy依存
- 実行環境記録の限界（queue名を統制変数として扱わない）

### Target Length

6～8ページ。

### Writing Priority

最初に執筆する章の一つ。

---

## 8.7 `06_performance_evaluation.md`

### Heading

```markdown
# Chapter 6 Performance Evaluation
```

### Sections

```markdown
## 6.1 Main Runtime Comparison
## 6.2 Speedup over Tuned PathMerge
## 6.3 PathMerge Batch-Size Sensitivity
## 6.4 Throughput Analysis
## 6.5 Supplementary Baseline Results
## 6.6 Answer to RQ1
```

### Required Figures and Tables

- F1 Main Runtime Comparison
- F2 Main Speedup
- F3 PathMerge Batch Sweep
- T2 Main Performance

### Required Values

```text
email-EuAll: 3.17x
roadNet-PA: 1.31x
roadNet-TX: 1.51x
roadNet-CA: 1.45x
```

### Required Qualifications

- fixed GPU_Opt b512
- tuned third-party PathMerge
- median/median
- 4グラフ限定
- PathMerge一般へ一般化しない
- main baselineとsweep measurementの差を説明

### RQ1 Answer

章末に次の内容を日本語で説明し、英語版では以下を使用する。

> On the four evaluated graphs, the fixed-batch block-based GPU_Opt implementation was 1.31x to 3.17x faster than the tuned third-party PathMerge implementation evaluated in this study.

### Target Length

6～8ページ。

### Writing Priority

最初に執筆する。

---

## 8.8 `07_ablation_and_kernel_analysis.md`

### Heading

```markdown
# Chapter 7 Ablation and Kernel Analysis
```

### Sections

```markdown
## 7.1 Ablation Design
## 7.2 Effect of Hybrid BFS
## 7.3 Effect of Warp-Cooperative Accumulation
## 7.4 Effect of Dual-Stream Execution
## 7.5 Shared and Block Kernels
## 7.6 Phase Breakdown
## 7.7 Answer to RQ2
```

### Required Figures and Tables

- F4 Ablation Contributions
- F6 Shared vs Block Kernel
- F7 Phase Breakdown
- T3 Ablation Summary

### Required Results

```text
Synthetic geometric mean:
Hybrid BFS: 1.655x
Warp-Cooperative Accumulation: 1.065x
Dual Streams: 1.396x

email-EuAll:
Hybrid BFS: 1.429x
Warp-Cooperative Accumulation: 0.970x
Dual Streams: 1.720x

Kernel:
roadNet-PA block: 1.52x faster
roadNet-TX block: 1.66x faster
```

### Required Qualification

- Warp効果はグラフ依存
- emailではわずかに低速
- roadNet全体へアブレーションを一般化しない
- kernel比較はPA/TX限定
- phase breakdownから因果を断定しない

### RQ2 Answer

> Hybrid BFS and dual-stream execution provided the main observed improvements, whereas warp-cooperative accumulation was graph-dependent.

### Target Length

5～7ページ。

---

## 8.9 `08_memory_scalability.md`

### Heading

```markdown
# Chapter 8 Memory Scalability
```

### Sections

```markdown
## 8.1 Evaluation Scope
## 8.2 In-Capacity Performance
## 8.3 Unified Memory Oversubscription
## 8.4 Chunked Execution
## 8.5 Memory Migration and Profiling
## 8.6 Answer to RQ3
```

### Required Figures and Tables

- F5 Memory Scalability
- T4 Memory Scalability

### Required Results

```text
GPU_Opt_Pure: success through b4096
GPU_Opt: success through b10240; b12288 OOM_OR_FAIL (exit 137)
GPU_Opt_Pure_Chunked: success through tested b16384
```

### Required Qualification

- 325557_3216152限定
- legacy feasibility result
- current block-kernel performance comparisonではない
- UMは無制限ではない
- Chunkedの主な利点は容量
- 27.918 MBは25秒部分トレース
- 全実行migration量と書かない

### RQ3 Answer

> On the evaluated 325557 graph, Unified Memory extended the executable batch range beyond the pure device-memory variant, while chunked execution extended it further within the tested range.

### Target Length

4～6ページ。

---

## 8.10 `09_correctness_and_numerical_behavior.md`

### Heading

```markdown
# Chapter 9 Correctness and Numerical Behavior
```

### Sections

```markdown
## 9.1 Validation Levels
## 9.2 Small-Graph Independent Validation
## 9.3 Tuned-Configuration Consistency
## 9.4 Memory-Path Same-Batch Comparison
## 9.5 Stress-Condition Core Failures
## 9.6 Answer to RQ4
```

### Required Table

- T5 Correctness Summary

### Required Content

#### 9.1 Validation Levels

- full_vector_independent_reference
- full_vector_same_implementation
- max_bc_only
- none

#### 9.2 Independent Validation

- benchmark_7000_41459
- benchmark_11023_62184
- chain_200
- Sequential vs GPU_Opt
- mismatch=0
- missing=0
- NaN/Inf=0

#### 9.3 Tuned Consistency

- email b64 vs b2048
- CA b32 vs b64
- mixed tolerance
- Max BC

#### 9.4 Same-Batch

- memory paths
- tolerance内一致
- SHA256非一致
- byte-identicalではない

#### 9.5 Core Fail

- 2行を保持
- mismatch=6
- max relative error
- 原因未確定
- PathMerge cross correctnessはundetermined

### RQ4 Answer

> Independent full-vector agreement was confirmed on the three evaluated small graphs, but numerical agreement under all large memory-stress conditions was not established.

### Target Length

4～6ページ。

---

## 8.11 `10_discussion.md`

### Heading

```markdown
# Chapter 10 Discussion
```

### Sections

```markdown
## 10.1 Interpretation of the Performance Results
## 10.2 Relationship with Graph Characteristics
## 10.3 Performance-Capacity Trade-Off
## 10.4 Implications for GH200
## 10.5 Threats to Validity
### 10.5.1 Internal Validity
### 10.5.2 External Validity
### 10.5.3 Construct Validity
### 10.5.4 Baseline Validity
## 10.6 Limitations and Future Work
```

### Required Content

#### 10.1 Performance

結果とアブレーションを横断して説明する。測定していない因果を断定しない。

#### 10.2 Graph Characteristics

- average degree
- BFS depth
- graph size
- road network structure

4グラフ程度の相関から一般的な法則を断定しない。

#### 10.3 Trade-Off

- GPU_Opt
- Pure
- Chunked
- performance
- capacity
- numerical behavior

#### 10.4 GH200

GH200上での観測として説明し、他GPUへ一般化しない。

#### 10.5 Threats to Validity

PathMerge第三者実装、trial数、legacy測定、correctness tolerance、グラフ選択を含める。

#### 10.6 Future Work

- Core Fail原因分析
- official/independent PathMerge implementation comparison
- current blockによる統一7実装比較
- 他GPU
- 追加グラフ
- full-duration migration
- large-scale independent reference

### Target Length

5～7ページ。

---

## 8.12 `11_conclusion.md`

### Heading

```markdown
# Chapter 11 Conclusion
```

### Sections

```markdown
## 11.1 Summary
## 11.2 Answers to the Research Questions
## 11.3 Contributions
## 11.4 Final Remarks
```

### Required Content

#### 11.1 Summary

問題、提案、評価を簡潔に再説明する。

#### 11.2 RQ Answers

RQ1～RQ4を各1段落で回答する。

#### 11.3 Contributions

Introductionの4つの貢献と対応させる。

#### 11.4 Final Remarks

評価範囲内の結論と、未解決の正確性・一般化の制約を述べる。

新しい結果や新しい主張をConclusionで追加しない。

### Target Length

2～3ページ。

---

# 9. Appendix Plan

## 9.1 `appendix_a_experimental_parameters.md`

```markdown
# Appendix A Complete Experimental Parameters
```

- 全batch size
- environment variables
- PBS resources
- checkpoint
- timing scope
- tolerance

## 9.2 `appendix_b_pathmerge_batch_sweeps.md`

```markdown
# Appendix B Complete PathMerge Batch Sweeps
```

- 全trial
- requested/effective batch
- median
- sample SD
- clamp
- screening/confirmation

## 9.3 `appendix_c_complete_ablation_results.md`

```markdown
# Appendix C Complete Ablation Results
```

- H/W/A全8構成
- 全グラフ
- trial値
- main effect

## 9.4 `appendix_d_correctness_details.md`

```markdown
# Appendix D Correctness Details
```

- vector SHA256
- graph SHA256
- error index
- values at error index
- max absolute/relative error
- mismatch
- missing
- NaN/Inf

## 9.5 `appendix_e_supplementary_baselines.md`

```markdown
# Appendix E Supplementary Baseline Results
```

- Sequential
- OpenMP
- cuGraph
- legacy partial 7 implementations
- missing resultsをN/Aと表示
- current block統一比較ではないことを明記

## 9.6 `appendix_f_reproducibility.md`

```markdown
# Appendix F Reproducibility
```

- repository structure
- raw_data
- result
- code_snapshots
- figure/table generation
- integrity check
- hardware requirements
- private repositoryの限定共有
- third-party code notice

---

# 10. Figure and Table Placement

| Artifact | Chapter |
|---|---|
| Brandes Algorithm Flow | Chapter 2 |
| GH200 Memory Hierarchy | Chapter 2 |
| Overall Framework | Chapter 4 |
| Batch-to-Source Mapping | Chapter 4 |
| Hybrid BFS State Transition | Chapter 4 |
| Dual-Stream Timeline | Chapter 4 |
| Memory Management Variants | Chapter 4 |
| T1 Graph Metadata | Chapter 5 |
| T6 Experimental Environment | Chapter 5 |
| F1 Main Runtime Comparison | Chapter 6 |
| F2 Main Speedup | Chapter 6 |
| F3 PathMerge Batch Sweep | Chapter 6 |
| T2 Main Performance | Chapter 6 |
| F4 Ablation Contributions | Chapter 7 |
| F6 Shared vs Block Kernel | Chapter 7 |
| F7 Phase Breakdown | Chapter 7 |
| T3 Ablation Summary | Chapter 7 |
| F5 Memory Scalability | Chapter 8 |
| T4 Memory Scalability | Chapter 8 |
| T5 Correctness Summary | Chapter 9 |

同じ数値を図と表と本文で不必要に3回繰り返さない。本文では重要な傾向と代表値を説明し、全数値は表または付録へ置く。

---

# 11. Recommended Writing Order

章番号順ではなく、内容が確定している順に執筆する。

1. Chapter 5 Experimental Methodology
2. Chapter 6 Performance Evaluation
3. Chapter 7 Ablation and Kernel Analysis
4. Chapter 8 Memory Scalability
5. Chapter 9 Correctness and Numerical Behavior
6. Chapter 4 Proposed GPU Execution Framework
7. Chapter 2 Background
8. Chapter 3 Related Work
9. Chapter 10 Discussion
10. Chapter 11 Conclusion
11. Chapter 1 Introduction
12. Abstract
13. Appendices
14. English translation

---

# 12. Chapter Completion Checklist

各章の完了時に確認する。

- [ ] Chapter名とSection名が英語
- [ ] 日本語版の本文が日本語
- [ ] 略語を初出で定義
- [ ] 全引用keyが`references.bib`に存在
- [ ] 数値が`thesis_values.tsv`またはraw dataと一致
- [ ] trialsとaggregationが明記されている
- [ ] Figure/Tableを本文から参照
- [ ] Figure/Table captionが英語
- [ ] 図表内部に日本語がない
- [ ] 主張がevidence matrixの範囲内
- [ ] PathMergeを公式実装と誤記していない
- [ ] PathMergeをground truthとしていない
- [ ] Core Failを隠していない
- [ ] 未測定グラフ・他GPUへ一般化していない
- [ ] 同じ説明を複数章で過度に重複していない
- [ ] 新しい実験値を推定していない
- [ ] TODO_SOURCEを残していない
- [ ] 英訳しやすい文構造になっている

---

# 13. Cross-Chapter Consistency

次の用語を全章で統一する。

| Concept | Standard Term |
|---|---|
| 媒介中心性 | Betweenness Centrality (BC) |
| 幅優先探索 | Breadth-First Search (BFS) |
| 提案基盤 | batch-based GPU execution framework |
| 主実装 | GPU_Opt |
| 純GPUメモリ版 | GPU_Opt_Pure |
| 分割版 | GPU_Opt_Pure_Chunked |
| 統合メモリ | Unified Memory (UM) |
| 二方向BFS | Hybrid BFS |
| Warp協調 | Warp-Cooperative Accumulation |
| 2ストリーム | Dual-Stream Execution |
| 比較対象 | tuned third-party PathMerge implementation |
| 主要比較 | main performance comparison |
| 補助比較 | supplementary baseline comparison |
| 実行可能性 | feasibility |
| 容量拡張性 | memory scalability |
| 混合許容 | mixed absolute-relative tolerance |
| 未解決不一致 | Core Fail |

`GPU_Opt`、`GPU_Opt_Pure`、`GPU_Opt_Pure_Chunked`は3つの独立提案ではなく、共通GPU実行基盤におけるメモリ管理方式のバリエーションとして統一する。

---

# 14. Final Validation

日本語版全章完成後、次を実施する。

1. 全Chapter/Sectionの目次生成
2. Figure list生成
3. Table list生成
4. abbreviation list生成
5. symbol list生成
6. BibTeX key検査
7. 参考文献の未使用・重複検査
8. 数値の自動照合
9. 主要speedup 3.17/1.31/1.51/1.45の確認
10. PathMerge限定表現の確認
11. Core Fail記述の確認
12. 図表内の日本語文字検査
13. 全図表の目視検査
14. Chapter間の重複検査
15. 過大主張検査
16. 英訳前の指導教員レビュー

---

# 15. Definition of Done

日本語版の完了条件は次のとおりとする。

- Chapter 1～11がすべて存在する
- Abstractが存在する
- Appendix A～Fが存在する
- 全RQに対応する結果章がある
- 全主張がevidence matrixと対応する
- 全数値が正式データと一致する
- 関連研究に一次資料が引用されている
- PathMergeの第三者実装・ライセンス制約が記録されている
- small full-vector Passとmemory-path Core Failの両方が記載されている
- 図表が正しい章に配置されている
- 本文に未解決placeholderがない
- 指導教員が日本語版の構成と内容を確認できる状態になっている

英語版は、日本語版の内容承認後、同一の節構成・数値・引用を保ったまま翻訳する。英訳時に新しい主張や結果を追加しない。
