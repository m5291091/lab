# COVERAGE — 実験カバレッジ（叙述）

機械可読版は `coverage_matrix.tsv`（19列）。ここでは実験×グラフ×実装の充足と欠損理由を叙述する。

## 主軸(A): 提案 block vs PathMerge tuned
- **proposed_variants**（checkpoint phase_def_block_20260710, block）: email-EuAll / roadNet-PA/TX/CA × {GPU_Opt, GPU_Opt_Pure, GPU_Opt_Pure_Chunked}。in-capacity（BATCH=512, SUB_BATCH=512, num_subs=1, NS_eff=2）。email n=5, road n=3。**充足**。
- **tuning/pathmerge**（tuned 分母）: PA/TX/CA/email/325557 の掃引。tuned = PA/TX b64 / CA b32 / email b2048。**充足**。5 グラフすべてに graph 別 `SOURCE.md` が存在する（email-EuAll は Gate W14.1 で raw から再構成、新規測定値なし）。stage 別 job 対応は `tuning/pathmerge/SOURCE.md` の表を正本とする。
- **PathMerge tuning の正確性水準**: email（b64 vs b2048）と CA（b32 vs b64）だけが別設定の全ベクトル比較を持つ。PA/TX は tuned と default がともに b64 で、別の full-vector comparison artifact が存在しないため、両方 **max_bc_only** とする。b64 の自己比較を全ベクトル証拠には数えない。
- **email/CA の vector 保存状態（Gate W5.2）**: 上記 2 比較に用いた BC ベクトル 4 本は `currently_unavailable`（original runtime path = `build_miyabi/{t1_correctness,t1_ca_correctness}/bc_b*.txt`, 台帳 `EXTERNAL_ARTIFACTS.tsv`）。保存されているのは実験時の比較 summary のみで、archive-time に vector を再解析していない。当時の summary に基づく PASS 判定と `full_vector_same_implementation` は維持し、NaN/Inf/duplicate index は `not_recorded` とする。
- **tuning/kernel_selection**（forced shared/block 比較, 選択則非依存）: roadNet-PA/TX のみ（2グラフ）。強制比較で block が shared より PA 1.52×/TX 1.66× 高速・Max BC 一致を確認。未測定グラフへの一般化はしない。旧 avg_deg 選択則は現在不使用。
- 正確性: headline 4グラフは **max_bc_only**（提案3実装間 + 独立参照 PathMerge の Max BC 一致）。独立参照 full-vector は未実施 → `CLAIMS.md`。

## A 小規模 full-vector 正確性（DONE）
- benchmark_7000_41459 / benchmark_11023_62184 / chain_200 × Sequential(独立参照) vs GPU_Opt を checkpoint `small_correctness_20260712`, job `2367583.opbs` で各 n=1、warmup なし、requested/effective/SUB_BATCH=512/512/512、num_subs=1、NS_eff=2 で実行。
- 全3グラフで vector length=n、missing index=0、NaN/Inf=0、混合許容不一致=0、comparison exit=0。`correctness/small_full_vector/` に正式配置し、**この3グラフに限り full_vector_independent_reference を充足**。
- headline email/roadNet、GPU_Opt_Pure/Chunked、UM oversubscription 固有経路は対象外。個別の Hybrid BFS/warp 経路を専用カウンタで検証したものでもない。

## メモリ経路 correctness/diagnostic（memory_paths; Stage 4C, 325557 限定）
- canonical（checkpoint `memory_correctness_20260712`, job `2368587`）: 325557 × {gpu_opt b1024/b9792, gpu_opt_pure b1024, gpu_opt_pure_chunked b1024/b16384, pathmerge b4096} を各 n=1・warmup なしで全実行し比較行列化。runner 6/6 成功、formal overall status=`CORE_FAIL`（隠さない）。
- `same_batch_diff_path`（UM/Pure/Chunked b1024）は mismatch=0（`abs_tol=1e-3`, `rel_tol=1e-6`）だが**非 byte 一致**。`same_impl_diff_batch`（b9792 vs b1024, b16384 vs b1024）は厳格 `rel_tol=1e-6` を**和集合 8 頂点**で超過。`pathmerge_cross` は 5/5 DIFF（約 11027 要素, max_rel≈2.0e-3, 未解決; PathMerge は external comparator）。
- UM `b9792` はhost-memory-limited 100 GiB configurationで完走し oversubscription 経路証拠を満たす（migration byte 直接計測ではない）。Chunked `b16384` は `num_subs=3` で完走。
- 診断（checkpoint `memory_diagnostic_20260713`, job `2369632`）: full memset 強制（T-RESET）・NS_eff=1 強制（T-NSEFF）とも b1024 CONTROL との差なし（`RESET/NS_EFF_NOT_DISTINGUISHED`, mismatch=0）。原因未特定。
- 失敗: `b10240` UM はhost-memory-limited 100 GiB configurationでOOM（job `2368269`, `failure/failed/oom/`）。job `2368398` は pure 比較不一致で fail-fast（`failure/early_terminated/`）。
- **制約**: 325557・各構成 n=1・warmup なし。他グラフ/他バッチ/最新 block へ一般化しない。stress full-vector 正確性は `NOT_YET_SUPPORTED`（`CLAIMS.md`）。分析 TSV/MD は `analyze_memory_correctness.py` で raw vector から byte-identical 再生成可。

## 副次(B): メモリスケーラビリティ（memory_scalability）
- 325557_3216152（合成, 人為的バッチ強制）1グラフ × {gpu_opt(UM), gpu_opt_pure, gpu_opt_pure_chunked} × batch 512–16384, n=5（例外: gpu_opt b12288 は失敗 1 試行で掃引停止 n=1、gpu_opt b16384 は未試行）。
- feasibility: pure OOM@b8192+（CUDA OOM をログで確認） / UM→b10240（b12288 は OOM_OR_FAIL, exit 137, n=1, 原因の独立記録なし） / chunked→b16384 全SUCCESS。
- **制約**: 旧ツリー(oldtree_f05ec52_20260512)測定・時間値非採用（`provenance/um_code_diff_audit.md` でメモリサイジングコード文字単位同一を確認し feasibility を限定的に再利用（phase_def_block_20260710 で再実測はしていない））。BC計算 feasibility は代表 full-vector（C1）後に SUPPORTED。自然大規模グラフ(V≥5M)は不在。

## ablation（提案内部）
- synthetic_2354994: benchmark_7000/11023 + 56438 + 325557 × 8構成(H/W/A) × n=5。
- email_2354999: email-EuAll × 8構成 × n=3。
- H(hybrid BFS)が最大寄与。dir 内 max_bc なし（正確性 `none`）。

## 7実装比較（seven_implementations/legacy_partial, legacy 旧shared）
- small: 概ね7実装（Sequential は 56438 欠）。medium/large: 提案系4 + PathMerge のみ（Seq/OMP/cuGraph 欠）。
- 現行 block での完全統一表ではない（`seven_implementations/README.md`）。medium/large 完全7実装 = `NOT_YET_SUPPORTED`。

## profiling
- bandwidth（HBM3 1818.6 / C2C 177.7 GB/s）= 充足。nsys ablation_H1W1A0/A1（A旗効果）。um_prefetch = **25秒部分トレース**（総量主張不可）。

## 主要な欠損と理由
| 欠損 | 理由 | 区分 |
|:--|:--|:--|
| headline 4グラフの独立参照 full-vector | 未実施（小規模検証は 7000/11023/chain_200 のみ） | A正確性=SUPPORTED_WITH_LIMITATIONS |
| medium/large の Sequential/OpenMP/cuGraph | 未測定（Seq は非現実的コスト） | 7実装med/large=NOT_YET_SUPPORTED |
| UM の block 性能値 | 未再測定（副次のため必須でない） | B性能=NOT_YET_SUPPORTED |
| memory-path stress full-vector（325557, 大batch/NS_eff=1/num_subs>1） | 構成依存差が厳格 rel_tol=1e-6 を和集合8頂点で超過、原因未特定 | Stress full-vector=NOT_YET_SUPPORTED |
| memory-path vs PathMerge full-vector 一致（325557） | 約11027要素差(max_rel≈2.0e-3)、正誤未決定、PathMergeはexternal comparator | PathMerge cross=NOT_YET_SUPPORTED |
| 自然 V≥5M oversubscribe | グラフ不在 | 任意強化 |

## Gate W7.3A — 325557 入力不正と再検証準備（**案。再検証前の暫定状態**）
- `data/325557_3216152` は **malformed**（1-based を 0-based として格納。`2m` に 7 要素不足、範囲外 ID `325557` × 7、頂点 0 孤立、1-based 最終頂点の行が欠落）。カタログ全 12 行のうち malformed はこの 1 本のみで、**他 10 グラフは PASS**（判定変化なし）。
- 修復版 `data/325557_3216152_corrected_v1` を追加（`tools/repair_325557_graph.py`、決定的・byte-identical、`n`/`m`・self-loop 87,442 本・duplicate 866,924 を保存、範囲外 ID 0、対称、連結成分 1）。`ProvenanceStatus=internally_reconstructed_no_original_seed`。
- **本 Gate では GPU 実行・qsub をしていない**。以下は再検証待ちで、カバレッジは**未充足として扱う**。
  - Series A（正確性・memory-path）: UM/Pure/Chunked b1024、UM stress b9792、Chunked stress b16384、PathMerge b4096 の全 vector 生成後に比較。PathMerge は cross-implementation comparator（ground truth ではない）。
  - Series B（容量境界の最小確認）: Pure b4096/b8192、UM b10240/b12288、Chunked b16384 を各 1 試行。feasibility confirmation であり性能測定ではない。**失敗を 0 秒として集計しない**。
  - Series C（アブレーション）: 修復版 325557 × 8 構成 × n=5。job 2354994 の一次資料監査に合わせ、各 `run_ablation ... all` invocation（8構成セット）の先頭で global・untimed H1W1A1 warmup を1回実行し、40本試行には含めない。8構成集合、各trial 1〜5、合計40行、有限正のTime/GTEPS、runner exit 0、失敗markerなしを機械検証する。既存 3 グラフは再実行せず、325557 の値のみ差し替えて幾何平均を再集計できる構成にする。
- 既存の 325557 由来の結果（`correctness/memory_paths/`、`memory_scalability/`、ablation synthetic の 325557 分、`CORE_FAIL`）は **malformed legacy input 上の履歴結果**として保存し、削除・置換しない。
- **RQ1 main performance（email-EuAll / roadNet-PA/TX/CA）は 325557 を使用しないため影響を受けない。**

## Gate W7.4 — 修正版325557 再検証 **完了**（案→充足）
再検証を実施し、上記 Series A/B/C はすべて完了した（GPU 実行済み・独立監査済み; Gate W7.3C1）。

- **Series A/B（job 2404743, checkpoint `45352a3`, `SUCCESS`）**: raw=`raw_data/corrected_325557/job_2404743/`、正式=`result/correctness/corrected_325557/`・`result/memory_scalability/corrected_325557/`。
  - 正確性: 6ベクトル完全、**10比較すべて mismatch=0**（旧 malformed で FAIL していた stress `same_impl_diff_batch`（b9792/b1024, b16384/b1024）も PASS, max_rel<=5.09e-13, 非byte一致）。PathMerge=external comparator。→ memory-path stress full-vector は corrected 入力で **一致（旧 CORE_FAIL を置換）**。
  - 容量境界（feasibility, 各1試行）: pure_b8192=**CUDA OOM**（ログ確認）、um_b10240=SUCCESS、**um_b12288=host/cgroup memory OOM kill（exit137, CUDA/HBM OOM ではない）**、chunked_b16384=SUCCESS。入力ファイル≈43.25 MiB、容量問題は batch 依存 working set。
- **Series C（job 2406254, checkpoint `45352a3`, `SUCCESS_COMPLETE_40`）**: raw=`raw_data/corrected_325557/job_2406254/`、正式=`result/ablation/corrected_325557/`。40行完全・warmup 5回(untimed H1W1A1, 統計非算入)。**H=1.4767 / W=1.1012 / A=1.5563**。合成4集約（他3=job2354994 raw不変 + 325557修正版）= **H=1.6787 / W=1.0661 / A=1.3914**（本文丸め 1.679/1.066/1.391, **mixed-checkpoint**）。
- 失敗系列: build 失敗 job 2403658（`failure/failed/build/`）、OOM マーカー誤判定 job 2404249（`failure/failed/validation/`）を自己完結で保持。
- 旧 malformed 入力上の `CORE_FAIL`・legacy feasibility・旧 synthetic 325557 値は **historical** として保存（削除・上書きせず、現行 claim 非使用）。
- 制約: 4 synthetic graphs、mixed checkpoints、325557 のみ修正版再測定、各境界 1 試行、修正版は内部再構成(seed 不明)。roadNet・他 GPU へ一般化しない。PathMerge は独立正解ではない。
