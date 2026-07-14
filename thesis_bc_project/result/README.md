# result/ — 修士論文用 実験結果（正規アーカイブ）

Gate B 整理で、`results_miyabi/`（checkpoint `phase_def_block_20260710`）と `legacy_results_miyabi/`（旧ツリー baseline）を
論文主張に沿って再編した正規ディレクトリ。生データは canonical に一度だけ保存し、比較表は派生。

## 主要主張と数値の索引（詳細は `CLAIMS.md`）
- **主軸(A)** 提案 block GPU_Opt vs PathMerge **tuned**（tuned基準）:
  email **3.17×** / roadNet-PA **1.31×** / roadNet-TX **1.51×** / roadNet-CA **1.45×**
- **副次(B)** UM/Chunked メモリスケーラビリティ（feasibility; 旧カーネル測定・時間値非採用）
- **正確性(A, 小規模限定)** benchmark_7000_41459 / benchmark_11023_62184 / chain_200 で Sequential 独立参照 vs GPU_Opt の full-vector 不一致0（email/roadNetは未実施）

## 構成
| パス | 内容 |
|:--|:--|
| `main_performance/proposed_variants/` | 提案3実装(block, UM/Pure/Chunked)再計測 — **canonical raw**（checkpoint phase_def_block_20260710） |
| `main_performance/proposed_vs_pathmerge/` | 主軸(A)最終比較表 + canonical link（生データ重複なし） |
| `main_performance/seven_implementations/legacy_partial/` | 7実装(部分)legacy baseline（旧shared）。制約は同 README |
| `tuning/pathmerge/` | PathMerge tuned バッチ掃引 — canonical raw |
| `tuning/kernel_selection/` | BFS カーネル forced shared/block 比較（選択則非依存, PA/TX）= block 採用の裏付け |
| `phase_breakdown/` | BFS/Backward 内訳 + `phase_breakdown.pdf` |
| `ablation/` | Ablation(H/W/A) — synthetic_2354994(n5) / email_2354999(n3) |
| `correctness/pathmerge_tuned/` | PathMerge tuned batch 間 全ベクトル一致検証 |
| `correctness/small_full_vector/` | 小規模3グラフの Sequential 独立参照 vs GPU_Opt 全ベクトル検証（SourceSnapshotID small_correctness_20260712, job 2367583.opbs） |
| `correctness/memory_paths/` | GH200 メモリ経路(UM/Pure/Chunked)・大バッチ正確性/診断（325557限定, canonical=memory_correctness_20260712/2368587, 診断=memory_diagnostic_20260713/2369632）。同一batch mismatch=0(非byte一致)、stress差=未解決、formal `CORE_FAIL` 保存 |
| `memory_scalability/` | UM/pure/chunked feasibility（副次B; 意図的OOM同居） |
| `profiling/` | 帯域 + nsys（um_prefetch は25秒部分トレース注記） |
| `tables/` `figures/` | 最終表 / 図 |
| `environment/` `datasets/` `provenance/` | 実行環境 / グラフカタログ / git SHA・コード差分監査 |

## 追跡可能性・監査
- `MANIFEST.md` 実行環境・PBS job・入力・集計
- `CLAIMS.md` 各主張の支持状態（**未実行検証を完了扱いしない**）
- `COVERAGE.md` / `coverage_matrix.tsv` カバレッジ（`SourceSnapshotID` 列で実験時コードを特定）
- `TABLES_AND_FIGURES.md` 表・図の入力(新パス)・再生成コマンド
- `MIGRATION_MAP.tsv` 移行表（Original/New SHA256 + HashMatch、`SourceSnapshotID` 列で commit 非依存）
- `EXTERNAL_ARTIFACTS.tsv` **Git外に残す最小台帳**（Gate J1.1 でスリム化, 4 件）: 再生成可能 `.sqlite` 3 + ビルド成果物ディレクトリ補助記録 1。全件監査は `provenance/EXTERNAL_ARTIFACTS_AUDIT.tsv`（42 件×11 列, Decision 付）、`.sqlite` 再生成は `provenance/SQLITE_REGENERATION.tsv`
- **Git 履歴非依存化**: 生データは `../raw_data/`（`MANIFEST.tsv`/`SHA256SUMS`, Gate J1.1 で 164 件）、実験時コードは `../code_snapshots/<SourceSnapshotID>/` に凍結保存。commit SHA は `../code_snapshots/_legacy_audit/LEGACY_COMMIT_TO_SNAPSHOT.tsv`（監査用）にのみ保持し、データ/コードアクセスに不要。
- 非正常データは `../failure/`（要約）+ `../raw_data/unsuccessful/`（生データ本体）

## 集計規約
中央値(median)、warmup=なし、SourceSnapshotID `phase_def_block_20260710`。主要値は元 TSV から再生成し変更しない。
