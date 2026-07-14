# seven_implementations — 7実装比較（legacy 部分データ）

> **重要: これは現行 block 実装による完全な7実装統一表ではない。** legacy（旧 shared カーネル・旧ツリー測定）の**部分的**なアーカイブである。

生データは `legacy_partial/` に**行単位で分割・書換えせず**そのまま保存している。

## 対象7実装
Sequential / OpenMP / cuGraph_BC / GPU_Opt / GPU_Opt_Pure / GPU_Opt_Pure_Chunked / PathMerge_BC

## カバレッジの制約（必読）
- **7実装すべてが全グラフで揃っているわけではない。**
- **提案実装（GPU_Opt / GPU_Opt_Pure / GPU_Opt_Pure_Chunked）は旧 shared 経路**で測定されている（現行の常時 block ではない）。
- **medium / large では Sequential / OpenMP / cuGraph_BC が欠ける**（提案系4 + PathMerge_BC のみ）。
- small のみ Sequential / OpenMP / cuGraph_BC / 提案系 / PathMerge の 7実装が概ね揃う（Sequential は 56438 欠）。

## 論文で使用できる行 / 使用できない行
| 用途 | 使用可否 | 備考 |
|:--|:--|:--|
| PathMerge_BC（baseline） | **使用可（legacy baseline）** | 主軸(A)の PathMerge 既定 b64 baseline は large の `results_no_gpu_opt.tsv` を使用 |
| Sequential / OpenMP / cuGraph_BC（small） | 使用可（`SUPPORTED_WITH_LIMITATIONS`, small 限定） | 7実装 small 比較のみ |
| 提案系（GPU_Opt* = 旧 shared） | **headline には使用しない** | 現行 block 値は `../proposed_variants/` に置換済み。headline 4グラフ(email/PA/TX/CA)の提案値はこちらを使う |
| 提案系（medium/large の非headlineグラフ, 旧shared） | 参考のみ | 追加測定なしでは統一比較に使えない |

## 追加実験を行わない場合の制約
- medium/large の完全7実装統一比較は **`NOT_YET_SUPPORTED`**（Sequential/OpenMP/cuGraph 欠 + 提案系が旧shared）。
- 現行 block での 7実装統一表を主張するには GPU 再測定が必要（`../../CLAIMS.md` / `COVERAGE.md` 参照; Sequential は med/large で非現実的コスト）。

## checkpoint / provenance
- 測定: 旧ツリー(mylab/research, pre-consolidation, 旧 shared)。UM を除く正確な測定commitは未固定（アーカイブ `e7b86de`）。
- Max BC 一致（正確性）: `legacy_partial/{small,medium,large}/correctness_no_gpu_opt.md` 等。
