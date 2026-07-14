# Memory-path correctness (325557_3216152)

GH200 のメモリ経路 (UM / Pure / Chunked) と大バッチ・oversubscription 条件で、
提案実装が出力する全 BC ベクトルを混合許容基準で比較した正確性・診断アーカイブである。
対象グラフは `data/325557_3216152` の 1 グラフに限定する。canonical は checkpoint
`memory_correctness_20260712` / PBS job `2368587`、診断は checkpoint `memory_diagnostic_20260713` / PBS job `2369632`。
各構成 n=1、warmup なしで、時間値は性能評価・性能主張に使用しない。

判定は `abs_tol=1e-3`、`rel_tol=1e-6`、`abs_diff <= abs_tol + rel_tol * max(|a|,|b|)`。
**PathMerge は external comparator であり ground truth ではない。**

## 構成

| パス | 内容 |
|:--|:--|
| `canonical_job_2368587/` | canonical 比較行列 (checkpoint `memory_correctness_20260712`, job `2368587`)。6 構成すべて runner 成功。formal overall status=`CORE_FAIL` |
| `diagnostic_job_2369632/` | T-RESET / T-NSEFF 一因子診断 (checkpoint `memory_diagnostic_20260713`, job `2369632`)。`DIAGNOSTIC_COMPLETE` |
| `analysis/` | Gate G2.2/G2.3 の read-only 分析 (run-to-run / stress / 影響頂点 / 許容感度 / 監査)。raw ベクトルから `scripts/analyze_memory_correctness.py` で再生成可能 |
| `SOURCE.md` | provenance・入力/ベクトル SHA256・再現手順・制約 |

## 支持できる結論

- 325557、`b1024` において UM / Pure / Chunked は**事前設定した混合許容内で一致**した
  (`same_batch_diff_path` の mismatch=0)。
- この一致は **byte 一致 (SHA256 一致) ではない**。
- UM `b9792` は 100 GiB 制限内で**完走**し、oversubscription 経路証拠 (est>free_before,
  HBM3 streaming, NS_eff=1, num_subs=2, SUB_BATCH<batch, Prefetch cum>0) を満たした。
  ただしこれは migration byte 量の直接計測ではない。
- Chunked `b16384` は `num_subs=3` で**完走**した。
- 診断では **full memset 強制**と **NS_eff=1 強制**のいずれも、`b1024` CONTROL との差を
  生じなかった (CONTROL vs T-RESET / T-NSEFF いずれも mismatch=0)。
- 同一構成の**実行間差は混合許容内**だった (Pure/PathMerge の run-to-run mismatch=0)。

## 未解決 (支持しない)

stress 条件では、事前設定した `rel_tol=1e-6` を 8 頂点の和集合で超過する構成依存差を
観測した。full reset および NS_eff=1 の単独変更では差を再現できず、large batch、
sub-batch 分割、grid/occupancy またはその組合せとの関連が残った。原因は未特定で、
通常の run-to-run 丸め変動だけでは説明困難である。

PathMerge (external comparator) との差 (約 11027 要素、最大相対差約 0.2%) は本 stress 差とは
別 regime であり、正誤は未決定である。**この差から提案実装が誤りとは断定しない。**

## スコープと非主張

- 対象は 325557 のみ。他グラフ・他バッチ・最新 block へ一般化しない。
- `rel_tol=3e-6` で消える (許容感度) ことは補助情報であり、正式 FAIL を PASS に変更しない。
- stress full-vector 正確性、UM/Chunked の全条件正確性は**証明していない**。
- 各構成 n=1・warmup なし。時間値は性能結果ではない。

数値の一次情報は各 job ディレクトリの `comparison_matrix.tsv` / `execution_summary.tsv` /
`FINAL_STATUS.txt`。raw BC ベクトルは Git 管理外で、`../../EXTERNAL_ARTIFACTS.tsv` に
size/SHA256 を登録する。詳細は `SOURCE.md`。
