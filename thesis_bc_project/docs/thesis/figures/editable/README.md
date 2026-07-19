# Editable Thesis and Presentation Figure Library

このディレクトリは、修士論文と発表資料で再利用する **1 slide = 1 figure** の編集用ライブラリである。caption と slide title は図へ焼き込まず、論文・story deck 側で管理する。既存の正式 F1--F7 は置換しない。

## Figure ID namespace（重要）

本ライブラリの ID **`F01`--`F15`** と、`result/figures/thesis/` の canonical result
figure ID **`F1`--`F7`** は **別の namespace** である。

> **`F01` は `F1` ではない。** 桁数が異なるだけの別体系であり、対応関係は
> `FIGURE_MANIFEST.tsv` の `CanonicalResultFigureID` 列だけが定義する。

- `Namespace` 列は本ライブラリの全行で `editable_library` である。
- `ThesisFigureNumber` 列が、読者が本文で見る番号（Figure 1.1 など）である。
  library ID は本文に出さない。
- conceptual 図 (`F01`--`F08`) には対応する result figure が無いため
  `CanonicalResultFigureID` は `not_applicable` とする。
- chart 図 (`F09`--`F15`) は canonical result figure の編集用複製であり、
  論文本体へ掲載するのは `scripts/generate_thesis_artifacts.py` が生成する
  canonical result figure の方である。

## Files

- `thesis_figure_library.pptx`: 16:9、白背景、英語のみの図。概念図は native shape / connector / text、データ図は native chart または editable shape chart。
- `figure_data.xlsx`: 図別worksheet。データ図はcanonical TSV/logから取得した未丸め値、conceptual図はlabel・source・notesを保持する。
- `FIGURE_MANIFEST.tsv`: 図番号、目的、章、canonical source、編集形式の対応表。
- `../../../../scripts/generate_editable_figure_library.py`: 再生成器。

## Regeneration

repository内に成果物以外の依存物を置かないGate V0実行例:

```bash
cd thesis_bc_project
PYTHONPATH=/tmp/gate_v0_editable_deps python3 scripts/generate_editable_figure_library.py
```

必要libraryは `python-pptx`, `openpyxl`, `lxml`。generatorは `raw_data/`, `result/`, `docs/thesis/` のcanonical sourceを読み、主要値の照合に失敗すると停止する。

PPTXとXLSXはzip entry timestampと `docProps/core.xml` の日時を固定値へ正規化してから書き出すため、同一interpreterでの再生成はbyte-identicalになる。

## Layout and OOXML invariants

generatorは出力後に次を自己検証し、違反があれば停止する。

- 方向付きconnectorのarrowheadは終点側 (`a:tailEnd`)。無方向は `line()`、双方向は `bi_arrow()` を使い分ける。GH200のhost/HBM間migrationは双方向で描く。
- 全labelはgenerator入力の `
` を標準DrawingMLの `<a:br/>` へ変換し、各行がbox内幅へ収まる (Arial互換advance widthで検査)。`<a:t>` 内のliteral LFやPowerPoint側の再折返しには依存しない。
- chartの `c:axId` / `c:crossAx` は正のunsigned integerで、chart内の参照が一致し、chart間で衝突しない。
- slide外objectなし、raster画像なし、12 pt未満の表示文字なし、non-ASCII表示文字なし。

Slide 9のvalue axisは実際の底10対数軸 (`c:logBase`, 10--10000 s)。Slide 10のparity 1.0xとSlide 12の1.0x基準線は、floating shapeではなく値1.0の定数line seriesで、chartをresizeしてもY=1.0へ追従する。Slide 5のCSRは1個のみ配置し、両streamから参照する構図とする (batchはsourceを分割するのであってgraphを分割しない)。

## Editing in PowerPoint

PowerPointでPPTXを開き、図形、connector、text box、chart、legend、axis、data labelを個別に選択して移動・再着色・文字編集できる。chartは右クリックして **Edit Data** を選ぶとembedded workbookを編集できる。Slide 13の成功点・OOM記号はnative shape/lineであり個別編集する。

boxを縮めたりfont sizeを上げたりすると上記の改行不変条件は崩れるため、恒久的な変更はgeneratorへ反映して再生成し、自己検証を通すこと。

論文または発表へ図を移す場合は、対象slideで `Ctrl+A` → `Ctrl+C` を行い、宛先へ **Keep Source Formatting** で貼り付ける。captionはPPTXに含めず、論文側のcaption/label機構で付与する。

## Slide map

`Figure` 列が本文の番号、`Result ID` 列が canonical result figure との対応である
（`not_applicable` は対応する result figure が存在しない conceptual 図）。

| Slide | Library ID | Figure library title | Figure | Result ID |
|---:|---|---|---|---|
| 1 | F01 | Thesis Overview | Figure 1.1 | not_applicable |
| 2 | F02 | Brandes Algorithm Flow | Figure 2.1 | not_applicable |
| 3 | F03 | GH200 Memory Hierarchy | Figure 2.2 | not_applicable |
| 4 | F04 | Overall GPU Execution Framework | Figure 4.1 | not_applicable |
| 5 | F05 | Batch-to-Source Mapping | Figure 4.2 | not_applicable |
| 6 | F06 | Hybrid BFS State Transition | Figure 4.3 | not_applicable |
| 7 | F07 | Dual-Stream Timeline | Figure 4.4 | not_applicable |
| 8 | F08 | Memory Management Variants | Figure 4.5 | not_applicable |
| 9 | F09 | Main Runtime Comparison | Figure 6.1 | F1 |
| 10 | F10 | Speedup over Tuned PathMerge | Figure 6.2 | F2 |
| 11 | F11 | PathMerge Batch Sweep | Figure 6.3 | F3 |
| 12 | F12 | Ablation Contributions | Figure 7.1 | F4 |
| 13 | F13 | Memory Scalability | Figure 8.1 | F5 |
| 14 | F14 | Shared vs Block Kernel | Figure 7.2 | F6 |
| 15 | F15 | Phase Breakdown | Figure 7.3 | F7 |

## Canonical data policy

主要データは `result/tables/thesis/T2_main_performance.tsv`, `T3_ablation_summary.tsv`, `T4_memory_scalability.tsv` を監査表として照合し、計算には対応するraw trial TSV/logを使う。PathMerge sweepは `raw_data/tuning/pathmerge/`、kernel比較は `raw_data/tuning/kernel_selection/`、phase breakdownは `raw_data/main_performance/proposed_variants/*/_run/job_2357334_20260711/phase_timing.log` がsourceである。修正版325557だけをcurrent resultとして使う。

`figure_data.xlsx` やPowerPointのembedded chart dataを手作業で変更すると、canonical sourceからの再現性は失われる。PowerPoint上の図形編集もgenerator出力との差分になる。再生成すると手作業変更は上書きされるため、必要な変更は別名保存するかgeneratorへ反映する。

## Export

conceptual 図 (`F01`--`F08`) の組版用 asset は、本PPTXを唯一の編集正本として
`scripts/export_conceptual_figures.py` が機械的に生成する。exporterはPPTX package
を直接読み、slideのshape・connector・text runをSVGへ描き起こすため、ライブラリと
export assetが食い違うことはない。

```bash
cd thesis_bc_project
PYTHONPATH=/tmp/gate_v0_editable_deps39 python3 scripts/export_conceptual_figures.py
```

出力は `../exported/` に `figure_<章>_<番号>_<slug>.{svg,pdf,png}` として置かれ、
SVG/PDFはvector（PDFはfont subset埋込み）、PNGは300 dpiのreview用である。詳細と
tool版数は `../exported/README.md` を参照。

chart 図 (`F09`--`F15`) はexport対象ではない。論文本体へ掲載するのは
`scripts/generate_thesis_artifacts.py` が生成する canonical result figure
(`result/figures/thesis/`) である。

PowerPointから手動で書き出す場合は、対象slideの全objectを選択して右クリック
**Save as Picture** (SVG/PNG)、PDFは **File > Export > Create PDF/XPS** を使う。
LibreOfficeでは `soffice --headless --convert-to pdf thesis_figure_library.pptx`
が使えるが、font/connector/chartの差を目視確認すること。手動書き出しは正式assetの
生成手段ではなく、確認用途に限る。
