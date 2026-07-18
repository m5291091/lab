# Editable Thesis and Presentation Figure Library

このディレクトリは、修士論文と発表資料で再利用する **1 slide = 1 figure** の編集用ライブラリである。caption と slide title は図へ焼き込まず、論文・story deck 側で管理する。既存の正式 F1--F7 は置換しない。

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
- 全labelは `
` で明示的に改行済みで、各行がbox内幅へ収まる (Arial互換advance widthで検査)。PowerPoint側の再折返しによる単語途中改行は発生しない。
- chartの `c:axId` / `c:crossAx` は正のunsigned integerで、chart内の参照が一致し、chart間で衝突しない。
- slide外objectなし、raster画像なし、12 pt未満の表示文字なし、non-ASCII表示文字なし。

Slide 9のvalue axisは実際の底10対数軸 (`c:logBase`, 10--10000 s)。Slide 10のparity 1.0xとSlide 12の1.0x基準線は、floating shapeではなく値1.0の定数line seriesで、chartをresizeしてもY=1.0へ追従する。Slide 5のCSRは1個のみ配置し、両streamから参照する構図とする (batchはsourceを分割するのであってgraphを分割しない)。

## Editing in PowerPoint

PowerPointでPPTXを開き、図形、connector、text box、chart、legend、axis、data labelを個別に選択して移動・再着色・文字編集できる。chartは右クリックして **Edit Data** を選ぶとembedded workbookを編集できる。Slide 13の成功点・OOM記号はnative shape/lineであり個別編集する。

boxを縮めたりfont sizeを上げたりすると上記の改行不変条件は崩れるため、恒久的な変更はgeneratorへ反映して再生成し、自己検証を通すこと。

論文または発表へ図を移す場合は、対象slideで `Ctrl+A` → `Ctrl+C` を行い、宛先へ **Keep Source Formatting** で貼り付ける。captionはPPTXに含めず、論文側のcaption/label機構で付与する。

## Slide map

| Slide | Figure | Figure library title | Thesis use |
|---:|---|---|---|
| 1 | F01 | Thesis Overview | Chapter 1 |
| 2 | F02 | Brandes Algorithm Flow | Chapter 2 |
| 3 | F03 | GH200 Memory Hierarchy | Chapter 3/4/8 |
| 4 | F04 | Overall GPU Execution Framework | Chapter 4 |
| 5 | F05 | Batch-to-Source Mapping | Chapter 4 |
| 6 | F06 | Hybrid BFS State Transition | Chapter 4 |
| 7 | F07 | Dual-Stream Timeline | Chapter 4 |
| 8 | F08 | Memory Management Variants | Chapter 4/8 |
| 9 | F09 | Main Runtime Comparison | Chapter 6 |
| 10 | F10 | Speedup over Tuned PathMerge | Chapter 6 |
| 11 | F11 | PathMerge Batch Sweep | Chapter 6 / Appendix B |
| 12 | F12 | Ablation Contributions | Chapter 7 / Appendix C |
| 13 | F13 | Memory Scalability | Chapter 8 |
| 14 | F14 | Shared vs Block Kernel | Chapter 7 |
| 15 | F15 | Phase Breakdown | Chapter 6/7 |

## Canonical data policy

主要データは `result/tables/thesis/T2_main_performance.tsv`, `T3_ablation_summary.tsv`, `T4_memory_scalability.tsv` を監査表として照合し、計算には対応するraw trial TSV/logを使う。PathMerge sweepは `raw_data/tuning/pathmerge/`、kernel比較は `raw_data/tuning/kernel_selection/`、phase breakdownは `raw_data/main_performance/proposed_variants/*/_run/job_2357334_20260711/phase_timing.log` がsourceである。修正版325557だけをcurrent resultとして使う。

`figure_data.xlsx` やPowerPointのembedded chart dataを手作業で変更すると、canonical sourceからの再現性は失われる。PowerPoint上の図形編集もgenerator出力との差分になる。再生成すると手作業変更は上書きされるため、必要な変更は別名保存するかgeneratorへ反映する。

## Export

PowerPointでは対象slideの全objectを選択し、右クリック **Save as Picture** からSVG/PNGを保存できる。PDFは **File > Export > Create PDF/XPS** を使う。LibreOfficeでは `soffice --headless --convert-to pdf thesis_figure_library.pptx` を使用できるが、font/connector/chartの差を目視確認すること。本Gate環境にはPowerPoint/LibreOffice rendererがないため、不完全な自動SVG/PDF/PNG exportは生成していない。
