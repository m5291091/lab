# Exported Thesis Figures (conceptual figures F01--F08)

論文本体へ掲載する概念図の組版用 asset。**編集正本は
`../editable/thesis_figure_library.pptx`** であり、このディレクトリの
SVG/PDF/PNG はすべてその PPTX から機械的に導出される。ここのファイルを直接
編集してはならない。図を変更する場合は
`scripts/generate_editable_figure_library.py` を修正して PPTX を再生成し、
続いて `scripts/export_conceptual_figures.py` を実行する。

## Figure ID namespace

`F01`--`F15` は **editable library** の ID であり、
`result/figures/thesis/` の canonical result figure `F1`--`F7` とは
**別の namespace** である。`F01` は `F1` ではない。論文本文で参照する番号は
`ThesisFigureNumber`（Figure 1.1 など）であり、library ID は本文に出さない。
対応表は `../editable/FIGURE_MANIFEST.tsv` と `EXPORT_MANIFEST.tsv` にある。

## Formats

| 形式 | 用途 | 性質 |
|---|---|---|
| `.svg` | Word / Web 貼り込み | vector、テキストは `<text>` のまま（抽出・再検索可能） |
| `.pdf` | 最終組版 | vector、font subset 埋め込み（`/FontFile3`）、raster image なし |
| `.png` | review 用 preview | 300 dpi 相当。**組版には使わない** |

caption と Figure 番号は画像へ焼き込まない。論文側の caption 機構が担当する。
背景は全図とも白で統一し、ink bounding box に 9 pt の余白を付けて
crop している（slide 全面の余白帯は含めない）。

## Export tool and command

```text
rsvg-convert : rsvg-convert version 2.50.7
python       : 3.12.8
source pptx  : docs/thesis/figures/editable/thesis_figure_library.pptx
```

```bash
cd thesis_bc_project
PYTHONPATH=/tmp/gate_v0_editable_deps python3.12 scripts/export_conceptual_figures.py
```

interpreter は editable library generator と同じ Python 3.12 系を用いる
（`generate_editable_figure_library.py` の `figure_data.xlsx` は
`statistics.stdev` の実装差で Python 3.9 系だと sample SD 2 セルが 1 ULP
ずれるため、両 script とも 3.12 で実行する）。

`PYTHONPATH` は `python-pptx` を含む依存 set を指す。exporter 自身は PPTX を
`zipfile` + `xml.etree` で読むため python-pptx を必要としないが、Arial の
advance width table を `generate_editable_figure_library.py` から import して
PPTX 側の layout guard と同一の値を使うため、同 module が import 可能である
必要がある。

## Determinism

SVG は固定小数点で書き出し timestamp を持たない。`rsvg-convert` が PDF へ
書き込む `/CreationDate` は固定値へ正規化する（同一 byte 長のため xref offset
は不変）。連続 2 回の実行は全 asset が byte-identical になる。

## Renderer approximation (要人間確認)

- DrawingML の `type="triangle" w="sm" len="sm"` arrowhead は、線幅の
  3x（長さ）/ 1.5x（底辺幅）の三角形として描画している。PowerPoint の
  内部係数は公開されていないため、これは見た目を合わせた近似である。
- PPTX の複数行 label は標準DrawingMLの `<a:br/>` で表現される。本 exporter は
  `<a:t>`、`<a:br/>`、複数 `<a:p>`、paragraph alignment、およびrun別font
  propertiesを解釈する。旧PPTXとの比較用に `<a:t>` 内literal LFも読み取れるが、
  正式生成物はliteral LFへ依存しない。
- font は Arial を第一候補とし、Linux 上では metric 互換の Nimbus Sans /
  Liberation Sans へ fallback する。

## Files

- `figure_1_1_thesis_overview.svg` / `figure_1_1_thesis_overview.pdf` / `figure_1_1_thesis_overview.png` -- Figure 1.1 Thesis Overview (library F01, slide 1, 12.61 x 6.52 in)
- `figure_2_1_brandes_algorithm.svg` / `figure_2_1_brandes_algorithm.pdf` / `figure_2_1_brandes_algorithm.png` -- Figure 2.1 Brandes Algorithm Flow (library F02, slide 2, 13.22 x 3.61 in)
- `figure_2_2_gh200_memory_hierarchy.svg` / `figure_2_2_gh200_memory_hierarchy.pdf` / `figure_2_2_gh200_memory_hierarchy.png` -- Figure 2.2 GH200 Memory Hierarchy (library F03, slide 3, 12.72 x 6.71 in)
- `figure_4_1_gpu_execution_framework.svg` / `figure_4_1_gpu_execution_framework.pdf` / `figure_4_1_gpu_execution_framework.png` -- Figure 4.1 Overall GPU Execution Framework (library F04, slide 4, 11.47 x 6.44 in)
- `figure_4_2_batch_source_mapping.svg` / `figure_4_2_batch_source_mapping.pdf` / `figure_4_2_batch_source_mapping.png` -- Figure 4.2 Batch-to-Source Mapping (library F05, slide 5, 13.05 x 6.52 in)
- `figure_4_3_hybrid_bfs.svg` / `figure_4_3_hybrid_bfs.pdf` / `figure_4_3_hybrid_bfs.png` -- Figure 4.3 Hybrid BFS State Transition (library F06, slide 6, 12.47 x 5.75 in)
- `figure_4_4_dual_stream_timeline.svg` / `figure_4_4_dual_stream_timeline.pdf` / `figure_4_4_dual_stream_timeline.png` -- Figure 4.4 Dual-Stream Timeline (library F07, slide 7, 12.59 x 6.70 in)
- `figure_4_5_memory_management_variants.svg` / `figure_4_5_memory_management_variants.pdf` / `figure_4_5_memory_management_variants.png` -- Figure 4.5 Memory Management Variants (library F08, slide 8, 11.96 x 6.61 in)
