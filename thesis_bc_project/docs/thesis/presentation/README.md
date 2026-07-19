# Master's Thesis Presentation (Gate V1.1 draft)

修士論文発表用スライドである。**スライド面に表示される文字はすべて英語**であり、
日本語は speaker notes にのみ存在する。正本は論文本文（`../writing/japanese/`）と
編集可能図ライブラリ（`../figures/editable/`）であり、本ディレクトリはそれらから
決定的に生成される。

## Files

- `editable/master_thesis_presentation_v1.pptx`: 16:9、白背景の発表スライド。本編 15 枚 + Backup 7 枚。表示文字はすべて英語。
- `presentation_plan.md`: narrative、時間配分、slide map、言語・デザイン規則、主張の限定。
- `speaker_notes_bilingual.md`: スライドごとの想定時間・英語スクリプト・日本語説明・transition・限定・想定質問。
- `PRESENTATION_MANIFEST.tsv`: スライド番号、section、タイトル、narrative purpose、図番号、オブジェクト種別、canonical source。
- `../../../scripts/generate_thesis_presentation.py`: 再生成器。

## Presentation length

**発表時間 15 分は暫定値である。** リポジトリ内に公式の発表時間指定は存在しない。
本編の想定時間合計は 900 秒（15.0 分）である。

20 分などへ変更する場合は、`scripts/generate_thesis_presentation.py` の `TALK_MINUTES`
と `NOTES` の想定秒数を更新して再生成する。スライド構成自体は増減可能な形で分離してある。

## Bilingual speaker notes

各スライドは完全な英語スクリプトと完全な日本語説明の両方を持つ。二つは**代替**であり、
発表言語に応じてどちらか一方だけを読む。両方を続けて読むことは想定していない。

- 英語版のみを読んだ場合の本編推定合計: 約 802 秒
- 日本語版のみを読んだ場合の本編推定合計: 約 830 秒
- Backup 7 枚（合計 280 秒）は本編合計に含まない

推定は英語 2.5 words/秒、日本語 5.5 文字/秒による計画値であり、実測ではない。

## Regeneration

```bash
cd thesis_bc_project
PYTHONPATH=<deps> python3 scripts/generate_thesis_presentation.py
```

必要 library は `python-pptx`, `openpyxl`, `lxml`。生成器は
`scripts/generate_editable_figure_library.py` の helper・palette・data loader・
package normalizer を import して再利用するため、図ライブラリと配色・書体・数値が
乖離しない。canonical 値の照合に失敗した場合は生成を停止する。

PPTX は zip entry timestamp と `docProps/core.xml` の日時を固定値へ正規化するため、
同一 interpreter での再生成は byte-identical になる。

## Editability

- raster-only slide: 0
- 埋め込み raster 画像: 0
- native chart: 6（embedded workbook あり）
- native table: 6
- 編集可能オブジェクト総数: 257

すべての図は native shape / connector / text box / chart / table であり、
PowerPoint 上で個別に選択・移動・再着色・文字編集できる。chart は右クリック
**Edit Data** で embedded workbook を編集できる。図を PNG として貼り付けた箇所はない。

## Self-validation

生成器は出力後に次を検査し、違反があれば停止する。

- スライド面（`ppt/slides`・`ppt/charts`・`ppt/diagrams`・`ppt/embeddings`）に仮名・漢字が 1 文字も存在しないこと。notes slide は検査対象外。
- 表示文字が ASCII、または明示的に許可した約物（×–—）のみであること。
- 全 22 notes slide に `[English Script]` と `[日本語説明]` の双方があり、英語側に日本語が混入せず、日本語側に日本語が存在すること。
- 各スクリプトが宣言した想定秒数に対して妥当な長さであること（一方が他方の要約になっていないこと）。
- 全 shape の各行が、指定 font size で box 内幅に収まること（途中改行の防止）。
- 表示文字（表セルを含む）が全スライドで 16 pt 以上であること。
- 主張の限定文（PathMerge・working set・容量上限・Tier A/B・適用範囲）が字句どおり残存すること。
- 表セルの文字が列幅に収まり、行の自動伸長による枠外はみ出しを起こさないこと。
- slide 境界外の object が存在しないこと。
- raster 画像が存在しないこと。
- chart の `c:axId` / `c:crossAx` が正の整数で、chart 間で衝突しないこと。
- 主要値（speedup、tuned batch、H/W/A、容量境界、correctness tier 件数）が canonical source と一致すること。

## Human review still required

本環境に PowerPoint / LibreOffice renderer は存在しないため、OOXML・幾何・text-fit
検査のみを実施しており、**実際のレンダリング目視は未実施**である。実機 PowerPoint で
次を確認すること。

- Slide 1 の `[TO BE FILLED]`（Name・Affiliation・Supervisor・Date）の記入。
- chart の data label と凡例の実描画（Slide 8, 9, 10, 18, 19, 20）。
- 2 行に分割した長いタイトル（Slide 9, 12, 15）の折り返し位置と title rule との間隔。
- Slide 11 の feasibility marker と注記の重なり。
- Slide 12 および Backup（Slide 16-22）の表の可読性。
- notes pane に英語・日本語の双方が正しく表示されること。
