# code_snapshots/_legacy_audit — 旧 commit 監査用（raw 取得・再生成には不要）

このディレクトリは **raw データの取得・再生成には不要な旧履歴対応表** のみを保持する。

- `LEGACY_COMMIT_TO_SNAPSHOT.tsv` — 過去の Git commit SHA と `SourceSnapshotID` の対応表（移行監査用）。

## 位置づけ

raw データ・実験時コードへの **正式なアクセス経路** は commit SHA に依存しない:

- raw: `RawPath` / `SHA256` / `PBSJobID`（`../../raw_data/RAW_DATA_INDEX.tsv`, `MANIFEST.tsv`）
- コード: `SourceSnapshotID`（`../<SourceSnapshotID>/` = 実験時コードの凍結コピー）+ `SOURCE_MANIFEST.tsv`

commit SHA は以下でのみ許可される:
1. 内容不変の生ログ（`raw_data/**/pbs_stdout*.log`, `MANIFEST.txt` の `checkpoint_sha` 等）
2. 過去履歴との移行監査表（この `LEGACY_COMMIT_TO_SNAPSHOT.tsv`）

formal な README / SOURCE / CLAIMS / COVERAGE / TABLES / 実行スクリプトは commit SHA を
**必須参照にしない**（すべて `SourceSnapshotID` / `RawPath` / `SHA256` / `PBSJobID` で完結する）。
Git 履歴を削除しても raw 取得・主要結果の再生成は可能。
