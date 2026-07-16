# 実行環境

| 項目 | 値 |
|:--|:--|
| 実験時コード（提案 block・新規 PathMerge 測定） | SourceSnapshotID `phase_def_block_20260710`（`../../code_snapshots/phase_def_block_20260710/`；監査用元 commit は `../provenance/provenance.md`） |
| GPU | NVIDIA GH200 |
| 公称 HBM3 | 96 GB |
| 記録されたデバイスメモリ | 97,871 MiB（約95.6 GiB、約102.6 decimal GB；公称96 GBと同一のHBM3） |
| 実行開始時の runtime 照会 | total 約102.0 GB、free (`free_before`) 約101.4 GB（decimal GB；freeは総容量ではなくメモリ予算計算の基準） |
| NVIDIA driver | 595.58.03 |
| CUDA (nvcc) | release 13.0, V13.0.48 |
| CMake | 4.3.4（`~/.local/bin/cmake`） |
| C++ コンパイラ | g++ (GCC) 11.4.1 |
| nsys | 2025.5.1.121 |
| PBS system | Miyabi-G PBS batch system |
| Group | `gj17` |
| Queue | Not independently verifiable from retained job logs |
| memory-path実験の資源構成 | Host-memory-limited 100 GiB configuration |

公称96 GB、記録値97,871 MiB（約95.6 GiB、約102.6 decimal GB）、runtime照会のtotal約102.0 GBは、同一のオンパッケージHBM3容量を異なる単位系または取得方法で示したものであり、別個のメモリ階層ではない。free約101.4 GBは実行開始時の利用可能量であり、総容量ではない。

実験はMiyabi-G上のPBS batch systemを通じてgroup `gj17`で実行した。保存された従来の正式文書と投入スクリプトの間でqueue名の記録が一致せず、保存済みジョブログから実際のqueue名を独立に確定できないため、queue名は本評価の統制変数として扱わない。100 GiBはqueue名、submission resource limit、node configurationのいずれかとは断定せず、host-memory-limited configurationとしてのみ記録する。

## 出典（SourceSnapshotID 別）
`result/` 全体が単一スナップショットではない。実験ごとに実験時コードが異なる（正式参照 = SourceSnapshotID）:
| 実験群 | SourceSnapshotID | 備考 |
|:--|:--|:--|
| 提案 block 再計測（proposed_variants）・kernel_selection・PathMerge 掃引・correctness・profiling | **`phase_def_block_20260710`** | 常時 block 化後 |
| legacy baseline（seven_implementations, PathMerge 既定 b64 / 旧 shared 提案） | **`oldtree_f05ec52_20260512`** | 旧 mylab/research 由来（近似・pre-consolidation） |
| UM オーバーサブスクリプション（memory_scalability） | **`oldtree_f05ec52_20260512`**（2026-05-12, 旧 tree） | 時間値は非採用（下記） |
| ablation（synthetic_2354994 / email_2354999） | `phase_def_block_20260710`（測定 2026-07-10, 常時block化後） | build_miyabi から curate |

## 副次(B) UM 実験の測定環境（旧ツリー）
- 実験時コード: SourceSnapshotID `oldtree_f05ec52_20260512`（2026-05-12, 旧 mylab/research ツリー）
- 同一 GH200 系。メモリロジックは `phase_def_block_20260710` と文字単位同一（`../provenance/um_code_diff_audit.md`）。
- **時間値は最新 block 性能値としては非採用**（旧セッション未再検証）。旧実験で観測された、試験範囲内の SUCCESS/OOM 傾向を限定的な feasibility 根拠として再利用する。ただし `phase_def_block_20260710` で同じ境界を再実測したものではない。

## 集計規約
- Aggregation = median（中央値）
- **Warmup**:
  - 新規測定（proposed_variants / kernel_selection / PathMerge 掃引 / correctness, SourceSnapshotID phase_def_block_20260710）= **なし**（ベンチスクリプトは全 TRIALS を記録・discard なし。スクリプトで確認）。
  - legacy baseline・UM（旧 tree）= 当時ログ準拠（明示的 warmup 記録なし → `not_recorded`）。
- TimingScope = runner が BC 計算全体（host 制御 + カーネル）を Time_sec として stdout 出力。phase 内訳(BFS/Backward)は stderr。
