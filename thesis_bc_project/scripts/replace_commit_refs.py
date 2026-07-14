#!/usr/bin/env python3
"""ナラティブ/SOURCE メタデータ内の commit SHA 正式参照を SourceSnapshotID へ置換する。
- 生証跡コピー(PRESERVE)・索引TSV・provenance.md・environment.md は対象外(別処理)。
- 64-hex の SHA256 データ値は word-boundary により不変（commit SHA は 7-8/40 hex）。
"""
import os, re, sys

TBP = "/work/gj17/j17000/m5291091/lab/thesis_bc_project"

FULL = {
 "e32b03e9b73e9eb294685c58e488ce2a92521852":"small_correctness_20260712",
 "88faffa391026852a4440e5b9a063c08c29624f7":"phase_def_block_20260710",
 "ac2b409c25c49c41608749afba8c7081871bfe45":"memory_correctness_20260712",
 "43d1cf5542f3234dddc93c88c5fdd72761f52271":"memory_diagnostic_20260713",
 "6282798ce9942c6297cbdf2963aa7a3c65c6b807":"memory_correctness_oom_20260712",
 "29d28c50dec5e70f8d3a9a2341904e1ee94c65f3":"memory_correctness_failfast_20260712",
 "f05ec52ae657df40224f624e30f9cc78aaa3bd48":"oldtree_f05ec52_20260512",
}
SHORT = {
 "e32b03e9":"small_correctness_20260712",
 "88faffa":"phase_def_block_20260710",
 "ac2b409":"memory_correctness_20260712",
 "43d1cf5":"memory_diagnostic_20260713",
 "6282798":"memory_correctness_oom_20260712",
 "29d28c50":"memory_correctness_failfast_20260712",
 "f05ec52":"oldtree_f05ec52_20260512",
}
COMPOSITE = {
 "88faffa(2026-07-10)":"phase_def_block_20260710",
 "old-tree(f05ec52-era)":"oldtree_f05ec52_20260512",
 "f05ec52(旧tree測定)":"oldtree_f05ec52_20260512",
}

# 特殊ケース: 完全一致で置換（正規表現前）
SPECIAL = {
 "result/correctness/small_full_vector/SOURCE.md":[
   ("- Checkpoint: `e32b03e9b73e9eb294685c58e488ce2a92521852` (`EXPECTED_SHA=ACTUAL_SHA`).",
    "- SourceSnapshotID: `small_correctness_20260712`（実験時コード `code_snapshots/small_correctness_20260712/`；元 commit は `code_snapshots/_legacy_audit/LEGACY_COMMIT_TO_SNAPSHOT.tsv`）。runner は `EXPECTED_SHA=ACTUAL_SHA` を確認。"),
   ('CHECKPOINT_SHA="$(git rev-parse HEAD)"\ntest "$CHECKPOINT_SHA" = "e32b03e9b73e9eb294685c58e488ce2a92521852"\nqsub -v EXPECTED_SHA="$CHECKPOINT_SHA",BC_BATCH_OVERRIDE=512 \\\n  scripts/run_small_correctness.sh',
    '# 実験時コード = code_snapshots/small_correctness_20260712/（元 commit は _legacy_audit/LEGACY_COMMIT_TO_SNAPSHOT.tsv）\nqsub -v BC_BATCH_OVERRIDE=512 scripts/run_small_correctness.sh'),
 ],
 "result/ablation/synthetic_2354994/SOURCE.md":[
   ("- **checkpoint**: `88faffa`（測定 2026-07-10, 常時block化 `1ae987c` 後）",
    "- **SourceSnapshotID**: `phase_def_block_20260710`（測定 2026-07-10, 常時 block 化後）"),
 ],
}

FILES = [
 "docs/kernel_selection_decision.md",
 "docs/thesis/00_thesis_positioning.md","docs/thesis/01_research_questions.md",
 "docs/thesis/05_experimental_setup.md","docs/thesis/07_results_ablation.md",
 "docs/thesis/08_results_memory.md","docs/thesis/09_results_correctness.md",
 "docs/thesis/11_limitations.md","docs/thesis/README.md",
 "docs/thesis/evidence_matrix.tsv","docs/thesis/thesis_values.tsv",
 "failure/README.md",
 "result/CLAIMS.md","result/COVERAGE.md","result/MANIFEST.md","result/README.md",
 "result/ablation/email_2354999/SOURCE.md","result/ablation/synthetic_2354994/SOURCE.md",
 "result/correctness/memory_paths/README.md","result/correctness/memory_paths/SOURCE.md",
 "result/correctness/memory_paths/analysis/Gate_G2_3_audit.md",
 "result/correctness/pathmerge_tuned/README.md","result/correctness/pathmerge_tuned/SOURCE.md",
 "result/correctness/pathmerge_tuned/email-EuAll_b64_vs_b2048.md",
 "result/correctness/pathmerge_tuned/roadNet-CA_b32_vs_b64.md",
 "result/correctness/small_full_vector/README.md","result/correctness/small_full_vector/SOURCE.md",
 "result/main_performance/proposed_variants/SOURCE.md",
 "result/main_performance/proposed_vs_pathmerge/README.md",
 "result/memory_scalability/SOURCE.md","result/phase_breakdown/SOURCE.md",
 "result/profiling/SOURCE.md","result/provenance/um_code_diff_audit.md",
 "result/tuning/kernel_selection/SOURCE.md","result/tuning/pathmerge/SOURCE.md",
 "result/tuning/pathmerge/roadNet-CA/SOURCE.md","result/tuning/pathmerge/roadNet-TX/SOURCE.md",
]

def wb_sub(text, token, repl):
    # word-boundary 置換（64-hex SHA256 内部は境界なしで不一致→保全）
    return re.sub(r'(?<![0-9a-zA-Z])'+re.escape(token)+r'(?![0-9a-zA-Z])', repl, text)

DRY = "--apply" not in sys.argv
total=0; per=[]
for rel in FILES:
    fp=os.path.join(TBP,rel)
    if not os.path.isfile(fp):
        print("MISSING",rel); continue
    with open(fp,encoding='utf-8') as f: t=f.read()
    orig=t
    for a,b in SPECIAL.get(rel,[]):
        if a in t: t=t.replace(a,b)
        else: print(f"  [warn] special block not found in {rel}")
    for a,b in COMPOSITE.items(): t=t.replace(a,b)
    for a,b in FULL.items(): t=wb_sub(t,a,b)
    for a,b in SHORT.items(): t=wb_sub(t,a,b)
    if t!=orig:
        n=sum(1 for _ in re.finditer('|'.join(re.escape(x) for x in list(FULL.values())+["small_correctness_20260712"]) , t))
        per.append((rel, len(orig), len(t)))
        if not DRY:
            with open(fp,'w',encoding='utf-8') as f: f.write(t)
        total+=1
print(f"MODE={'DRY' if DRY else 'APPLY'} changed_files={total}")
for r,a,b in per: print(f"  {r}  ({a}->{b} bytes)")

# 残存 commit SHA スキャン
print("\n=== residual target commit-SHA tokens after replace ===")
resid=0
for rel in FILES:
    fp=os.path.join(TBP,rel)
    with open(fp,encoding='utf-8') as f: t=f.read()
    for tok in list(SHORT)+["4b41eab","1ae987c"]:
        for m in re.finditer(r'(?<![0-9a-zA-Z])'+re.escape(tok)+r'(?![0-9a-zA-Z])', t):
            resid+=1; print(f"  {rel}: {tok}")
            break
print("residual:",resid,"(0 desired for target checkpoints; 4b41eab/1ae987c only if intended)")
