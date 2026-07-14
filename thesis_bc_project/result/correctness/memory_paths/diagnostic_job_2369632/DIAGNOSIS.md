# DIAGNOSIS — T-RESET / T-NSEFF (checkpoint 43d1cf5542f3234dddc93c88c5fdd72761f52271)

- non_interference (CONTROL vs old_b1024): verified_mismatch0
- CONTROL vs T-RESET mixed-tol mismatch: 0
- CONTROL vs T-NSEFF mixed-tol mismatch: 0

## reset judgment: **RESET_NOT_DISTINGUISHED**
control_vs_cand_mismatch0_within_mixed_tol

## NS_eff judgment: **NS_EFF_NOT_DISTINGUISHED**
control_vs_cand_mismatch0_within_mixed_tol

注: mismatch=0 は「事前設定した混合許容内で一致」であり bitwise/SHA256 一致ではない。
PathMerge は本診断に含めない。許容値は不変で FAIL を PASS に書き換えない。
「近づく」は affected 8 頂点の値/誤差距離/mismatch 集合で定量判断 (affected_vertices.tsv 参照)。
判定名は非因果 (ASSOCIATED): RESET_PATH_OR_SCHEDULING_ASSOCIATED は full memset に伴う実行
タイミング・GPU スケジューリング・atomicAdd 順序を、NS_EFF_OR_OCCUPANCY_ASSOCIATED は stream
数・occupancy・atomic 順序を含む「関連」であり、reset 内容/stream 数の単独原因断定ではない。
