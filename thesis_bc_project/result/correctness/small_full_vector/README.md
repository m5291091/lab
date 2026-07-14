# Small full-vector correctness

Sequential を独立参照、GPU_Opt を candidate として、3 個の小規模グラフの全 BC 要素を比較した正確性検証である。checkpoint は `small_correctness_20260712`、PBS job は `2367583.opbs`。各構成 n=1、warmup なしであり、記録された時間値は性能評価・性能主張に使用しない。

判定は `abs_tol=1e-3`、`rel_tol=1e-6` とし、各 index で次を満たすことを要求した。

```text
abs(reference - candidate) <= abs_tol + rel_tol * max(abs(reference), abs(candidate))
```

## Results

| Graph | n / m | Length ref/cand | Missing ref/cand | Mismatch | NaN/Inf ref/cand | Max abs error (index; ref, cand) | Max rel error (index; ref, cand) | Max BC ref/cand |
|:--|:--|:--|:--|--:|:--|:--|:--|:--|
| benchmark_7000_41459 | 7000 / 41459 | 7000 / 7000 | 0 / 0 | 0 | 0 / 0 | 6.053597e-09 (0; 2549196.725646447, 2549196.725646441) | 4.563145e-15 (1186; 11161.53593043016, 11161.53593043011) | index 4, value 3935437.257858 / index 4, value 3935437.257858 |
| benchmark_11023_62184 | 11023 / 62184 | 11023 / 11023 | 0 / 0 | 0 | 0 / 0 | 2.980232e-08 (10; 11951000.93285756, 11951000.93285759) | 1.789722e-14 (3092; 11789.69389924664, 11789.69389924643) | index 10, value 11951000.932858 / index 10, value 11951000.932858 |
| chain_200 | 200 / 199 | 200 / 200 | 0 / 0 | 0 | 0 / 0 | 0.000000e+00 (0; 0.0, 0.0) | 0.000000e+00 (0; 0.0, 0.0) | index 99, value 9900.000000 / index 99, value 9900.000000 |

全 runner exit、comparison exit は 0、全行 `PASS (mixed_tolerance_mismatches_0)`。requested/effective batch は 512/512、`SUB_BATCH=512`、`num_subs=1`、`NS_eff=2`（GPU_Opt stderr 実測）である。

## Scope

この結果が支持するのは上記 3 グラフの Sequential vs GPU_Opt のみである。email-EuAll と roadNet-PA/TX/CA の独立参照 full-vector 比較、GPU_Opt_Pure、GPU_Opt_Pure_Chunked、UM oversubscription 固有経路は未検証である。また、Hybrid BFS、warp 等の個別経路が通過したことを専用カウンタで検証した結果ではない。

raw BC vector は Git 管理外の `build_miyabi/result_small_correctness_20260712_181140_2367583.opbs/<graph>/{sequential,gpu_opt}.bc.tsv` に保持し、個別の size/SHA256 は `../../EXTERNAL_ARTIFACTS.tsv` に記録する。詳細な provenance と再現方法は `SOURCE.md` を参照。
