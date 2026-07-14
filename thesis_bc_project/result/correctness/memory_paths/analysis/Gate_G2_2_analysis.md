# Gate G2.2 分析 (read-only, raw値無補正; 正式判定 abs_tol=1e-3 rel_tol=1e-6)
graph n=325557  graph_sha256=a095b2e7564e6c620bd0f5437917e0b28f4fecab289adf77633e850aa07da584

## vector SHA256 と length/NaN,Inf
  pm_269        len=325557 nonfinite=0 sha256=1569b9e341f1baecaca010aa36f01d1b9bc1e97530a6156de3d6a9874acf9f84
  pm_398        len=325557 nonfinite=0 sha256=c895c6121671af6605d834e68bc1443b84624ca595506267901c3a59a643e454
  pm_587        len=325557 nonfinite=0 sha256=94e6379ac52e76025052ff98e97274a16b6467b57454d73c6d68aaa9eeeebd9d
  pure_398      len=325557 nonfinite=0 sha256=b4f0674431bc4c918b7b5a597ca98be6132c6f9ad06b2f31b300ff31f6b23e95
  pure_587      len=325557 nonfinite=0 sha256=fc95255fe422693d5f7c8d80d39624015b3f9ad2020b54916f61358718439984
  gpu_b1024     len=325557 nonfinite=0 sha256=4a40a553a388ba2cb29d4ea366db979983fa398c55bb8a694882f260efd431cb
  gpu_b9792     len=325557 nonfinite=0 sha256=be8f52d32ac03cd495a08c5c6cd138fcdcec916a16830e62e7bc8d3c968d25c5
  chunk_b1024   len=325557 nonfinite=0 sha256=5ff0bef083dff51e0c6f15894912549dd5e28bb42e5eaabf626654edbf4627de
  chunk_b16384  len=325557 nonfinite=0 sha256=618ffdc4108f0c24a148bc1aa2a18b83d14b48319cd18438f89196b40930a86d

## 3. 実行間再現性 (same impl/batch)
  [Pure_b1024 398 vs 587] byte_identical=no mismatch=0 max_abs=7.629395e-06@272817 max_rel=3.015985e-14@260562 B>A=23842 B<A=34153 verdict=within_mixed_tol
  [PathMerge 269 vs 398] byte_identical=no mismatch=0 max_abs=3.051758e-05@272817 max_rel=6.405048e-14@52821 B>A=42582 B<A=43973 verdict=within_mixed_tol
  [PathMerge 269 vs 587] byte_identical=no mismatch=0 max_abs=3.051758e-05@272817 max_rel=9.491433e-14@68170 B>A=47584 B<A=40376 verdict=within_mixed_tol
  [PathMerge 398 vs 587] byte_identical=no mismatch=0 max_abs=2.479553e-05@48187 max_rel=7.923404e-14@68170 B>A=50901 B<A=38532 verdict=within_mixed_tol

## 4. stress 直接比較 (11th, 一時のみ): GPU_Opt b9792 vs Chunked b16384
  mismatch=4 max_abs=1.000000e+00@325556 (A=894728.6570761406 B=894727.6570761404) max_rel=2.847745e-06@95156 B>A=35384 B<A=56595 verdict=exceeds(4)
  MaxBC A(b9792)=idx272817,39343001000.108543  B(b16384)=idx272817,39343001000.108582

## 5. 厳格許容超過 index 集合の比較
  S1 (gpu_b9792 vs gpu_b1024) : 6 indices = [7954, 143358, 165886, 228350, 289284, 325556]
  S2 (chunk_b16384 vs chunk_b1024): 6 indices = [95156, 143358, 165886, 226184, 228350, 289284]
  S1==S2 ? False  intersection=4 union=8
  PathMerge(pm_587) vs gpu_b1024 差 index 数=11027; union(S1,S2) との重なり=5 / 8

## 6構成の該当 index BC値
  idx 7954 (deg 15): gpu_b9792=224196.2897792182 gpu_b1024=224196.789779218 chunk_b16384=224196.789779218 chunk_b1024=224196.7897792181 pure_587=224196.7897792181 pm_587=224196.1647792153
      b9792==b16384? False  b1024(UM/Pure/Chunk)混合許容内? True  PathMerge差に含む? True
  idx 95156 (deg 17): gpu_b9792=21947.19184601125 gpu_b1024=21947.19184601126 chunk_b16384=21947.12934601126 chunk_b1024=21947.19184601126 pure_587=21947.19184601126 pm_587=21946.24068051007
      b9792==b16384? False  b1024(UM/Pure/Chunk)混合許容内? True  PathMerge差に含む? True
  idx 143358 (deg 4): gpu_b9792=325554.5000000001 gpu_b1024=325554.0000000001 chunk_b16384=325554.5000000001 chunk_b1024=325554.0000000001 pure_587=325554.0000000001 pm_587=325553.9999999964
      b9792==b16384? True  b1024(UM/Pure/Chunk)混合許容内? True  PathMerge差に含む? False
  idx 165886 (deg 3): gpu_b9792=339765.0374047094 gpu_b1024=339764.5374047094 chunk_b16384=339765.0374047093 chunk_b1024=339764.5374047094 pure_587=339764.5374047094 pm_587=339764.537404708
      b9792==b16384? False  b1024(UM/Pure/Chunk)混合許容内? True  PathMerge差に含む? False
  idx 226184 (deg 4): gpu_b9792=325705.0630518222 gpu_b1024=325705.0630518222 chunk_b16384=325704.5630518222 chunk_b1024=325705.0630518222 pure_587=325705.0630518222 pm_587=325704.5628149052
      b9792==b16384? False  b1024(UM/Pure/Chunk)混合許容内? True  PathMerge差に含む? True
  idx 228350 (deg 4): gpu_b9792=325706.0279660214 gpu_b1024=325705.5279660214 chunk_b16384=325706.0279660214 chunk_b1024=325705.5279660214 pure_587=325705.5279660214 pm_587=325705.5277291043
      b9792==b16384? True  b1024(UM/Pure/Chunk)混合許容内? True  PathMerge差に含む? False
  idx 289284 (deg 18): gpu_b9792=314545.1056143004 gpu_b1024=314544.6056143006 chunk_b16384=314545.1056143006 chunk_b1024=314544.6056143005 pure_587=314544.6056143005 pm_587=314487.4291436755
      b9792==b16384? False  b1024(UM/Pure/Chunk)混合許容内? True  PathMerge差に含む? True
  idx 325556 (deg 9): gpu_b9792=894728.6570761406 gpu_b1024=894727.1570761407 chunk_b16384=894727.6570761404 chunk_b1024=894727.1570761407 pure_587=894727.1570761405 pm_587=892937.5267641294
      b9792==b16384? False  b1024(UM/Pure/Chunk)混合許容内? True  PathMerge差に含む? True

## 6. 許容値感度分析 (abs_tol=1e-3 固定; 正式は rel_tol=1e-6, 他は補助のみ)
                                    1e-6(official)	         2e-6	         3e-6	         1e-5
  gpu_b9792 vs gpu_b1024                        6	            1	            0	            0
  chunk_b16384 vs chunk_b1024                   6	            1	            0	            0
  gpu_b9792 vs chunk_b16384                     4	            2	            0	            0
  Pure_b1024 398 vs 587                         0	            0	            0	            0
  PathMerge 398 vs 587                          0	            0	            0	            0
  PathMerge 269 vs 587                          0	            0	            0	            0
  PathMerge(587) vs gpu_b1024               11027	         1355	          952	          283
  PathMerge(587) vs pure_b1024              11027	         1355	          952	          283
  PathMerge(587) vs chunk_b16384            11028	         1355	          952	          283

