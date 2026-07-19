# Chapter 10 Discussion

本章では、Chapter 6 から Chapter 9 の結果を横断し、性能、最適化要因、容量、数値挙動の関係を解釈する。各 Research Question への正式な回答は Chapter 11 に集約し、本章では観測事実と未検証の推論を区別する。また、修正版 325557 の結果を旧 malformed input の historical evidence と混同しない。

## 10.1 Interpretation of Performance Results

Chapter 6 では、固定 b512 の block-based GPU_Opt が、評価した 4 グラフすべてで tuned third-party PathMerge より短い median 実行時間を示した。ただし、speedup の大きさはグラフ間で一様ではない。中心性能値と統計量は Table 6.1 および Figure 6.1～6.2 に集約されており、本章では再掲しない。この差は評価条件下の観測であり、単一の最適化要因だけで説明できるとは限らない。

Chapter 7 の factorial ablation では、Hybrid BFS と Dual-Stream Execution に正の main effect が観測され、Warp-Cooperative Accumulation はグラフにより効果の方向と大きさが異なった。一方、ablation の対象グラフ集合は RQ1 の主性能比較と一致せず、合成 4 グラフ集約も mixed-checkpoint である。したがって、H/W/A の main effect を加算または乗算して Chapter 6 の end-to-end speedup を因果分解しない。

roadNet-PA/TX の forced comparison では block カーネルが shared カーネルより短い実行時間を示し、現行の block-based source assignment を選ぶ設計根拠となった。ただし、この比較は 2 グラフに限定される。Figure 7.3 の phase breakdown と単一 trace のカーネル構成比も、測定した構成の時間配分を示す観測であり、最適化別の寄与率ではない。

## 10.2 Graph Characteristics and Observed Behavior

Table 5.3 が示すように、email-EuAll と roadNet 系グラフでは平均次数、次数分布、BFS depth、グラフ規模が異なる。email-EuAll はハブを含み探索が比較的浅い一方、roadNet は低次数で探索が深い。この構造差と Chapter 6 の speedup 差は整合するが、評価グラフ数が少なく、graph characteristics と性能差の因果関係を推定する解析は行っていない。

email-EuAll では Dual-Stream Execution の大きな正の main effect が観測され、初期化と kernel execution の overlap が実行時間に影響した可能性がある。ただし、roadNet 系では H/W/A factorial ablation を実施していないため、この解釈を roadNet の speedup へ適用しない。同様に、Warp-Cooperative Accumulation のグラフ別差は隣接走査量や lane utilization と関係する可能性があるものの、専用 hardware counter による検証はなく、平均次数だけから効果を予測する一般則は導かない。

グラフ規模は静的な CSR と BC 出力の大きさだけでなく、1 source 当たりの状態量を通じて batch-dependent working set にも影響する。ただし、本研究の容量評価は修正版 325557 の 1 グラフに限られるため、頂点数や辺数だけから他グラフの実行可能境界を外挿しない。

## 10.3 Performance–Capacity–Numerical Trade-Off

Chapter 8 の容量評価が示す中心的な区別は、static graph storage と batch-dependent source-local state の違いである。修正版 325557 では input graph file と CSR topology 自体は HBM3 容量を超えず、同時に保持する source-local state が working set を支配する。この関係は 8.1.2 項の working-set 式で定義したとおりであり、本章では再掲しない。

GPU_Opt、GPU_Opt_Pure、GPU_Opt_Pure_Chunked は、独立した 3 つの提案ではなく、この working set を異なる方法で管理する共通 framework の memory-management variants である。GPU_Opt は Unified Memory による managed placement を利用し、GPU_Opt_Pure は device memory に明示的に配置する。GPU_Opt_Pure_Chunked は source sub-batching により同時 resident な state を制限する。batch と sub-batch が分割するのは source 集合であり、graph partitioning や source sampling ではないため、いずれも全 source を処理する exact BC の実行単位を保つ。

Chapter 8 の targeted boundary は、Unified Memory が Pure より大きい batch を扱い、Chunked が tested upper bound をさらに拡張したことを示す一方、方式間の性能順位は定めない。各境界条件は 1 試行で、batch、allocation path、sub-batch 数も異なるためである。Unified Memory は有限の host memory と resource limit に依存し、Chunked も未測定条件で OOM を回避する保証を持たない。

数値面では、Chapter 9 の Tier A が小規模グラフで independent ground truth との full-vector consistency を評価し、Tier B が修正版 325557 で batch と memory path をまたぐ cross-implementation consistency を評価した。混合許容内の PASS は byte-identical を意味せず、PathMerge も independent ground truth ではない。したがって、性能または容量を変える実行経路の選択は、実行可能性だけでなく、証拠 tier と数値判定の意味を併記して評価する必要がある。

## 10.4 Implications for GH200

評価した GH200 構成では、Hopper GPU の HBM3 と Grace CPU 側 memory を接続する coherent memory architecture により、同じ計算基盤で device-only、Unified Memory、source sub-batching の経路を比較できた。観測された境界は、graph file size だけでなく source-local state の同時 resident 量を設計変数として扱う必要性を示す。

Unified Memory は allocation と placement の管理を簡潔にする一方、host-memory pressure、resource limit、migration の影響を受ける。source sub-batching は resident working set を明示的に制御できる一方、sub-batch loop と追加 launch を必要とする。本研究は full-run process RSS、physical HBM residency、total host residency、full-run migration bytes を取得していないため、両経路の物理配置量や migration cost の原因を定量比較しない。

Hybrid BFS、Warp-Cooperative Accumulation、Dual-Stream Execution、block-based source assignment の観測も GH200 上の評価に限られる。stream concurrency、memory capacity、interconnect、runtime policy が異なる GPU で同じ性能関係や容量境界が得られるとは推論しない。

## 10.5 Threats to Validity

**Internal validity.** RQ2 の合成 4 グラフ集約は mixed-checkpoint であり、same-checkpoint four-graph remeasurement ではない。RQ3 の targeted boundary は各条件 1 試行であり、RQ2、RQ3、RQ4 は別々の測定系列から得た。OOM については Pure の CUDA device-memory OOM と Unified Memory の cgroup host-memory OOM kill を区別し、失敗を 0 秒の runtime として扱わない。

**External validity.** RQ1 は 4 グラフ、RQ2 は合成 4 グラフと email-EuAll、RQ3 と RQ4 Tier B は修正版 325557、Tier A は小規模 3 グラフに限定される。評価装置も GH200 1 台である。この範囲から未測定グラフ、他 GPU、異なる host-memory limit へ結果を一般化しない。修正版 325557 は RQ1 の主性能比較には含まれない。

**Construct validity.** working-set values は配列寸法から求めた code-derived allocation estimate であり、measured RSS、physical residency、migration bytes ではない。正確性の PASS は既定の mixed absolute-relative tolerance 内の numerical consistency を示し、bitwise identity を示さない。観測された実行時間、main effect、phase 構成比、feasibility を同一の性能指標として扱わない。

**Baseline and data validity.** PathMerge は評価対象の第三者実装であり、原著論文著者の公式実装でも ground truth でもない。結果を PathMerge 一般へ拡張しない。修正版 325557 は決定的な内部再構成データであるが、元 seed と上流原本が未確認である。また、修復は範囲外頂点 ID と CSR 要素数の不一致を是正したもので、self-loop 87,442 本と多重度 2 の duplicate ordered pairs 866,924 組を保持する。これは保存された adjacency representation の観測事実であり、性能差の原因とは断定しない。旧 malformed input の `CORE_FAIL` は historical invalid-input evidence として保存し、current conclusion には用いない。

## 10.6 Limitations and Future Work

現時点の主要な制約は、headline 4 グラフに対する independent full-vector reference がないこと、現行 block-based 実装による統一的な 7 実装比較がないこと、PathMerge 比較が単一の第三者実装に限られることにある。容量評価では 1 グラフ・各条件 1 試行であり、full-duration の RSS、HBM residency、host residency、migration measurement もない。要因分析には mixed-checkpoint 集約が残り、修正版 325557 には上流原本と生成 seed の provenance 制約がある。

これらに対応するには、まず headline graph と大規模条件に対する独立 full-vector reference を整備し、公式または別系統の独立 PathMerge 実装と現行 block-based 実装を同じ条件で比較する必要がある。次に、same-checkpoint synthetic-4 ablation、追加グラフ、複数 trial の容量境界、他 GPU での再評価により観測範囲を広げる必要がある。さらに、full-run memory telemetry を取得し、修正版 graph の上流原本または生成 seed を確認することで、現在は分離している性能、容量、数値挙動、provenance の制約を個別に検証できる。
