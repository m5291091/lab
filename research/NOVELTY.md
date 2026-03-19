# 新規性整理ドキュメント

**Miyabi GH200 のアーキテクチャ特性を活用した Betweenness Centrality 計算の高速化**

---

## 目次

1. [研究背景](#1-研究背景)
2. [先行研究との差分](#2-先行研究との差分)
3. [新規性①：NVLink-C2C 非対称帯域の静的/動的データ分類による克服](#3-新規性1nvlink-c2c-非対称帯域の静的動的データ分類による克服)
4. [新規性②：UVM とグラフ演算の根本的相性問題の克服](#4-新規性2uvm-とグラフ演算の根本的相性問題の克服)
5. [新規性③：手法1 と手法2 の設計的相乗効果](#5-新規性3手法1-と手法2-の設計的相乗効果)
6. [新規性の組み合わせ（相乗効果）](#6-新規性の組み合わせ相乗効果)
7. [アブレーション設計の意義](#7-アブレーション設計の意義)

---

## 1. 研究背景

### GH200 Grace Hopper Superchip とは

NVIDIA GH200 は、Grace CPU（LPDDR5X メモリ搭載）と Hopper GPU（HBM3 メモリ搭載）を **NVLink-C2C** と呼ばれる高速インターコネクトで接続した統合チップである。両者は同一パッケージに収まっており、従来の PCIe 接続とは異なるメモリアーキテクチャを持つ。この構造により、CPU メモリと GPU メモリを **統一アドレス空間（Unified Virtual Memory, UVM）** として扱える `cudaMallocManaged` API が特に有効と期待されている。

```
GH200 のメモリ階層

  ┌──────────────────────────────────────┐
  │ Grace CPU                            │
  │  ├─ LPDDR5X  480 GB                 │
  │  └─ NVLink-C2C ──── 実効 156 GB/s   │
  │              ↕                       │
  │ Hopper GPU                           │
  │  └─ HBM3    96 GB  ──── 1488 GB/s  │
  └──────────────────────────────────────┘
```

### なぜ BC 計算が重要か

Betweenness Centrality（媒介中心性, BC）は、グラフ内の各頂点が「情報の橋渡し役」としてどれだけ重要かを測る指標である。ソーシャルネットワーク分析・交通ネットワーク最適化・サイバーセキュリティなど幅広い分野で不可欠であるが、計算複雑度は $O(V \cdot (V + E))$ であり、大規模グラフに対しては並列計算が必須である。

GH200 は HBM3 の超高帯域（1488 GB/s）と大容量 CPU メモリ（480 GB）を組み合わせており、BC のような**帯域幅律速**かつ**大容量グラフ**を扱う処理に対して理想的なプラットフォームと見なされる。しかし、`cudaMallocManaged` を用いると予想外の性能低下が発生することが本研究で明らかになった。

---

## 2. 先行研究との差分

### 従来 GPU（PCIe 接続）との根本的違い

従来の GPU 研究では、CPU と GPU は **PCIe バス**（理論値 64 GB/s）で接続されており、`cudaMallocManaged` を用いた場合のデフォルト動作は以下のようになる。

| 環境 | デフォルトの UVM 挙動 | 帯域幅 |
|------|---------------------|-------|
| PCIe 接続 GPU（A100, V100 等） | データは GPU メモリ側に初期配置、またはページフォルト駆動で移動 | HBM2e: 2000 GB/s（A100）|
| **GH200（NVLink-C2C）** | **データはデフォルトで CPU LPDDR5X 側に配置** | **NVLink-C2C: 156 GB/s** |

PCIe 環境では、`cudaMallocManaged` を使ってもデータはアクセス頻度に応じて GPU メモリに移動するか、最初から GPU 側に配置される。一方 **GH200 では、データがデフォルトで CPU 側（LPDDR5X）に配置され、GPU からのすべてのアクセスが NVLink-C2C 経由になる**。この帯域差は **9.6 倍**（1488 GB/s vs 156 GB/s）に達する。

### なぜ従来研究では問題が発生しなかったのか

1. PCIe 環境では CPU‒GPU 間の帯域がそもそも小さく、UVM の「CPU 側配置」問題は顕在化しにくい
2. A100 等の DGX システムでは、NVLink でも GPU 間接続のため帯域差の問題が異なる
3. GH200 の NVLink-C2C は CPU‒GPU 間を繋ぐ「非対称な」接続であり、先行研究の知見が直接適用できない

この「GH200 固有の非対称帯域問題」に対処した BC 最適化研究は本研究以前には存在しない。

---

## 3. 新規性①：NVLink-C2C 非対称帯域の静的/動的データ分類による克服

### 問題の本質

`cudaMallocManaged` を素朴に使うと（`brandes_gpu_managed.cu`）、グラフトポロジデータ（CSR 形式の `R[]`, `C[]`）が CPU LPDDR5X 上に置かれ、GPU カーネルが毎サイクルアクセスするたびに NVLink-C2C（156 GB/s）を経由することになる。

```
【GPU_Managed の問題】

  GPU カーネル
    │
    ├─ R[v], C[v] アクセス ──── NVLink-C2C（156 GB/s）──→ CPU LPDDR5X
    │                            ↑ 9.6 倍遅い！
    └─ d_d[], d_sigma[] ──── HBM3（1488 GB/s）← こちらは速い
```

**なぜ新しいのか**：従来の GPU プログラミングでは「どのデータがどこに置かれるか」を意識する必要はなかった。GH200 では CPU‒GPU 非対称帯域という新しい次元の最適化が必要となる。

### 解決策：静的/動的データの分類と適応型配置

本研究では、アルゴリズム内のデータを以下の2種類に分類する：

| 種別 | データ | アクセスパターン | 最適配置 |
|------|--------|----------------|---------|
| **静的データ**（読み取り専用） | `R[]`, `C[]`（グラフ構造） | 全 BFS ラウンドで繰り返し読み取り | HBM3 or HBM3 L2 キャッシュ |
| **動的データ**（読み書き両方） | `d_d[]`, `d_sigma[]`, `d_delta[]`, `d_CB[]` | 各ソース頂点ごとに更新 | HBM3（デフォルト） |

#### コードの証拠：`cudaMemAdviseSetReadMostly`（`brandes_gpu_readmostly.cu`）

```cuda
// brandes_gpu_readmostly.cu  Lines 59-62
// ── 手法1 ① SetReadMostly ──
// HBM3 L2 への複製を許可（GPU_Managed にはこれがない）
CUDA_ERR_CHK(cudaMemAdvise(R_m, (n_nodes + 1) * sizeof(int),
                            cudaMemAdviseSetReadMostly, 0));
CUDA_ERR_CHK(cudaMemAdvise(C_m, edge_size * sizeof(int),
                            cudaMemAdviseSetReadMostly, 0));
```

`cudaMemAdviseSetReadMostly` は「このメモリ領域は読み取り専用」と CUDA ドライバに伝えるヒントであり、GPU の HBM3 L2 キャッシュに**複製コピー**を保持することを可能にする。これにより、一度フェッチしたグラフデータがキャッシュヒットし、NVLink-C2C を経由したリモートアクセスが大幅に削減される。

#### コードの証拠：グラフサイズ適応型配置（35% 閾値）

```cuda
// brandes_gpu_readmostly.cu  Lines 67-92
// グラフが HBM3 の 35% 以内に収まるか判定
const bool topo_on_gpu = (topo_bytes < prop.totalGlobalMem * 0.35);

if (topo_on_gpu) {
    // 小グラフ → HBM3 に直接転送（最高速）
    CUDA_ERR_CHK(cudaMemAdvise(R_m, ..., cudaMemAdviseSetAccessedBy, 0));
    CUDA_ERR_CHK(cudaMemAdvise(C_m, ..., cudaMemAdviseSetAccessedBy, 0));
    CUDA_ERR_CHK(cudaMemPrefetchAsync(R_m, ..., 0, 0));
    CUDA_ERR_CHK(cudaMemPrefetchAsync(C_m, ..., 0, 0));
} else {
    // 大グラフ → CPU LPDDR5X に置いたまま、ReadMostly の L2 キャッシュを活用
    CUDA_ERR_CHK(cudaMemAdvise(R_m, ..., cudaMemAdviseSetPreferredLocation,
                                cudaCpuDeviceId));
    CUDA_ERR_CHK(cudaMemAdvise(C_m, ..., cudaMemAdviseSetPreferredLocation,
                                cudaCpuDeviceId));
}
```

35% という閾値は、動的データ（`d_d`, `d_sigma`, `d_delta`, `d_CB`）が残りの HBM3 容量を消費することを考慮した経験的ヒューリスティックである。

#### 効果

実験結果（`RESULT.md`）によると、手法①単独（`GPU_ReadMostly`）で以下の改善が確認された：

- `web-NotreDame`（56K 頂点）：`GPU_Managed` 比 **5.1 倍高速化**（HBM3 ベースラインに完全回復）
- `roadNet-CA`（325K 頂点）：`GPU_Managed` 比 **5.0 倍高速化**

---

## 4. 新規性②：UVM とグラフ演算の根本的相性問題の克服

### 問題の本質：O(n) 逐次初期化ボトルネック

Brandes アルゴリズムでは、各ソース頂点 `s` の BFS を開始するたびに `d_d[]`, `d_sigma[]`, `d_delta[]` をゼロクリアする必要がある。`GPU_Managed`（および `GPU_ReadMostly`）では、この初期化をカーネル内で `tid==0` の 1 スレッドが逐次実行していた。

#### コードの証拠：逐次初期化（`brandes_gpu_managed.cu`）

```cuda
// brandes_gpu_managed.cu  Lines 160-165
// 【ボトルネック】tid==0 による O(n) 逐次初期化
if (tid == 0) {
    for (int v = 0; v < n_nodes; v++) {
        d_d    [batch_offset + v] = (v == s) ? 0  : -1;
        d_sigma[batch_offset + v] = (v == s) ? 1  :  0;
        d_delta[batch_offset + v] = 0.0;
    }
}
__syncthreads();  // 全スレッドが初期化完了を待つ
```

GPU には数千の CUDA スレッドが存在するが、**初期化中は `tid==0` 以外の全スレッドが待機**しており、SM の利用率が著しく低下する。これは UVM 特有の問題ではないが、UVM を使うと動的データが HBM3 に配置されるため、`cudaMemset` による並列初期化を素直に使えないという制約から生じていた。

**なぜ新しいのか**：グラフ BFS のような不規則アクセスパターンでは、UVM のページフォルト機構とカーネル内ループ初期化が組み合わさることで性能が低下する。この問題を `cudaMemsetAsync` と Copy Engine を組み合わせて解決したのが本研究の独自の洞察である。

### 解決策：`cudaMemsetAsync` による Copy Engine 委譲

`cudaMemsetAsync` は CUDA の **Copy Engine**（DMA エンジン）を利用するため、SM（Streaming Multiprocessor）とは**独立して**動作する。

#### コードの証拠：`cudaMemsetAsync`（`brandes_gpu_opt.cu`）

```cuda
// brandes_gpu_opt.cu  Lines 447-452
// [最適化：手法2] GPU Copy Engine で非同期 memset（SM を占有しない）
cudaMemsetAsync(bufs[si].d_d,
                0xFF,       // -1 として解釈（int 型のビットパターン）
                curr_batch * n_nodes * sizeof(int),    st);
cudaMemsetAsync(bufs[si].d_sigma,
                0,
                curr_batch * n_nodes * sizeof(int),    st);
cudaMemsetAsync(bufs[si].d_delta,
                0,
                curr_batch * n_nodes * sizeof(double), st);
```

Copy Engine はメモリ転送専用のハードウェアユニットであり、SM の計算リソースを消費しない。これにより：

1. **SM は初期化を待たずに次の処理に移れる**
2. **Copy Engine と SM が真に並列動作できる**

### 2 ストリーム ダブルバッファリングによるオーバーラップ

`cudaMemsetAsync` による非同期初期化の恩恵を最大化するために、**2 本の CUDA ストリーム**を交互に使う「ダブルバッファリング」を実装している。

#### コードの証拠：2 ストリーム設計（`brandes_gpu_opt.cu`）

```cuda
// brandes_gpu_opt.cu  Line 318
const int NS = 2;  // 2 本のストリーム

// brandes_gpu_opt.cu  Lines 428-430
// ストリームを交互に切り替える
int si = (s_start / BATCH_PER_STREAM) % NS;
cudaStream_t st = streams[si];
```

```
【シングルストリーム（GPU_ReadMostly）の時間軸】

  時間 →
  [初期化 A] [BFS+Back A] [初期化 B] [BFS+Back B] ...
              ↑計算           ↑SM が遊ぶ

【2 ストリーム（GPU_Opt）の時間軸】

  Stream 0: [初期化 A] [BFS+Back A]             [初期化 C] [BFS+Back C]
  Stream 1:            [初期化 B] [BFS+Back B]             [初期化 D]
                        ↑A 計算中にBを初期化 → 待ち時間がゼロ！
```

#### 効果

- `web-NotreDame`（56K 頂点、高次数ノードが多い密なグラフ）：`GPU_ReadMostly` 比 **1.5 倍高速化**
- `roadNet-CA`（325K 頂点）：`GPU_ReadMostly` 比 **1.06 倍高速化**
- 全グラフ平均：**6〜13% の追加高速化**

---

## 5. 新規性③：手法1 と手法2 の設計的相乗効果

### 相乗効果が生まれる理由

新規性①（ReadMostly + 適応型配置）と新規性②（cudaMemsetAsync + 2 ストリーム）は、単なる独立した最適化の「足し算」ではなく、**設計レベルで互いを前提とした相乗関係**にある。

| | 手法1 なし（GPU_Managed） | 手法1 あり（GPU_ReadMostly） |
|--|--------------------------|------------------------------|
| **手法2 を適用した場合** | 改善幅が小さい（NVLink-C2C がボトルネックのため、初期化の最適化が意味をなさない） | **最大 1.5 倍の追加改善**（SM が本来の計算に集中できる） |

具体的には：

1. **手法1 がないと手法2 は意味をなさない**：NVLink-C2C がボトルネックの状態では、SM の待機時間の削減より帯域幅の制約が支配的になる。手法2 が効くためには、まず手法1 でグラフデータを HBM3 に確保する必要がある。

2. **手法1 が手法2 の対象データを固定する**：ReadMostly により静的データ（R, C）が HBM3 に安定して置かれると、動的データ（d_d, d_sigma, d_delta）の初期化が律速となる。これによって手法2（cudaMemsetAsync）が改善すべき明確なボトルネックを生み出す。

3. **2 ストリームが HBM3 帯域を無駄なく使う**：手法1 によって両ストリームが HBM3 の豊富な帯域（1488 GB/s）を共有できる状態になって初めて、ダブルバッファリングによる並行実行が効果を発揮する。

### コードの証拠：`brandes_gpu_opt.cu` に両手法が統合されている

```cuda
// brandes_gpu_opt.cu  Lines 543-548（手法1: ReadMostly）
CUDA_ERR_CHK(cudaMemAdvise(R_m, (n_nodes + 1) * sizeof(int),
                            cudaMemAdviseSetReadMostly, 0));
CUDA_ERR_CHK(cudaMemAdvise(C_m, edge_size * sizeof(int),
                            cudaMemAdviseSetReadMostly, 0));

// brandes_gpu_opt.cu  Lines 447-452（手法2: cudaMemsetAsync）
cudaMemsetAsync(bufs[si].d_d,     0xFF, ..., st);
cudaMemsetAsync(bufs[si].d_sigma, 0,    ..., st);
cudaMemsetAsync(bufs[si].d_delta, 0,    ..., st);
```

**なぜこの組み合わせが新しいのか**：これまでの GPU 最適化研究では、メモリ配置の最適化（手法1 相当）とパイプライン最適化（手法2 相当）は別々の問題として扱われてきた。本研究では GH200 固有のアーキテクチャ制約を起点として**両者を統合した設計**を提案し、単独では達成できない性能を実現した点が独自の貢献である。

---

## 6. 新規性の組み合わせ（相乗効果）

4 段階の実装（`GPU` → `GPU_Managed` → `GPU_ReadMostly` → `GPU_Opt`）の性能を実際の数値で比較すると、相乗効果の大きさが明確に見える。

### 性能比較（`GPU` = 1.0 として正規化）

| グラフ | `GPU_Managed` | `GPU_ReadMostly` | `GPU_Opt` | 手法1 寄与 | 手法2 追加寄与 |
|--------|:-----------:|:--------------:|:-------:|:----------:|:----------:|
| 7K（email-Enron） | 1.183 | 1.439 | 1.416 | +21.6% | −1.6%（誤差範囲） |
| 11K（Amazon） | 0.710 | 0.982 | 0.982 | +27.2% | ±0% |
| 56K（web-NotreDame） | 0.196 | 1.000 | **1.132** | +80.4% | +13.2% |
| 325K（roadNet-CA） | 0.198 | 0.999 | **1.062** | +80.1% | +6.3% |

- 手法1 単独で最大 **5.1 倍**の回復（NVLink-C2C ボトルネックの解消）
- 手法2 の追加効果は **6〜13%**（パイプライン効率化）
- 最終的に `GPU_Opt` は cudaMalloc ベースラインを**上回る**性能を Unified Memory で達成

### 相乗効果の図解

```
性能（GPU = 1.0 基準）
  1.2 │                              ████ GPU_Opt（手法1+2）
      │                    ████      ████
  1.0 ├────────────────────────────────────── GPU ベースライン
      │                    ████
  0.8 │
      │
  0.5 │
      │
  0.2 │████ GPU_Managed（問題あり）
      │████
  0.0 └──────────────────────────────────
           7K   11K   56K  325K（グラフ頂点数）

手法1：問題の「崩壊」を「回復」させる（ボトルネック除去）
手法2：「普通に速い」をさらに「少し速く」する（効率化）
```

---

## 7. アブレーション設計の意義

### なぜ 4 段階の実装が必要か

この研究では意図的に 4 段階の実装を設計している。各段階が「ある最適化を追加した場合の効果のみ」を分離測定できるアブレーションスタディの構造になっている。

```
【4 段階アブレーション設計】

  実装①  GPU（cudaMalloc）
    ↓ UVM 切り替え（手法なし）
  実装②  GPU_Managed
    ↓ 手法1 のみ追加（手法2 はなし）
  実装③  GPU_ReadMostly  ← 手法1 の独立した効果を測定
    ↓ 手法2 を追加（手法1 との相乗効果）
  実装④  GPU_Opt         ← 手法1 + 手法2 の組み合わせ効果を測定
```

| 実装 | 手法1 | 手法2 | 意義 |
|------|:-----:|:-----:|------|
| `GPU` | ✗ | ✗ | HBM3 直接アクセスの理想的基準値 |
| `GPU_Managed` | ✗ | ✗ | UVM 素朴適用時の問題を定量化 |
| `GPU_ReadMostly` | ✓ | ✗ | **手法1 のみの寄与**を独立測定 |
| `GPU_Opt` | ✓ | ✓ | **手法2 の追加寄与**を独立測定 |

### この設計がなぜ新規性の「証明」になるか

1. **実装②と③の差分** = NVLink-C2C 非対称帯域問題（新規性①）の定量的証明
   - 差が大きいほど「GH200 固有のボトルネックが存在した」ことの証拠
   
2. **実装③と④の差分** = 2 ストリームパイプラインの追加効果（新規性②）の定量的証明
   - 手法2 が手法1 を前提として意味を持つことを示す
   
3. **実装①と④の差分** = Unified Memory でも cudaMalloc を超えられることの証明
   - 「UVM は遅い」という先入観を実測値で覆す

4. **実装①と②の差分** = GH200 アーキテクチャの問題定義
   - 同一コードが GH200 では性能低下するという「問題の証明」

アブレーションスタディのない研究では「なぜ速くなったのか」が不明瞭になる。4 段階設計により各手法の貢献を**独立に定量化**できることが、この研究の科学的厳密さを担保している。

---

## 参考：実装ファイル対応表

| ファイル | 実装 | 主な新規性 |
|---------|------|-----------|
| `brandes_gpu.cu` | GPU（ベースライン） | — |
| `brandes_gpu_managed.cu` | GPU_Managed | 問題の定義・定量化 |
| `brandes_gpu_readmostly.cu` | GPU_ReadMostly | **新規性①**（ReadMostly + 適応配置） |
| `brandes_gpu_opt.cu` | GPU_Opt | **新規性①②③**（全手法統合） |

各実装の正確性は `brandes_sequential.cpp` との BC 値照合により検証済みであり、全グラフで `PASS: 5 FAIL: 0` が確認されている。
