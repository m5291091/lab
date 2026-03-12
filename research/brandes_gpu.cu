#include "common.h"
#include "brandes.h"

namespace cg = cooperative_groups;
using namespace std;

__device__ bool isUndirected = true;

// find_shortest_paths は direction.cu と同じ実装（すでに高度に最適化されているため）
__device__ void find_shortest_paths(
		int *R, int *C, int *d_d, int *d_sigma, int *d_Q_curr, int *d_Q_next,
		int *d_S, int *d_S_ends, int batch_idx, int n_nodes, int &Q_curr_len,
		int &Q_next_len, int &S_len, int &S_ends_len, int &depth) {

	int tid = threadIdx.x;
	int bsize = blockDim.x;
	int v, w;

	while (true) {
		int threshold = min(max(n_nodes / 20, 32), 1024);

		if (Q_curr_len <= threshold) {
			// トップダウン探索
			for (int i = tid; i < Q_curr_len; i += bsize) {
				v = d_Q_curr[batch_idx * n_nodes + i];
				for (int j = R[v]; j < R[v+1]; j++) {
					w = C[j];
					if (atomicCAS(&d_d[batch_idx * n_nodes + w], -1, depth + 1) == -1) {
						int pos = atomicAdd(&Q_next_len, 1);
						d_Q_next[batch_idx * n_nodes + pos] = w;
					}
					if (d_d[batch_idx * n_nodes + w] == depth + 1) {
						atomicAdd(&d_sigma[batch_idx * n_nodes + w], d_sigma[batch_idx * n_nodes + v]);
					}
				}
			}
		} else {
			// ボトムアップ探索
			for (int i = tid; i < n_nodes; i += bsize) {
				w = i;
				if (d_d[batch_idx * n_nodes + w] == -1) {
					int sum_sigma = 0;
					for (int j = R[w]; j < R[w+1]; j++) {
						v = C[j];
						if (d_d[batch_idx * n_nodes + v] == depth) {
							sum_sigma += d_sigma[batch_idx * n_nodes + v];
						}
					}
					if (sum_sigma > 0) {
						int expected = -1;
						if (atomicCAS(&d_d[batch_idx * n_nodes + w], expected, depth + 1) == expected) {
							atomicAdd(&d_sigma[batch_idx * n_nodes + w], sum_sigma);
							int pos = atomicAdd(&Q_next_len, 1);
							d_Q_next[batch_idx * n_nodes + pos] = w;
						} else {
							if (d_d[batch_idx * n_nodes + w] == depth + 1) {
								atomicAdd(&d_sigma[batch_idx * n_nodes + w], sum_sigma);
							}
						}
					}
				}
			}
		}

		__syncthreads();

		if (Q_next_len == 0) {
			if (tid == 0) depth = d_d[batch_idx * n_nodes + d_S[batch_idx * n_nodes + S_len-1]];
			break;
		}

		int curr_Q_next_len = Q_next_len;
		for (int i = tid; i < curr_Q_next_len; i += bsize) {
			d_Q_curr[batch_idx * n_nodes + i] = d_Q_next[batch_idx * n_nodes + i];
			d_S[batch_idx * n_nodes + S_len + i] = d_Q_next[batch_idx * n_nodes + i];
		}

		__syncthreads();

		if (tid == 0) {
			d_S_ends[batch_idx * (n_nodes+1) + S_ends_len] = S_len + curr_Q_next_len;
			S_ends_len++;
			Q_curr_len = curr_Q_next_len;
			S_len += curr_Q_next_len;
			Q_next_len = 0;
			depth++;
		}

		__syncthreads();
	}
}

// Cooperative Groups を用いて最適化した依存関係の計算
__device__ void accumulate_dependencies_cooperative(
		int *R, int *C, int *d_d, int *d_sigma, double *d_delta, 
		int *d_S, int *d_S_ends, int batch_idx, int n_nodes, int &depth) {

	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<32>(block);
	int tid_in_block = block.thread_rank();
	int warp_id = tid_in_block / warp.size();
	int num_warps_in_block = block.size() / warp.size();

	while (depth > 0) {
		int start = d_S_ends[batch_idx * (n_nodes+1) + depth];
		int end = d_S_ends[batch_idx * (n_nodes+1) + depth + 1];
		int nodes_in_level = end - start;

		// ブロック内のワープ単位で、レベル内のノードを処理
		for (int i = warp_id; i < nodes_in_level; i += num_warps_in_block) {
			int w = d_S[batch_idx * n_nodes + start + i];
			double sigma_w = d_sigma[batch_idx * n_nodes + w];
			
			double local_sum = 0.0;
			// ワープ内のスレッドで、一つの 'w' の隣接リスト処理を分担
			for (int j = R[w] + warp.thread_rank(); j < R[w+1]; j += warp.size()) {
				int v = C[j];
				if (d_d[batch_idx * n_nodes + v] == d_d[batch_idx * n_nodes + w] + 1) {
					local_sum += (sigma_w / (double)d_sigma[batch_idx * n_nodes + v]) * (1.0 + d_delta[batch_idx * n_nodes + v]);
				}
			}

			// ワープ内で local_sum をリダクション
			for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
				local_sum += warp.shfl_down(local_sum, offset);
			}
			
			// ワープの代表スレッド(rank 0)が結果を書き込む
			if (warp.thread_rank() == 0) {
				d_delta[batch_idx * n_nodes + w] = local_sum; // レーン0のlocal_sumが合計値
			}
		}
		
		block.sync();
		if (tid_in_block == 0) depth--;
		block.sync();
	}
}

// BFS フェーズカーネル: 初期化 + BFS 前向き探索。完了後に最大深さを d_depth に保存。
__global__ void brandes_bfs_kernel(
		int *R, int *C, int n_nodes,
		int *d_d, int *d_sigma, int *d_Q_curr, int *d_Q_next,
		int *d_S, int *d_S_ends, double *d_delta, int *d_depth, int s_start) {
	int batch_idx = blockIdx.x;
	int s   = s_start + batch_idx;
	int tid = threadIdx.x;

	__shared__ int Q_curr_len, Q_next_len, S_len, S_ends_len, depth;

	if (tid == 0) {
		for (int v = 0; v < n_nodes; v++) {
			d_d    [batch_idx * n_nodes + v] = (v == s) ? 0  : -1;
			d_sigma[batch_idx * n_nodes + v] = (v == s) ? 1  :  0;
			d_delta[batch_idx * n_nodes + v] = 0.0;
		}
		d_Q_curr[batch_idx * n_nodes] = s;
		Q_curr_len = 1; Q_next_len = 0;
		d_S[batch_idx * n_nodes] = s;
		S_len = 1;
		d_S_ends[batch_idx * (n_nodes+1)]     = 0;
		d_S_ends[batch_idx * (n_nodes+1) + 1] = 1;
		S_ends_len = 2;
		depth = 0;
	}
	__syncthreads();

	find_shortest_paths(R, C, d_d, d_sigma, d_Q_curr, d_Q_next, d_S, d_S_ends,
			batch_idx, n_nodes, Q_curr_len, Q_next_len, S_len, S_ends_len, depth);
	__syncthreads();

	// バックワードカーネルへ最大深さを受け渡す
	if (tid == 0) d_depth[batch_idx] = depth;
}

// バックワードフェーズカーネル: 依存値累積 + BC への集計。
__global__ void brandes_back_kernel(
		int *R, int *C, double *CB, int n_nodes,
		int *d_d, int *d_sigma, double *d_delta,
		int *d_S, int *d_S_ends, const int *d_depth, int s_start) {
	int batch_idx = blockIdx.x;
	int s   = s_start + batch_idx;
	int tid = threadIdx.x;

	__shared__ int depth;
	if (tid == 0) depth = d_depth[batch_idx];
	__syncthreads();

	accumulate_dependencies_cooperative(R, C, d_d, d_sigma, d_delta,
			d_S, d_S_ends, batch_idx, n_nodes, depth);
	__syncthreads();

	for (int v = tid; v < n_nodes; v += blockDim.x) {
		if (v != s) {
			double contribution = isUndirected
					? d_delta[batch_idx * n_nodes + v] / 2.0
					: d_delta[batch_idx * n_nodes + v];
			atomicAdd(&CB[v], contribution);
		}
	}
}

__global__ void brandes_kernel_cooperative(int *R, int *C, double *CB, int n_nodes, int *d_d, int *d_sigma, int *d_Q_curr, int *d_Q_next, int *d_S, int *d_S_ends, double *d_delta, int s_start) {
	int batch_idx = blockIdx.x;
	int s = s_start + batch_idx;
	int tid = threadIdx.x;

	__shared__ int Q_curr_len, Q_next_len, S_len, S_ends_len, depth;

	if (tid == 0) {
		for (int v = 0; v < n_nodes; v++) {
			d_d[batch_idx * n_nodes + v] = (v == s) ? 0 : -1;
			d_sigma[batch_idx * n_nodes + v] = (v == s) ? 1 : 0;
			d_delta[batch_idx * n_nodes + v] = 0.0;
		}
		d_Q_curr[batch_idx * n_nodes] = s;
		Q_curr_len = 1;
		Q_next_len = 0;
		d_S[batch_idx * n_nodes] = s;
		S_len = 1;
		d_S_ends[batch_idx * (n_nodes+1)] = 0;
		d_S_ends[batch_idx * (n_nodes+1) + 1] = 1;
		S_ends_len = 2;
		depth = 0;
	}
	__syncthreads();

	find_shortest_paths(R, C, d_d, d_sigma, d_Q_curr, d_Q_next, d_S, d_S_ends,
			batch_idx, n_nodes, Q_curr_len, Q_next_len, S_len, S_ends_len, depth);
	__syncthreads();

	// 最適化された関数を呼び出す
	accumulate_dependencies_cooperative(R, C, d_d, d_sigma, d_delta, d_S, d_S_ends,
			batch_idx, n_nodes, depth);
	__syncthreads();

	for (int v = tid; v < n_nodes; v += blockDim.x) {
		if (v != s) {
			double contribution = isUndirected ? d_delta[batch_idx * n_nodes + v] / 2.0 : d_delta[batch_idx * n_nodes + v];
			atomicAdd(&CB[v], contribution);
		}
	}
}

vector<double> brandes_gpu(Graph &G) {
	int *R = G.getAdjacencyListPointers();
	int *C = G.getAdjacencyList();
	int n_nodes = G.getNodeCount();
	int edge_size = 2 * G.getEdgeCount();

	int num_gpus;
	CUDA_ERR_CHK(cudaGetDeviceCount(&num_gpus));
	num_gpus = std::min(num_gpus, omp_get_max_threads());
	if(num_gpus == 0) exit(EXIT_FAILURE);

	vector<double> CB(n_nodes, 0.0);

#pragma omp parallel num_threads(num_gpus)
	{
		int gpu_id = omp_get_thread_num();
		CUDA_ERR_CHK(cudaSetDevice(gpu_id));

		cudaDeviceProp prop;
		CUDA_ERR_CHK(cudaGetDeviceProperties(&prop, gpu_id));

		int threads_per_block = std::min(prop.maxThreadsPerBlock, n_nodes);
		threads_per_block = (threads_per_block / 32) * 32;
		threads_per_block = std::max(threads_per_block, 32);

		int *d_R, *d_C, *d_d, *d_sigma, *d_Q_curr, *d_Q_next, *d_S, *d_S_ends;
		double *d_CB, *d_delta;

		CUDA_ERR_CHK(cudaMalloc(&d_R, (n_nodes + 1) * sizeof(int)));
		CUDA_ERR_CHK(cudaMalloc(&d_C, edge_size * sizeof(int)));
		CUDA_ERR_CHK(cudaMalloc(&d_CB, n_nodes * sizeof(double)));
		CUDA_ERR_CHK(cudaMemset(d_CB, 0, n_nodes * sizeof(double)));

		CUDA_ERR_CHK(cudaMemcpy(d_R, R, (n_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_ERR_CHK(cudaMemcpy(d_C, C, edge_size * sizeof(int), cudaMemcpyHostToDevice));

		size_t free_mem, total_mem;
		CUDA_ERR_CHK(cudaMemGetInfo(&free_mem, &total_mem));

		const size_t per_batch_mem =
			n_nodes * (4*sizeof(int) + sizeof(double)) +
			n_nodes * sizeof(int) +
			(n_nodes + 1) * sizeof(int) +
			threads_per_block * sizeof(int);

		const size_t safety_margin = static_cast<size_t>(free_mem * 0.15);
		int BATCH_SIZE = (free_mem - safety_margin) / per_batch_mem;
		BATCH_SIZE = std::max(1, std::min(BATCH_SIZE, 1024));

		// 使用メモリを報告 (全データが HBM3 上の baseline)
		size_t topology_bytes = ((size_t)(n_nodes + 1) + (size_t)edge_size) * sizeof(int);
		size_t dynamic_bytes  = (size_t)BATCH_SIZE * per_batch_mem;
		fprintf(stderr, "  > [Mem] GPU HBM3: total=%.1f GB, free_before=%.1f GB\n",
		        total_mem / 1e9, free_mem / 1e9);
		fprintf(stderr, "  > [Mem] topology(HBM3)=%.2f GB, dynamic(HBM3)=%.2f GB, batch=%d\n",
		        topology_bytes / 1e9, dynamic_bytes / 1e9, BATCH_SIZE);

		CUDA_ERR_CHK(cudaMalloc(&d_d, BATCH_SIZE * n_nodes * sizeof(int)));
		CUDA_ERR_CHK(cudaMalloc(&d_sigma, BATCH_SIZE * n_nodes * sizeof(int)));
		CUDA_ERR_CHK(cudaMalloc(&d_Q_curr, BATCH_SIZE * n_nodes * sizeof(int)));
		CUDA_ERR_CHK(cudaMalloc(&d_Q_next, BATCH_SIZE * n_nodes * sizeof(int)));
		CUDA_ERR_CHK(cudaMalloc(&d_S, BATCH_SIZE * n_nodes * sizeof(int)));
		CUDA_ERR_CHK(cudaMalloc(&d_S_ends, BATCH_SIZE * (n_nodes + 1) * sizeof(int)));
		CUDA_ERR_CHK(cudaMalloc(&d_delta, BATCH_SIZE * n_nodes * sizeof(double)));

		int *d_depth;
		CUDA_ERR_CHK(cudaMalloc(&d_depth, BATCH_SIZE * sizeof(int)));

		// フェーズ別時間計測用 CUDA イベント
		cudaEvent_t ev_bfs_s, ev_bfs_e, ev_back_e;
		CUDA_ERR_CHK(cudaEventCreate(&ev_bfs_s));
		CUDA_ERR_CHK(cudaEventCreate(&ev_bfs_e));
		CUDA_ERR_CHK(cudaEventCreate(&ev_back_e));
		float total_bfs_ms = 0.0f, total_back_ms = 0.0f;

#pragma omp for schedule(static, 1)
		for (int s_start = 0; s_start < n_nodes; s_start += BATCH_SIZE) {
			int curr_batch = std::min(BATCH_SIZE, n_nodes - s_start);
			if (curr_batch <= 0) continue;

			CUDA_ERR_CHK(cudaEventRecord(ev_bfs_s));
			brandes_bfs_kernel<<<curr_batch, threads_per_block>>>(
					d_R, d_C, n_nodes,
					d_d, d_sigma, d_Q_curr, d_Q_next,
					d_S, d_S_ends, d_delta, d_depth, s_start);
			CUDA_ERR_CHK(cudaEventRecord(ev_bfs_e));

			brandes_back_kernel<<<curr_batch, threads_per_block>>>(
					d_R, d_C, d_CB, n_nodes,
					d_d, d_sigma, d_delta,
					d_S, d_S_ends, d_depth, s_start);
			CUDA_ERR_CHK(cudaEventRecord(ev_back_e));

			CUDA_ERR_CHK(cudaEventSynchronize(ev_back_e));
			float b_ms = 0.0f, bk_ms = 0.0f;
			CUDA_ERR_CHK(cudaEventElapsedTime(&b_ms,  ev_bfs_s, ev_bfs_e));
			CUDA_ERR_CHK(cudaEventElapsedTime(&bk_ms, ev_bfs_e, ev_back_e));
			total_bfs_ms  += b_ms;
			total_back_ms += bk_ms;

			CUDA_ERR_CHK(cudaPeekAtLastError());
		}
		CUDA_ERR_CHK(cudaDeviceSynchronize());

		CUDA_ERR_CHK(cudaEventDestroy(ev_bfs_s));
		CUDA_ERR_CHK(cudaEventDestroy(ev_bfs_e));
		CUDA_ERR_CHK(cudaEventDestroy(ev_back_e));

		vector<double> gpu_CB(n_nodes);
		CUDA_ERR_CHK(cudaMemcpy(gpu_CB.data(), d_CB, n_nodes * sizeof(double), cudaMemcpyDeviceToHost));

#pragma omp critical
		{
			fprintf(stderr, "  > [GPU%d Phase] BFS: %.4f sec, Backward: %.4f sec\n",
					gpu_id, total_bfs_ms / 1000.0f, total_back_ms / 1000.0f);
			for(int i = 0; i < n_nodes; ++i) {
				CB[i] += gpu_CB[i];
			}
		}

		CUDA_ERR_CHK(cudaFree(d_R));
		CUDA_ERR_CHK(cudaFree(d_C));
		CUDA_ERR_CHK(cudaFree(d_CB));
		CUDA_ERR_CHK(cudaFree(d_d));
		CUDA_ERR_CHK(cudaFree(d_sigma));
		CUDA_ERR_CHK(cudaFree(d_Q_curr));
		CUDA_ERR_CHK(cudaFree(d_Q_next));
		CUDA_ERR_CHK(cudaFree(d_S));
		CUDA_ERR_CHK(cudaFree(d_S_ends));
		CUDA_ERR_CHK(cudaFree(d_delta));
		CUDA_ERR_CHK(cudaFree(d_depth));
	}

	return CB;
}
