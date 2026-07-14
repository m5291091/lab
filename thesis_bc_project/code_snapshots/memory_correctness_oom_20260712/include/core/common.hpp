#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <functional>
#include <stack>
#include <queue>
#include <algorithm>
#include <cstdio>

#include <omp.h>

#ifdef __CUDACC__
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <iostream>

#define K_SOURCES_PER_BLOCK 16
#define THREADS_PER_SOURCE  16

#define CUDA_ERR_CHK(err) cuda_check_error(err, __FILE__, __LINE__)
inline void cuda_check_error(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at " << file << ":" << line << ": " 
			<< cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}
}
#endif // __CUDACC__

struct GpuSingleSourceDepResult {
    std::vector<double> delta;
    int reachableCount = 0;
};

#endif // COMMON_HPP
