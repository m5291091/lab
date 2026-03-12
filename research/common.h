#ifndef COMMON_H
#define COMMON_H

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

#define CUDA_ERR_CHK(err) cuda_check_error(err, __FILE__, __LINE__)
inline void cuda_check_error(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at " << file << ":" << line << ": " 
			<< cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}
}
#endif // __CUDACC__

#endif // COMMON_H
