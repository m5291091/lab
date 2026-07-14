#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <string>
#include <vector>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Check last CUDA error
#define CUDA_CHECK_LAST() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Timer class for performance measurement
class Timer {
private:
    cudaEvent_t start, stop;
    bool started;
    
public:
    Timer();
    ~Timer();
    
    void startTimer();
    float stopTimer(); // Returns elapsed time in milliseconds
};

// File I/O utilities
void saveBCToFile(const std::string& filename, 
                  const std::vector<double>& bc,
                  bool verbose = true);

// Print utilities
void printProgress(int current, int total, const std::string& message);
void printVerbose(const std::string& message, bool verbose);

// Device query
void printDeviceInfo();

#endif // UTILS_H