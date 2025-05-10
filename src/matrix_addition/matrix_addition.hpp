#pragma once

#include <vector>
#include <cstddef>
#include <CL/cl.hpp>
#include <cuda_runtime.h>
#include "../utils/timer.hpp"

namespace matrix_addition {

// CPU single-threaded implementation
template<typename T>
void addMatricesCPU(const std::vector<T>& A, const std::vector<T>& B, 
                    std::vector<T>& C, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t idx = i * cols + j;
            C[idx] = A[idx] + B[idx];
        }
    }
}

// CPU OpenMP implementation
template<typename T>
void addMatricesOMP(const std::vector<T>& A, const std::vector<T>& B, 
                    std::vector<T>& C, size_t rows, size_t cols) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t idx = i * cols + j;
            C[idx] = A[idx] + B[idx];
        }
    }
}

// CUDA implementation declarations
template<typename T>
void initializeCUDA();

template<typename T>
void addMatricesCUDA(const std::vector<T>& A, const std::vector<T>& B,
                     std::vector<T>& C, size_t rows, size_t cols);

template<typename T>
void cleanupCUDA();

// OpenCL implementation declarations
class OpenCLWrapper {
public:
    OpenCLWrapper();
    ~OpenCLWrapper();

    template<typename T>
    void addMatricesOpenCL(const std::vector<T>& A, const std::vector<T>& B,
                          std::vector<T>& C, size_t rows, size_t cols);

private:
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;
    
    void buildProgram();
};

} // namespace matrix_addition
