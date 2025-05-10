#pragma once

#include <vector>
#include <cstddef>
#include <CL/cl.hpp>
#include <cuda_runtime.h>
#include "../utils/timer.hpp"

namespace matrix_multiplication {

// CPU single-threaded implementation
template<typename T>
void multiplyMatricesCPU(const std::vector<T>& A, const std::vector<T>& B,
                        std::vector<T>& C, size_t M, size_t N, size_t K) {
    // Matrix A: M x K
    // Matrix B: K x N
    // Matrix C: M x N
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// CPU OpenMP implementation
template<typename T>
void multiplyMatricesOMP(const std::vector<T>& A, const std::vector<T>& B,
                        std::vector<T>& C, size_t M, size_t N, size_t K) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// CUDA implementation declarations
template<typename T>
void initializeCUDA();

template<typename T>
void multiplyMatricesCUDA(const std::vector<T>& A, const std::vector<T>& B,
                         std::vector<T>& C, size_t M, size_t N, size_t K);

template<typename T>
void cleanupCUDA();

// OpenCL implementation class
class OpenCLWrapper {
public:
    OpenCLWrapper();
    ~OpenCLWrapper();

    template<typename T>
    void multiplyMatricesOpenCL(const std::vector<T>& A, const std::vector<T>& B,
                               std::vector<T>& C, size_t M, size_t N, size_t K);

private:
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;
    
    void buildProgram();
};

} // namespace matrix_multiplication
