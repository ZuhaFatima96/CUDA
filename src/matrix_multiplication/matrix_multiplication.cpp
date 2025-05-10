#include "matrix_multiplication.hpp"
#include "../../kernels/cuda/matrix_multiply.cuh"
#include <stdexcept>
#include <fstream>
#include <sstream>

namespace matrix_multiplication {

// CUDA Implementation
template<typename T>
void initializeCUDA() {
    cudaError_t error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to initialize CUDA device");
    }
}

template<typename T>
void multiplyMatricesCUDA(const std::vector<T>& A, const std::vector<T>& B,
                         std::vector<T>& C, size_t M, size_t N, size_t K) {
    // Allocate device memory
    T *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(T));
    cudaMalloc(&d_B, K * N * sizeof(T));
    cudaMalloc(&d_C, M * N * sizeof(T));

    // Copy input data to device
    cudaMemcpy(d_A, A.data(), M * K * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(T), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    dim3 blockSize(16, 16);  // Optimal for most cases
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);

    // Copy result back to host
    cudaMemcpy(C.data(), d_C, M * N * sizeof(T), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

template<typename T>
void cleanupCUDA() {
    cudaDeviceReset();
}

// OpenCL Implementation
OpenCLWrapper::OpenCLWrapper() {
    // Get platform
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found");
    }
    platform = platforms[0];

    // Get device
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.empty()) {
        throw std::runtime_error("No OpenCL devices found");
    }
    device = devices[0];

    // Create context and command queue
    context = cl::Context({device});
    queue = cl::CommandQueue(context, device);

    buildProgram();
}

OpenCLWrapper::~OpenCLWrapper() {
    // OpenCL cleanup is handled by RAII
}

void OpenCLWrapper::buildProgram() {
    // Read kernel source
    std::ifstream kernel_file("../kernels/opencl/matrix_multiply.cl");
    std::string kernel_str((std::istreambuf_iterator<char>(kernel_file)),
                          std::istreambuf_iterator<char>());

    cl::Program::Sources sources;
    sources.push_back({kernel_str.c_str(), kernel_str.length()});

    // Build program
    program = cl::Program(context, sources);
    if (program.build({device}) != CL_SUCCESS) {
        throw std::runtime_error("Error building OpenCL program: " + 
                               program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
    }

    // Create kernel
    kernel = cl::Kernel(program, "matrixMultiply");
}

template<typename T>
void OpenCLWrapper::multiplyMatricesOpenCL(const std::vector<T>& A,
                                          const std::vector<T>& B,
                                          std::vector<T>& C,
                                          size_t M, size_t N, size_t K) {
    // Create buffers
    cl::Buffer buffer_A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       M * K * sizeof(T), (void*)A.data());
    cl::Buffer buffer_B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       K * N * sizeof(T), (void*)B.data());
    cl::Buffer buffer_C(context, CL_MEM_WRITE_ONLY,
                       M * N * sizeof(T));

    // Set kernel arguments
    kernel.setArg(0, buffer_A);
    kernel.setArg(1, buffer_B);
    kernel.setArg(2, buffer_C);
    kernel.setArg(3, (int)M);
    kernel.setArg(4, (int)N);
    kernel.setArg(5, (int)K);

    // Execute kernel
    cl::NDRange global(N, M);  // Global work size
    cl::NDRange local(16, 16); // Work group size
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

    // Read result
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0,
                          M * N * sizeof(T), C.data());
}

// Explicit template instantiations
template void multiplyMatricesCPU<float>(const std::vector<float>&,
                                        const std::vector<float>&,
                                        std::vector<float>&, size_t, size_t, size_t);
template void multiplyMatricesOMP<float>(const std::vector<float>&,
                                        const std::vector<float>&,
                                        std::vector<float>&, size_t, size_t, size_t);
template void initializeCUDA<float>();
template void multiplyMatricesCUDA<float>(const std::vector<float>&,
                                         const std::vector<float>&,
                                         std::vector<float>&, size_t, size_t, size_t);
template void cleanupCUDA<float>();
template void OpenCLWrapper::multiplyMatricesOpenCL<float>(const std::vector<float>&,
                                                          const std::vector<float>&,
                                                          std::vector<float>&,
                                                          size_t, size_t, size_t);

} // namespace matrix_multiplication
