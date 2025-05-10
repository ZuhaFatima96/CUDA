#include "matrix_addition.hpp"
#include "../../kernels/cuda/matrix_add.cuh"
#include <stdexcept>
#include <fstream>
#include <sstream>

namespace matrix_addition {

// CUDA Implementation
template<typename T>
void initializeCUDA() {
    cudaError_t error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to initialize CUDA device");
    }
}

template<typename T>
void addMatricesCUDA(const std::vector<T>& A, const std::vector<T>& B,
                     std::vector<T>& C, size_t rows, size_t cols) {
    size_t size = rows * cols * sizeof(T);
    T *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy input data to device
    cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    matrixAdd2DKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);

    // Copy result back to host
    cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);

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
    std::ifstream kernel_file("../kernels/opencl/matrix_add.cl");
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
    kernel = cl::Kernel(program, "matrixAdd2D");
}

template<typename T>
void OpenCLWrapper::addMatricesOpenCL(const std::vector<T>& A, const std::vector<T>& B,
                                     std::vector<T>& C, size_t rows, size_t cols) {
    // Create buffers
    cl::Buffer buffer_A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(T) * A.size(), (void*)A.data());
    cl::Buffer buffer_B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(T) * B.size(), (void*)B.data());
    cl::Buffer buffer_C(context, CL_MEM_WRITE_ONLY,
                       sizeof(T) * C.size());

    // Set kernel arguments
    kernel.setArg(0, buffer_A);
    kernel.setArg(1, buffer_B);
    kernel.setArg(2, buffer_C);
    kernel.setArg(3, (int)rows);
    kernel.setArg(4, (int)cols);

    // Execute kernel
    cl::NDRange global(cols, rows);
    cl::NDRange local(16, 16);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

    // Read result
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(T) * C.size(), C.data());
}

// Explicit template instantiations
template void addMatricesCPU<float>(const std::vector<float>&, const std::vector<float>&,
                                   std::vector<float>&, size_t, size_t);
template void addMatricesOMP<float>(const std::vector<float>&, const std::vector<float>&,
                                   std::vector<float>&, size_t, size_t);
template void initializeCUDA<float>();
template void addMatricesCUDA<float>(const std::vector<float>&, const std::vector<float>&,
                                    std::vector<float>&, size_t, size_t);
template void cleanupCUDA<float>();
template void OpenCLWrapper::addMatricesOpenCL<float>(const std::vector<float>&,
                                                     const std::vector<float>&,
                                                     std::vector<float>&, size_t, size_t);

} // namespace matrix_addition
