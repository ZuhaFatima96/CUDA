#include "matrix_multiplication.hpp"
#include <iostream>
#include <iomanip>

using namespace matrix_multiplication;

void runTests() {
    // Test parameters
    const size_t M = 1024;  // Matrix A: M x K
    const size_t K = 1024;  // Matrix B: K x N
    const size_t N = 1024;  // Matrix C: M x N

    std::cout << "Matrix dimensions: A(" << M << "x" << K << ") * B("
              << K << "x" << N << ") = C(" << M << "x" << N << ")" << std::endl;

    // Initialize matrices
    std::vector<float> A(M * K), B(K * N), C_cpu(M * N), 
                      C_omp(M * N), C_cuda(M * N), C_opencl(M * N);

    utils::initializeMatrix(A, M * K);
    utils::initializeMatrix(B, K * N);

    std::vector<std::pair<std::string, double>> results;

    // CPU single-threaded
    {
        std::cout << "Running CPU single-threaded implementation..." << std::endl;
        utils::Timer timer;
        multiplyMatricesCPU(A, B, C_cpu, M, N, K);
        double time = timer.elapsed();
        results.push_back({"CPU (Single-threaded)", time});
        std::cout << "Time: " << time << " ms" << std::endl;
    }

    // CPU OpenMP
    {
        std::cout << "Running CPU OpenMP implementation..." << std::endl;
        utils::Timer timer;
        multiplyMatricesOMP(A, B, C_omp, M, N, K);
        double time = timer.elapsed();
        results.push_back({"CPU (OpenMP)", time});
        std::cout << "Time: " << time << " ms" << std::endl;
        std::cout << "Result matches CPU: "
                  << utils::compareResults(C_cpu, C_omp) << std::endl;
    }

    // CUDA
    try {
        std::cout << "Running CUDA implementation..." << std::endl;
        initializeCUDA<float>();
        utils::Timer timer;
        multiplyMatricesCUDA(A, B, C_cuda, M, N, K);
        double time = timer.elapsed();
        results.push_back({"CUDA", time});
        std::cout << "Time: " << time << " ms" << std::endl;
        std::cout << "Result matches CPU: "
                  << utils::compareResults(C_cpu, C_cuda) << std::endl;
        cleanupCUDA<float>();
    } catch (const std::exception& e) {
        std::cerr << "CUDA error: " << e.what() << std::endl;
    }

    // OpenCL
    try {
        std::cout << "Running OpenCL implementation..." << std::endl;
        OpenCLWrapper opencl;
        utils::Timer timer;
        opencl.multiplyMatricesOpenCL(A, B, C_opencl, M, N, K);
        double time = timer.elapsed();
        results.push_back({"OpenCL", time});
        std::cout << "Time: " << time << " ms" << std::endl;
        std::cout << "Result matches CPU: "
                  << utils::compareResults(C_cpu, C_opencl) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "OpenCL error: " << e.what() << std::endl;
    }

    // Save and plot results
    utils::savePerformanceResults("../results/matrix_multiplication_performance.csv",
                                results);
    utils::generatePlot("../results/matrix_multiplication_performance.csv",
                       "../results/plots/matrix_multiplication_performance.png");
}

int main() {
    std::cout << "\n=== Matrix Multiplication Performance Test ===\n" << std::endl;
    
    // Print system information
    std::cout << "Hardware Information:" << std::endl;
    system("lscpu | grep 'Model name\\|CPU(s)\\|Thread'");
    std::cout << "\nGPU Information:" << std::endl;
    system("nvidia-smi -L");
    
    std::cout << "\nRunning tests..." << std::endl;
    runTests();
    
    std::cout << "\nTests completed. Check results/plots directory for performance comparison."
              << std::endl;
    return 0;
}
