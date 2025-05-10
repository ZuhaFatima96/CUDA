#include "matrix_addition.hpp"
#include <iostream>

using namespace matrix_addition;

void runTests() {
    // Test parameters
    const size_t rows = 1024;
    const size_t cols = 1024;
    const size_t size = rows * cols;

    // Initialize data
    std::vector<float> A(size), B(size), C_cpu(size), C_omp(size), 
                      C_cuda(size), C_opencl(size);
    utils::initializeMatrix(A, size);
    utils::initializeMatrix(B, size);

    std::vector<std::pair<std::string, double>> results;

    // CPU single-threaded
    {
        utils::Timer timer;
        addMatricesCPU(A, B, C_cpu, rows, cols);
        results.push_back({"CPU (Single-threaded)", timer.elapsed()});
    }

    // CPU OpenMP
    {
        utils::Timer timer;
        addMatricesOMP(A, B, C_omp, rows, cols);
        results.push_back({"CPU (OpenMP)", timer.elapsed()});
        std::cout << "OpenMP result matches CPU: " 
                  << utils::compareResults(C_cpu, C_omp) << std::endl;
    }

    // CUDA
    try {
        initializeCUDA<float>();
        utils::Timer timer;
        addMatricesCUDA(A, B, C_cuda, rows, cols);
        results.push_back({"CUDA", timer.elapsed()});
        std::cout << "CUDA result matches CPU: " 
                  << utils::compareResults(C_cpu, C_cuda) << std::endl;
        cleanupCUDA<float>();
    } catch (const std::exception& e) {
        std::cerr << "CUDA error: " << e.what() << std::endl;
    }

    // OpenCL
    try {
        OpenCLWrapper opencl;
        utils::Timer timer;
        opencl.addMatricesOpenCL(A, B, C_opencl, rows, cols);
        results.push_back({"OpenCL", timer.elapsed()});
        std::cout << "OpenCL result matches CPU: " 
                  << utils::compareResults(C_cpu, C_opencl) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "OpenCL error: " << e.what() << std::endl;
    }

    // Save and plot results
    utils::savePerformanceResults("../results/matrix_addition_performance.csv", results);
    utils::generatePlot("../results/matrix_addition_performance.csv", 
                       "../results/plots/matrix_addition_performance.png");
}

int main() {
    std::cout << "Running matrix addition tests..." << std::endl;
    runTests();
    std::cout << "Tests completed. Check results/plots for performance comparison." << std::endl;
    return 0;
}
