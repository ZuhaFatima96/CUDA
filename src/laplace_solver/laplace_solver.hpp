#pragma once

#include <vector>
#include <cstddef>
#include <CL/cl.hpp>
#include <cuda_runtime.h>
#include "../utils/timer.hpp"

namespace laplace_solver {

// Constants for boundary conditions
constexpr float TOP_VOLTAGE = 5.0f;      // Top boundary: +5V
constexpr float BOTTOM_VOLTAGE = -5.0f;   // Bottom boundary: -5V
constexpr float LEFT_VOLTAGE = 0.0f;      // Left boundary: 0V
constexpr float RIGHT_VOLTAGE = 0.0f;     // Right boundary: 0V
constexpr float TOLERANCE = 1e-6f;        // Convergence tolerance
constexpr int MAX_ITERATIONS = 10000;     // Maximum iterations

// CPU single-threaded implementation
void solveLaplaceCPU(std::vector<float>& grid, size_t width, size_t height);

// CPU OpenMP implementation
void solveLaplaceOMP(std::vector<float>& grid, size_t width, size_t height);

// Initialize grid with boundary conditions
void initializeGrid(std::vector<float>& grid, size_t width, size_t height);

// CUDA implementation declarations
void initializeCUDA();
void solveLaplaceCUDA(std::vector<float>& grid, size_t width, size_t height);
void solveLaplaceCUDAShared(std::vector<float>& grid, size_t width, size_t height);
void cleanupCUDA();

// OpenCL implementation class
class OpenCLWrapper {
public:
    OpenCLWrapper();
    ~OpenCLWrapper();

    void solveLaplaceOpenCL(std::vector<float>& grid, size_t width, size_t height);
    void solveLaplaceOpenCLLocal(std::vector<float>& grid, size_t width, size_t height);
    void solveLaplaceOpenCLMultiDevice(std::vector<float>& grid, size_t width, size_t height);

private:
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;
    cl::Kernel kernelLocal;
    
    void buildProgram();
    void setupMultiDevice();
};

// Helper functions
float calculateError(const std::vector<float>& old_grid,
                    const std::vector<float>& new_grid);
bool checkConvergence(const std::vector<float>& old_grid,
                     const std::vector<float>& new_grid);

} // namespace laplace_solver
