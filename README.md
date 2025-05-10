# Parallel Computing with CUDA and OpenCL

This repository contains implementations of various parallel computing problems using CUDA, OpenCL, and OpenMP.

## Problems Implemented

1. Matrix Addition (1D and 2D)
   - CUDA Implementation
   - OpenCL Implementation
   - CPU Single-threaded
   - CPU Multi-threaded (OpenMP)

2. Matrix Multiplication
   - CUDA Implementation
   - OpenCL Implementation
   - CPU Single-threaded
   - CPU Multi-threaded (OpenMP)

3. Laplace Solver
   - CUDA Implementation
   - OpenCL Implementation
   - CPU Single-threaded
   - Shared Memory Optimization

4. Heterogeneous Computing (Extra)
   - Multi-device OpenCL Implementation

## Project Structure

```
├── src/
│   ├── matrix_addition/
│   ├── matrix_multiplication/
│   ├── laplace_solver/
│   └── utils/
├── kernels/
│   ├── opencl/
│   └── cuda/
├── results/
│   └── plots/
└── CMakeLists.txt
```

## Build Instructions

### Prerequisites
- CUDA Toolkit
- OpenCL SDK
- CMake (>= 3.10)
- C++ Compiler with OpenMP support

### Building the Project
```bash
mkdir build
cd build
cmake ..
make
```

## Running the Tests
Each executable will be created in the `build/bin` directory.

```bash
# Matrix Addition
./matrix_addition_test

# Matrix Multiplication
./matrix_multiplication_test

# Laplace Solver
./laplace_solver_test

# Heterogeneous Computing
./heterogeneous_test
```