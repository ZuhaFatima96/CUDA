#pragma once

template<typename T>
__global__ void matrixMultiplyKernel(const T* A, const T* B, T* C,
                                    int M, int N, int K) {
    // Calculate thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within bounds
    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Optimized version using shared memory
template<typename T>
__global__ void matrixMultiplySharedKernel(const T* A, const T* B, T* C,
                                          int M, int N, int K) {
    const int TILE_SIZE = 16;  // Must match block size
    
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    T sum = 0;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile into shared memory
        if (row < M && tile * TILE_SIZE + tx < K)
            As[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        else
            As[ty][tx] = 0;
            
        if (tile * TILE_SIZE + ty < K && col < N)
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0;
            
        __syncthreads();
        
        // Compute partial sum for the tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
