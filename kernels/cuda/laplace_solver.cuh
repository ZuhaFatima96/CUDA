#pragma once

// Regular CUDA implementation
__global__ void laplaceSolverKernel(float* grid, float* new_grid,
                                   int width, int height,
                                   float top, float bottom,
                                   float left, float right) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
        // Apply 5-point stencil
        new_grid[row * width + col] = 0.25f * (
            grid[(row - 1) * width + col] +  // Top
            grid[(row + 1) * width + col] +  // Bottom
            grid[row * width + (col - 1)] +  // Left
            grid[row * width + (col + 1)]    // Right
        );
    }
    else {
        // Handle boundary conditions
        if (row == 0)          new_grid[row * width + col] = top;    // Top boundary
        if (row == height - 1) new_grid[row * width + col] = bottom; // Bottom boundary
        if (col == 0)          new_grid[row * width + col] = left;   // Left boundary
        if (col == width - 1)  new_grid[row * width + col] = right;  // Right boundary
    }
}

// Shared memory implementation
template<int BLOCK_SIZE>
__global__ void laplaceSolverSharedKernel(float* grid, float* new_grid,
                                         int width, int height,
                                         float top, float bottom,
                                         float left, float right) {
    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    
    // Load the main cell
    if (row < height && col < width) {
        tile[ty][tx] = grid[row * width + col];
    }
    
    // Load halo cells
    if (threadIdx.y == 0 && row > 0) {
        tile[0][tx] = grid[(row - 1) * width + col];
    }
    if (threadIdx.y == BLOCK_SIZE - 1 && row < height - 1) {
        tile[BLOCK_SIZE + 1][tx] = grid[(row + 1) * width + col];
    }
    if (threadIdx.x == 0 && col > 0) {
        tile[ty][0] = grid[row * width + (col - 1)];
    }
    if (threadIdx.x == BLOCK_SIZE - 1 && col < width - 1) {
        tile[ty][BLOCK_SIZE + 1] = grid[row * width + (col + 1)];
    }
    
    __syncthreads();
    
    if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
        // Apply 5-point stencil using shared memory
        new_grid[row * width + col] = 0.25f * (
            tile[ty - 1][tx] +  // Top
            tile[ty + 1][tx] +  // Bottom
            tile[ty][tx - 1] +  // Left
            tile[ty][tx + 1]    // Right
        );
    }
    else {
        // Handle boundary conditions
        if (row == 0)          new_grid[row * width + col] = top;
        if (row == height - 1) new_grid[row * width + col] = bottom;
        if (col == 0)          new_grid[row * width + col] = left;
        if (col == width - 1)  new_grid[row * width + col] = right;
    }
}
