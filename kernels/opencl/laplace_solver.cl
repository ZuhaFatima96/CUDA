// Regular OpenCL implementation
__kernel void laplaceSolver(__global float* grid,
                           __global float* new_grid,
                           const int width,
                           const int height,
                           const float top,
                           const float bottom,
                           const float left,
                           const float right) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    
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
        if (row == 0)          new_grid[row * width + col] = top;
        if (row == height - 1) new_grid[row * width + col] = bottom;
        if (col == 0)          new_grid[row * width + col] = left;
        if (col == width - 1)  new_grid[row * width + col] = right;
    }
}

// Local memory implementation
__kernel void laplaceSolverLocal(__global float* grid,
                                __global float* new_grid,
                                const int width,
                                const int height,
                                const float top,
                                const float bottom,
                                const float left,
                                const float right,
                                __local float* tile) {
    const int BLOCK_SIZE = 16;  // Must match work-group size
    
    int col = get_global_id(0);
    int row = get_global_id(1);
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int tile_width = BLOCK_SIZE + 2;
    
    // Convert 2D local index to 1D for the tile
    int local_idx = (ty + 1) * tile_width + (tx + 1);
    
    // Load the main cell
    if (row < height && col < width) {
        tile[local_idx] = grid[row * width + col];
    }
    
    // Load halo cells
    if (ty == 0 && row > 0) {
        tile[tx + 1] = grid[(row - 1) * width + col];
    }
    if (ty == BLOCK_SIZE - 1 && row < height - 1) {
        tile[(BLOCK_SIZE + 1) * tile_width + tx + 1] = grid[(row + 1) * width + col];
    }
    if (tx == 0 && col > 0) {
        tile[(ty + 1) * tile_width] = grid[row * width + (col - 1)];
    }
    if (tx == BLOCK_SIZE - 1 && col < width - 1) {
        tile[(ty + 1) * tile_width + BLOCK_SIZE + 1] = grid[row * width + (col + 1)];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
        // Apply 5-point stencil using local memory
        new_grid[row * width + col] = 0.25f * (
            tile[(ty) * tile_width + (tx + 1)] +        // Top
            tile[(ty + 2) * tile_width + (tx + 1)] +    // Bottom
            tile[(ty + 1) * tile_width + tx] +          // Left
            tile[(ty + 1) * tile_width + (tx + 2)]      // Right
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

// Multi-device support
__kernel void laplaceSolverMultiDevice(__global float* grid,
                                      __global float* new_grid,
                                      const int width,
                                      const int height,
                                      const int start_row,
                                      const int end_row,
                                      const float top,
                                      const float bottom,
                                      const float left,
                                      const float right) {
    int col = get_global_id(0);
    int row = get_global_id(1) + start_row;
    
    if (row > start_row && row < end_row - 1 && col > 0 && col < width - 1) {
        // Apply 5-point stencil
        new_grid[(row - start_row) * width + col] = 0.25f * (
            grid[(row - 1 - start_row) * width + col] +  // Top
            grid[(row + 1 - start_row) * width + col] +  // Bottom
            grid[(row - start_row) * width + (col - 1)] +  // Left
            grid[(row - start_row) * width + (col + 1)]    // Right
        );
    }
    else {
        // Handle boundary conditions
        if (row == 0)          new_grid[(row - start_row) * width + col] = top;
        if (row == height - 1) new_grid[(row - start_row) * width + col] = bottom;
        if (col == 0)          new_grid[(row - start_row) * width + col] = left;
        if (col == width - 1)  new_grid[(row - start_row) * width + col] = right;
    }
}
