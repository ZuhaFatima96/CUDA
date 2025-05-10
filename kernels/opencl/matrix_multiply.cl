__kernel void matrixMultiply(__global const float* A,
                            __global const float* B,
                            __global float* C,
                            const int M,
                            const int N,
                            const int K) {
    // Get thread indices
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    // Check bounds
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Optimized version using local memory
__kernel void matrixMultiplyLocal(__global const float* A,
                                 __global const float* B,
                                 __global float* C,
                                 const int M,
                                 const int N,
                                 const int K) {
    const int TILE_SIZE = 16;  // Must match work-group size
    
    // Local memory tiles
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = get_global_id(1);
    int col = get_global_id(0);
    int local_row = get_local_id(1);
    int local_col = get_local_id(0);
    
    float sum = 0.0f;
    
    // Loop over tiles
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile) {
        // Load tile into local memory
        int tile_row = row;
        int tile_col = tile * TILE_SIZE + local_col;
        if (tile_row < M && tile_col < K)
            As[local_row][local_col] = A[tile_row * K + tile_col];
        else
            As[local_row][local_col] = 0.0f;
            
        tile_row = tile * TILE_SIZE + local_row;
        tile_col = col;
        if (tile_row < K && tile_col < N)
            Bs[local_row][local_col] = B[tile_row * N + tile_col];
        else
            Bs[local_row][local_col] = 0.0f;
            
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial sum for the tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[local_row][k] * Bs[k][local_col];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
