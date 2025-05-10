__kernel void matrixAdd(__global const float* A,
                         __global const float* B,
                         __global float* C,
                         const int rows,
                         const int cols) {
    int idx = get_global_id(0);
    int stride = get_global_size(0);
    
    for (int i = idx; i < rows * cols; i += stride) {
        C[i] = A[i] + B[i];
    }
}

__kernel void matrixAdd2D(__global const float* A,
                         __global const float* B,
                         __global float* C,
                         const int rows,
                         const int cols) {
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}
