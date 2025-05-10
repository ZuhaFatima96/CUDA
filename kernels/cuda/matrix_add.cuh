template<typename T>
__global__ void matrixAddKernel(const T* A, const T* B, T* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < rows * cols; i += stride) {
        C[i] = A[i] + B[i];
    }
}

template<typename T>
__global__ void matrixAdd2DKernel(const T* A, const T* B, T* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}
