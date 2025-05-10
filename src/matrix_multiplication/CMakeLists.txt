# Matrix Multiplication Module
set(MATRIX_MULTIPLICATION_SOURCES
    matrix_multiplication.cpp
    main.cpp
)

# CUDA sources
cuda_compile(MATRIX_MULTIPLICATION_CUDA_OBJ kernels/cuda/matrix_multiply.cu)

# Create executable
add_executable(matrix_multiplication_test 
    ${MATRIX_MULTIPLICATION_SOURCES}
    ${MATRIX_MULTIPLICATION_CUDA_OBJ}
)

# Link libraries
target_link_libraries(matrix_multiplication_test
    ${CUDA_LIBRARIES}
    ${OpenCL_LIBRARIES}
    OpenMP::OpenMP_CXX
)
