# Matrix Addition Module
set(MATRIX_ADDITION_SOURCES
    matrix_addition.cpp
    main.cpp
)

# CUDA sources
cuda_compile(MATRIX_ADDITION_CUDA_OBJ kernels/cuda/matrix_add.cu)

# Create executable
add_executable(matrix_addition_test ${MATRIX_ADDITION_SOURCES} ${MATRIX_ADDITION_CUDA_OBJ})

# Link libraries
target_link_libraries(matrix_addition_test
    ${CUDA_LIBRARIES}
    ${OpenCL_LIBRARIES}
    OpenMP::OpenMP_CXX
)
