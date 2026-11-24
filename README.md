# PCA-EXP-5-MATRIX-MULTIPLICATION-USING-CUDA-AY-23-24
<h3>AIM:</h3>
<h3>ENTER YOUR NAME: YASHVANDAN K</h3>
<h3>ENTER YOUR REGISTER NO: 212225100060</h3>
<h3>EX. NO: 5</h3>
<h3>DATE: 24-11-2026</h3>
<h1> <align=center> MATRIX MULTIPLICATION USING CUDA </h3>
  Implement Matrix Multiplication using GPU.</h3>

## AIM:
To perform Matrix Multiplication using CUDA and check its performance with nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Define Constants: Define the size of the matrices (SIZE) and the size of the CUDA blocks (BLOCK_SIZE).
2.	Kernel Function: Define a CUDA kernel function matrixMultiply that performs the matrix multiplication.
3.	In the main function, perform the following steps:
4.	Initialize Matrices: Initialize the input matrices ‘a’ and ‘b’ with some values.
5.	Allocate Device Memory: Allocate memory on the GPU for the input matrices ‘a’ and ‘b’, and the output matrix ‘c’.
6.	Copy Matrices to Device: Copy the input matrices from host (CPU) memory to device (GPU) memory.
7.	Set Grid and Block Sizes: Set the grid and block sizes for the CUDA kernel launch.
8.	Start Timer: Start a timer to measure the execution time of the kernel.
9.	Launch Kernel: Launch the matrixMultiply kernel with the appropriate grid and block sizes, and the input and output matrices as arguments.
10.	Copy Result to Host: After the kernel execution, copy the result matrix from device memory to host memory.
11.	Stop Timer: Stop the timer and calculate the elapsed time.
12.	Print Result: Print the result matrix and the elapsed time.
13.	Free Device Memory: Finally, free the device memory that was allocated for the matrices.
## PROGRAM:
```cuda
%%writefile matrix_mul.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define SIZE 1024        // matrix dimension (SIZE x SIZE)
#define BLOCK_SIZE 16    // threads per block dimension

// Kernel: each thread computes one element of C
__global__ void matrixMultiply(float *a, float *b, float *c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += a[row * n + k] * b[k * n + col];
        c[row * n + col] = sum;
    }
}

// Host (CPU) version for comparison
void matrixMultiplyHost(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++)
                sum += a[i * n + k] * b[k * n + j];
            c[i * n + j] = sum;
        }
}

int main()
{
    size_t bytes = SIZE * SIZE * sizeof(float);

    // Initialize host matrices
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);      // GPU result
    float *h_ref = (float *)malloc(bytes);    // CPU result

    for (int i = 0; i < SIZE * SIZE; i++) {
        h_a[i] = rand() % 10;
        h_b[i] = rand() % 10;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_b, bytes);
    cudaMalloc((void **)&d_c, bytes);

    // Copy inputs to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Grid and block sizes
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Device timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMultiply<<<grid, block>>>(d_a, d_b, d_c, SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float deviceTime = 0;
    cudaEventElapsedTime(&deviceTime, start, stop);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Host timing
    auto t1 = std::chrono::high_resolution_clock::now();
    matrixMultiplyHost(h_a, h_b, h_ref, SIZE);
    auto t2 = std::chrono::high_resolution_clock::now();
    double hostTime = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Verify
    int errors = 0;
    for (int i = 0; i < SIZE * SIZE; i++)
        if (fabs(h_c[i] - h_ref[i]) > 1e-3) errors++;

    // Print top-left 4x4 block of result
    printf("Result matrix (top-left 4x4 of %dx%d):\n", SIZE, SIZE);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)
            printf("%10.1f ", h_c[i * SIZE + j]);
        printf("\n");
    }

    printf("\nVerification        : %s\n", errors == 0 ? "PASSED" : "FAILED");
    printf("Device (GPU) time   : %f ms\n", deviceTime);
    printf("Host (CPU) time     : %f ms\n", hostTime);

    // Free memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c); free(h_ref);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
```

## OUTPUT:
<img width="1058" height="305" alt="Screenshot 2026-11-24 100022" src="https://github.com/user-attachments/assets/99fd319d-0bee-4d9c-ade5-170b633e8cf5" />
<img width="1597" height="660" alt="Screenshot 2026-11-24 100037" src="https://github.com/user-attachments/assets/181f34f7-e7a9-45d7-8d51-3b9c90d7210b" />


## RESULT:
Thus the program has been executed by using CUDA to multiply two matrices. It is observed that there are variations in host and device elapsed time. Device took 7.974592 ms time and host took 9108.776418 ms time.
