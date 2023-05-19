%%cu
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 32
#define J 1000
#define K 1100
#define L 1000

__global__ void matrixMultiplication(float *A, float *B, float *C, int rows_A, int cols_A, int cols_B) {
    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < (cols_A + TILE_WIDTH - 1) / TILE_WIDTH; tile++) {
        if (tile * TILE_WIDTH + threadIdx.x < cols_A && row < rows_A) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * cols_A + tile * TILE_WIDTH + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (tile * TILE_WIDTH + threadIdx.y < cols_A && col < cols_B) {
            shared_B[threadIdx.y][threadIdx.x] = B[(tile * TILE_WIDTH + threadIdx.y) * cols_B + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < rows_A && col < cols_B) {
        C[row * cols_B + col] = sum;
    }
}

int main() {
    // Allocate memory for matrices A, B, and C on the host
    float *h_A = (float *)malloc(J * K * sizeof(float));
    float *h_B = (float *)malloc(K * L * sizeof(float));
    float *h_C = (float *)malloc(J * L * sizeof(float));

    // Initialize matrices A and B with random values
    for (int i = 0; i < J * K; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < K * L; i++) {
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate memory for matrices A, B, and C on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, J * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * L * sizeof(float));
    cudaMalloc((void **)&d_C, J * L * sizeof(float));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, J * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * L * sizeof(float), cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((L + blockSize.x - 1) / blockSize.x, (J + blockSize.y - 1) / blockSize.y);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the matrix multiplication kernel on the device
    matrixMultiplication<<<gridSize, blockSize>>>(d_A, d_B, d_C, J, K, L);

    // Record the stop event
    cudaEventRecord(stop);

    // Synchronize to wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the result matrix C from device to host
    cudaMemcpy(h_C, d_C, J * L * sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU Runtime: %.3f ms\n", milliseconds);

    // Free memory on host and device
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
