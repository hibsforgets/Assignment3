%%cu
#include <stdio.h>
#include <stdlib.h>

__global__ void matrixMultiplication(float *A, float *B, float *C, int J, int K, int L) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < J && col < L) {
        float value = 0.0f;
        for (int k = 0; k < K; k++) {
            value += A[row * K + k] * B[k * L + col];
        }
        C[row * L + col] = value;
    }
}

int main() {
    int J = 1000;
    int K = 1100;
    int L = 1000;

    float *h_A = (float *)malloc(J * K * sizeof(float));
    float *h_B = (float *)malloc(K * L * sizeof(float));
    float *h_C = (float *)malloc(J * L * sizeof(float));
    
    for (int i = 0; i < J * K; i++) {
        h_A[i] = (float)(rand() % 100);
    }
    for (int i = 0; i < K * L; i++) {
        h_B[i] = (float)(rand() % 100);
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, J * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * L * sizeof(float));
    cudaMalloc((void **)&d_C, J * L * sizeof(float));
    
    cudaMemcpy(d_A, h_A, J * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * L * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockDim(32, 32);
    dim3 gridDim((L + blockDim.x - 1) / blockDim.x, (J + blockDim.y - 1) / blockDim.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMultiplication<<<gridDim, blockDim>>>(d_A, d_B, d_C, J, K, L);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(h_C, d_C, J * L * sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU Runtime: %.3f ms\n", milliseconds);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
