// ===== CUDA Lesson 3: Shared Memory & Error Handling =====
// Shared memory ကို သုံးပြီး matrix multiplication လုပ်မယ်

#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

// ===== Matrix Multiply Kernel (Tiled) =====
// Shared memory သုံးပြီး global memory access ကို လျှော့ချတယ်
__global__ void mat_mul(float *A, float *B, float *C, int N) {
    // __shared__ = block ထဲက threads အားလုံး share သုံးတဲ့ memory (fast!)
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Tile တစ်ခုချင်းစီ load ပြီး compute
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Shared memory ထဲ load
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N)
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * TILE_SIZE + threadIdx.y) < N)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        // Threads အားလုံး load ပြီးအောင် စောင့်
        __syncthreads();

        // Compute
        for (int k = 0; k < TILE_SIZE; k++)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        // နောက် tile load မလုပ်ခင် compute ပြီးအောင် စောင့်
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

int main() {
    int N = 512;
    size_t bytes = N * N * sizeof(float);

    // Host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // 2D grid and block
    dim3 threads(TILE_SIZE, TILE_SIZE);  // 16x16 = 256 threads per block
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);

    printf("Matrix %dx%d multiply...\n", N, N);
    mat_mul<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());  // kernel launch error စစ်

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify: 1.0 * 2.0 * N = 1024.0 ဖြစ်ရမယ်
    int correct = 1;
    for (int i = 0; i < N * N; i++) {
        if (h_C[i] != N * 2.0f) { correct = 0; break; }
    }
    printf("Result: %s (expected %.0f, got %.0f)\n",
           correct ? "CORRECT ✓" : "WRONG ✗", N * 2.0f, h_C[0]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}

// Compile: nvcc 03_matrix_mul.cu -o 03_matrix_mul
// Run:     ./03_matrix_mul
