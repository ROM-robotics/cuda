// ===== CUDA Lesson 7: CUDA Events & Performance Profiling =====
// Kernel execution time ကို တိကျစွာ တိုင်းတာမယ်
// Memory bandwidth ကို တွက်ချက်မယ်

#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

// Test kernels
__global__ void saxpy(float *y, float *x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

__global__ void saxpy_unrolled(float *y, float *x, float a, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    // Loop unrolling: thread တစ်ခုက element 4 ခု process
    if (i + 3 < n) {
        y[i]     = a * x[i]     + y[i];
        y[i + 1] = a * x[i + 1] + y[i + 1];
        y[i + 2] = a * x[i + 2] + y[i + 2];
        y[i + 3] = a * x[i + 3] + y[i + 3];
    }
}

// ===== Timing Function =====
float time_kernel(void (*launch)(float*, float*, float, int, int),
                  float *d_y, float *d_x, float a, int n, int threads,
                  int num_runs) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    launch(d_y, d_x, a, n, threads);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        launch(d_y, d_x, a, n, threads);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / num_runs;  // average per run
}

void launch_saxpy(float *y, float *x, float a, int n, int threads) {
    int blocks = (n + threads - 1) / threads;
    saxpy<<<blocks, threads>>>(y, x, a, n);
}

void launch_saxpy_unrolled(float *y, float *x, float a, int n, int threads) {
    int blocks = (n / 4 + threads - 1) / threads;
    saxpy_unrolled<<<blocks, threads>>>(y, x, a, n);
}

int main() {
    // ===== Device Properties =====
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("===== Device Info =====\n");
    printf("  Device:           %s\n", prop.name);
    printf("  Compute:          %d.%d\n", prop.major, prop.minor);
    printf("  SMs:              %d\n", prop.multiProcessorCount);
    printf("  Max threads/block: %d\n", prop.maxThreadsPerBlock);
    printf("  Shared mem/block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Global mem:       %.0f MB\n", prop.totalGlobalMem / 1048576.0);
    printf("  Memory Bus:       %d-bit\n", prop.memoryBusWidth);
    printf("  Bandwidth:        %.0f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

    // ===== Benchmark Setup =====
    int N = 1 << 24;  // 16M elements
    size_t bytes = N * sizeof(float);
    int num_runs = 100;

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) { h_x[i] = 1.0f; h_y[i] = 2.0f; }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    printf("\n===== SAXPY Benchmark (N=%d) =====\n", N);

    // ===== Different thread counts test =====
    int thread_counts[] = {32, 64, 128, 256, 512, 1024};
    printf("\n--- Thread Count Comparison ---\n");
    for (int t = 0; t < 6; t++) {
        float ms = time_kernel(launch_saxpy, d_y, d_x, 2.0f, N,
                               thread_counts[t], num_runs);
        float gb = 3.0f * bytes / (ms * 1e6);  // 2 reads + 1 write
        printf("  Threads=%4d: %.3f ms, %.1f GB/s\n",
               thread_counts[t], ms, gb);
    }

    // ===== Normal vs Unrolled =====
    printf("\n--- Normal vs Unrolled ---\n");
    float ms1 = time_kernel(launch_saxpy, d_y, d_x, 2.0f, N, 256, num_runs);
    float ms2 = time_kernel(launch_saxpy_unrolled, d_y, d_x, 2.0f, N, 256, num_runs);
    printf("  Normal:   %.3f ms\n", ms1);
    printf("  Unrolled: %.3f ms (%.1fx speedup)\n", ms2, ms1 / ms2);

    // Cleanup
    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y);
    return 0;
}

// Compile: nvcc 07_events_profiling.cu -o 07_events_profiling
// Run:     ./07_events_profiling
//
// ===== nsys/ncu Profiling Tools =====
// nsys profile ./07_events_profiling      # Timeline view
// ncu --set full ./07_events_profiling    # Kernel analysis
