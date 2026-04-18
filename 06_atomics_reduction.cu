// ===== CUDA Lesson 6: Atomic Operations & Parallel Reduction =====
// Threads တွေသည် တစ်ပြိုင်နက် data ရေးရင် race condition ဖြစ်နိုင်
// Atomic operations နဲ့ reduction technique သုံးပြီး ဖြေရှင်းမယ်

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

// ===== Part 1: Atomic Operations =====
// Histogram: data ထဲက value တစ်ခုချင်းစီ ဘယ်နှစ်ကြိမ် ပါလဲ ရေတွက်

__global__ void histogram_atomic(int *data, int *hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // atomicAdd = thread-safe ဖြစ်အောင် ပေါင်း
        // atomic မသုံးရင် race condition → wrong result!
        atomicAdd(&hist[data[i]], 1);
    }
}

// ===== Part 2: Parallel Reduction (Sum) =====
// Array ထဲက element အားလုံးကို ပေါင်း
// Tree-based reduction: O(log n) steps

__global__ void reduce_sum(float *input, float *output, int n) {
    extern __shared__ float sdata[];  // dynamic shared memory

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load + first add
    float val = 0.0f;
    if (i < n) val = input[i];
    if (i + blockDim.x < n) val += input[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    // Tree reduction in shared memory
    //   Step 1: stride=256 → thread 0 += thread 256
    //   Step 2: stride=128 → thread 0 += thread 128
    //   ...
    //   Final:  stride=1   → thread 0 += thread 1
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Block result ကို global memory ထဲ ရေး
    if (tid == 0) {
        atomicAdd(output, sdata[0]);  // blocks အကြား atomic add
    }
}

// ===== Part 3: Warp-Level Reduction (Advanced) =====
// Warp = 32 threads, __syncthreads() မလိုဘဲ communicate လုပ်နိုင်
__global__ void warp_reduce_sum(float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (i < n) ? input[i] : 0.0f;

    // Warp shuffle: register level data exchange (အမြန်ဆုံး!)
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Warp ရဲ့ lane 0 (first thread) မှာ warp sum ရှိနေပြီ
    if (threadIdx.x % 32 == 0) {
        atomicAdd(output, val);
    }
}

int main() {
    // ===== Histogram Test =====
    printf("===== Histogram (Atomic) =====\n");
    int N = 100000;
    int num_bins = 10;

    int *h_data = (int *)malloc(N * sizeof(int));
    int *h_hist = (int *)calloc(num_bins, sizeof(int));
    for (int i = 0; i < N; i++) h_data[i] = i % num_bins;

    int *d_data, *d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist, num_bins * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist, 0, num_bins * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

    histogram_atomic<<<(N + 255) / 256, 256>>>(d_data, d_hist, N);
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, num_bins * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_bins; i++)
        printf("  Bin %d: %d\n", i, h_hist[i]);

    // ===== Reduction Test =====
    printf("\n===== Parallel Reduction (Sum) =====\n");
    int M = 1 << 20;
    float *h_input = (float *)malloc(M * sizeof(float));
    for (int i = 0; i < M; i++) h_input[i] = 1.0f;  // sum = M

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, M * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (M + threads * 2 - 1) / (threads * 2);
    reduce_sum<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_output, M);

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  Tree reduction sum: %.0f (expected %d)\n", result, M);

    // Warp reduction test
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
    warp_reduce_sum<<<(M + 255) / 256, 256>>>(d_input, d_output, M);
    CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  Warp reduction sum:  %.0f (expected %d)\n", result, M);

    // Cleanup
    cudaFree(d_data); cudaFree(d_hist);
    cudaFree(d_input); cudaFree(d_output);
    free(h_data); free(h_hist); free(h_input);
    return 0;
}

// Compile: nvcc 06_atomics_reduction.cu -o 06_atomics_reduction
// Run:     ./06_atomics_reduction
//
// ===== Atomic Functions အမျိုးမျိုး =====
// atomicAdd()   - ပေါင်း
// atomicSub()   - နုတ်
// atomicMin()   - minimum
// atomicMax()   - maximum
// atomicAnd()   - bitwise AND
// atomicOr()    - bitwise OR
// atomicXor()   - bitwise XOR
// atomicCAS()   - Compare And Swap (custom atomics တည်ဆောက်နိုင်)
