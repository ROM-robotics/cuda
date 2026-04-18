// ===== CUDA Lesson 8: Dynamic Parallelism & Advanced Patterns =====
// GPU kernel ထဲကနေ နောက်ထပ် kernel တွေ launch လုပ်နိုင်
// Recursive algorithms, adaptive computation တွေအတွက် သင့်တော်

#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

// ===== Part 1: Dynamic Parallelism =====
// Quicksort ကို GPU ပေါ်မှာ recursive လုပ်
__device__ void swap(int *a, int *b) {
    int t = *a; *a = *b; *b = t;
}

__global__ void gpu_quicksort(int *data, int left, int right) {
    if (left >= right) return;

    // Small arrays → single thread sort (insertion sort)
    if (right - left < 32) {
        if (threadIdx.x == 0) {
            for (int i = left + 1; i <= right; i++) {
                int key = data[i];
                int j = i - 1;
                while (j >= left && data[j] > key) {
                    data[j + 1] = data[j];
                    j--;
                }
                data[j + 1] = key;
            }
        }
        return;
    }

    // Partition
    if (threadIdx.x == 0) {
        int pivot = data[right];
        int i = left - 1;
        for (int j = left; j < right; j++) {
            if (data[j] <= pivot) {
                i++;
                swap(&data[i], &data[j]);
            }
        }
        swap(&data[i + 1], &data[right]);
        int pivot_idx = i + 1;

        // Child kernels launch (Dynamic Parallelism!)
        if (pivot_idx - 1 > left)
            gpu_quicksort<<<1, 1>>>(data, left, pivot_idx - 1);
        if (pivot_idx + 1 < right)
            gpu_quicksort<<<1, 1>>>(data, pivot_idx + 1, right);
        cudaDeviceSynchronize();  // device-side sync
    }
}

// ===== Part 2: Cooperative Groups (CUDA 9+) =====
// Thread groups ကို flexible ဖန်တီးနိုင်
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cooperative_reduce(float *data, float *result, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? data[i] : 0.0f;

    // Warp-level reduce using cooperative groups
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        val += warp.shfl_down(val, offset);
    }

    // Warp leader writes result
    if (warp.thread_rank() == 0) {
        atomicAdd(result, val);
    }
}

// ===== Part 3: Function Pointers on GPU =====
typedef float (*OpFunc)(float, float);

__device__ float gpu_add(float a, float b) { return a + b; }
__device__ float gpu_mul(float a, float b) { return a * b; }
__device__ float gpu_max(float a, float b) { return fmaxf(a, b); }

__device__ OpFunc d_ops[] = {gpu_add, gpu_mul, gpu_max};

__global__ void apply_op(float *a, float *b, float *c, int n, int op_idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = d_ops[op_idx](a[i], b[i]);
    }
}

int main() {
    // ===== Dynamic Parallelism Test =====
    printf("===== Dynamic Parallelism: GPU Quicksort =====\n");
    int N = 1024;
    int *h_data = (int *)malloc(N * sizeof(int));
    srand(42);
    for (int i = 0; i < N; i++) h_data[i] = rand() % 10000;

    printf("Before: %d %d %d %d ...\n", h_data[0], h_data[1], h_data[2], h_data[3]);

    int *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

    gpu_quicksort<<<1, 1>>>(d_data, 0, N - 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));
    printf("After:  %d %d %d %d ... %d %d\n",
           h_data[0], h_data[1], h_data[2], h_data[3],
           h_data[N - 2], h_data[N - 1]);

    // Verify sorted
    int sorted = 1;
    for (int i = 1; i < N; i++) {
        if (h_data[i] < h_data[i - 1]) { sorted = 0; break; }
    }
    printf("Sorted: %s\n", sorted ? "YES ✓" : "NO ✗");

    // ===== Cooperative Groups Test =====
    printf("\n===== Cooperative Groups Reduce =====\n");
    int M = 1 << 16;
    float *h_fdata = (float *)malloc(M * sizeof(float));
    for (int i = 0; i < M; i++) h_fdata[i] = 1.0f;

    float *d_fdata, *d_result;
    CUDA_CHECK(cudaMalloc(&d_fdata, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fdata, h_fdata, M * sizeof(float), cudaMemcpyHostToDevice));

    cooperative_reduce<<<(M + 255) / 256, 256>>>(d_fdata, d_result, M);

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Sum: %.0f (expected %d)\n", result, M);

    // Cleanup
    cudaFree(d_data); cudaFree(d_fdata); cudaFree(d_result);
    free(h_data); free(h_fdata);
    return 0;
}

// Compile: nvcc -rdc=true 08_dynamic_parallelism.cu -o 08_dynamic -lcudadevrt
//   -rdc=true  = relocatable device code (dynamic parallelism အတွက် လို)
//   -lcudadevrt = device runtime library
// Run:     ./08_dynamic
