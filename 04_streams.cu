// ===== CUDA Lesson 4: CUDA Streams =====
// Streams သုံးပြီး computation နှင့် data transfer ကို overlap လုပ်မယ်
// Stream = GPU operations တွေကို queue လုပ်တဲ့ sequence

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

__global__ void process_kernel(float *data, int n, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (i < n + offset) {
        // အလုပ်များများ လုပ်ပြီး stream overlap ကို မြင်သာအောင်
        for (int j = 0; j < 100; j++) {
            data[i - offset] = sqrtf(data[i - offset]) * sqrtf(data[i - offset]) + 1.0f;
        }
    }
}

int main() {
    int N = 1 << 20;  // 1M elements
    int num_streams = 4;
    int chunk_size = N / num_streams;
    size_t bytes = N * sizeof(float);
    size_t chunk_bytes = chunk_size * sizeof(float);

    // ===== Pinned Memory =====
    // cudaMallocHost = pinned (page-locked) memory
    // pageable memory ထက် transfer မြန်ပြီး async copy လုပ်လို့ရ
    float *h_data;
    CUDA_CHECK(cudaMallocHost(&h_data, bytes));  // pinned memory!

    for (int i = 0; i < N; i++) h_data[i] = (float)i;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    // ===== Streams ဖန်တီး =====
    cudaStream_t streams[4];
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // ===== Overlap: Copy + Compute =====
    // Stream တစ်ခုစီမှာ: copy → compute → copy back
    // Streams တွေ တစ်ပြိုင်နက် run နိုင်
    int threads = 256;
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;

        // Async copy: CPU → GPU (stream i)
        CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset,
                                    chunk_bytes, cudaMemcpyHostToDevice,
                                    streams[i]));

        // Compute (stream i)
        int blocks = (chunk_size + threads - 1) / threads;
        process_kernel<<<blocks, threads, 0, streams[i]>>>(d_data + offset,
                                                            chunk_size, 0);

        // Async copy: GPU → CPU (stream i)
        CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset,
                                    chunk_bytes, cudaMemcpyDeviceToHost,
                                    streams[i]));
    }

    // Streams အားလုံး ပြီးအောင် စောင့်
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Streams %d ခုနဲ့ processing ပြီးပါပြီ!\n", num_streams);
    printf("Sample: h_data[0]=%.2f, h_data[100]=%.2f\n", h_data[0], h_data[100]);

    // Cleanup
    for (int i = 0; i < num_streams; i++)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    CUDA_CHECK(cudaFreeHost(h_data));  // pinned memory free
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}

// Compile: nvcc 04_streams.cu -o 04_streams
// Run:     ./04_streams
