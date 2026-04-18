// ===== CUDA Lesson 5: Unified Memory =====
// cudaMallocManaged သုံးပြီး CPU/GPU memory ကို အလိုအလျောက် manage လုပ်မယ်
// Manual cudaMemcpy မလိုတော့ဘူး!

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

// ===== Struct with Unified Memory =====
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};

__global__ void update_particles(Particle *particles, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }
}

// ===== Prefetch Hint =====
// GPU/CPU ဘက်ကို data ကြိုပို့ → performance ပိုကောင်း
void prefetch_example(Particle *p, int n, int device) {
    CUDA_CHECK(cudaMemPrefetchAsync(p, n * sizeof(Particle), device));
}

int main() {
    int N = 100000;

    // ===== Unified Memory Allocate =====
    // malloc/cudaMalloc အစား cudaMallocManaged သုံး
    // CPU ရော GPU ရော တိုက်ရိုက် access လုပ်လို့ရ
    Particle *particles;
    CUDA_CHECK(cudaMallocManaged(&particles, N * sizeof(Particle)));

    // CPU ကနေ တိုက်ရိုက် initialize (cudaMemcpy မလို!)
    for (int i = 0; i < N; i++) {
        particles[i].x = 0.0f;
        particles[i].y = 0.0f;
        particles[i].z = 0.0f;
        particles[i].vx = (float)(i % 100) * 0.01f;
        particles[i].vy = (float)(i % 50) * 0.02f;
        particles[i].vz = 1.0f;
    }

    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    float dt = 0.01f;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Simulation loop
    for (int step = 0; step < 100; step++) {
        // GPU ဘက် prefetch (optional, performance hint)
        prefetch_example(particles, N, device);

        update_particles<<<blocks, threads>>>(particles, N, dt);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // CPU ဘက် prefetch
    prefetch_example(particles, N, cudaCpuDeviceId);
    CUDA_CHECK(cudaDeviceSynchronize());

    // CPU ကနေ တိုက်ရိုက် read (cudaMemcpy မလို!)
    printf("Particle 0: (%.4f, %.4f, %.4f)\n",
           particles[0].x, particles[0].y, particles[0].z);
    printf("Particle 99: (%.4f, %.4f, %.4f)\n",
           particles[99].x, particles[99].y, particles[99].z);

    CUDA_CHECK(cudaFree(particles));  // unified free
    return 0;
}

// Compile: nvcc 05_unified_memory.cu -o 05_unified_memory
// Run:     ./05_unified_memory
//
// ===== Unified vs Manual Memory =====
//
// Manual (Lesson 2):            Unified (ဒီ lesson):
//   malloc(h_data)                cudaMallocManaged(&data)
//   cudaMalloc(&d_data)           (မလို)
//   cudaMemcpy(d, h, HtoD)       (auto)
//   kernel<<<...>>>(d_data)       kernel<<<...>>>(data)
//   cudaMemcpy(h, d, DtoH)       cudaDeviceSynchronize()
//   free(h); cudaFree(d);        cudaFree(data)
