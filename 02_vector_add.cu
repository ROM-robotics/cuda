// ===== CUDA Lesson 2: Vector Addition =====
// Array နှစ်ခုကို ပေါင်းမယ် - CUDA ရဲ့ classic ဥပမာ

#include <stdio.h>
#include <stdlib.h>

// GPU Kernel: element တစ်ခုချင်းစီကို thread တစ်ခုစီက ပေါင်းမယ်
__global__ void vector_add(float *a, float *b, float *c, int n) {
    // Global thread index တွက်ပုံ
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Array အပြင်ဘက် မရောက်အောင် စစ်
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1000000;  // element 1 million
    size_t bytes = n * sizeof(float);

    // ===== Step 1: Host (CPU) memory allocate =====
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    // Data ထည့်
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    // ===== Step 2: Device (GPU) memory allocate =====
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // ===== Step 3: CPU -> GPU data copy =====
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // ===== Step 4: Kernel launch =====
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;  // ceiling division

    printf("Launching %d blocks x %d threads\n", blocks, threads_per_block);
    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);

    // ===== Step 5: GPU -> CPU result copy =====
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // ===== Step 6: ရလဒ် စစ်ဆေး =====
    int correct = 1;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            correct = 0;
            break;
        }
    }
    printf("Result: %s\n", correct ? "CORRECT ✓" : "WRONG ✗");

    // ===== Step 7: Memory free =====
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return 0;
}

// Compile: nvcc 02_vector_add.cu -o 02_vector_add
// Run:     ./02_vector_add
