// ===== CUDA Lesson 1: Hello CUDA =====
// GPU ပေါ်မှာ function တစ်ခု run ကြည့်မယ်

#include <stdio.h>

// __global__ keyword က GPU ပေါ်မှာ run မယ့် function (kernel) ကို ပြောတာ
__global__ void hello_kernel() {
    // threadIdx.x = လက်ရှိ thread ရဲ့ index
    printf("Hello from GPU! Thread %d, Block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    // kernel<<<blocks, threads_per_block>>>()
    // Block 2 ခု, Thread 4 ခုစီ = စုစုပေါင်း 8 threads
    hello_kernel<<<2, 4>>>();

    // GPU အလုပ်ပြီးအောင် စောင့်
    cudaDeviceSynchronize();

    printf("CPU: GPU အလုပ်ပြီးပါပြီ!\n");
    return 0;
}

// Compile: nvcc 01_hello_cuda.cu -o 01_hello
// Run:     ./01_hello
