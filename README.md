# CUDA Programming သင်ခန်းစာ

CUDA (Compute Unified Device Architecture) သည် NVIDIA GPU များပေါ်တွင် parallel computing လုပ်ဆောင်နိုင်စေသော platform ဖြစ်ပါသည်။

## အခြေခံ သဘောတရား

| CPU | GPU |
|-----|-----|
| Core နည်း (4-64) | Core များ (thousands) |
| Sequential tasks ကောင်း | Parallel tasks ကောင်း |
| Low latency | High throughput |

## CUDA Program ဖွဲ့စည်းပုံ

- **Host** = CPU + RAM
- **Device** = GPU + VRAM
- **Kernel** = GPU ပေါ်မှာ run တဲ့ function (`__global__`)

## သင်ခန်းစာများ

### Lesson 1: `01_hello_cuda.cu` — Hello CUDA

- `__global__` keyword → GPU kernel ကြေညာခြင်း
- `<<<blocks, threads>>>` → kernel launch syntax
- `threadIdx.x`, `blockIdx.x` → thread/block ID များ

### Lesson 2: `02_vector_add.cu` — Vector Addition

CUDA program ရေးသားပုံ အဆင့် ၇ ဆင့်:

1. Host memory allocate (`malloc`)
2. Device memory allocate (`cudaMalloc`)
3. CPU → GPU copy (`cudaMemcpy`)
4. Kernel launch
5. GPU → CPU copy
6. ရလဒ် verify
7. Memory free

### Lesson 3: `03_matrix_mul.cu` — Shared Memory & Tiled Matrix Multiply

- `__shared__` memory → block အတွင်း threads များ share သုံးတဲ့ fast memory
- `__syncthreads()` → thread synchronization
- `dim3` → 2D grid/block configuration
- Error checking macro

---

## Advanced သင်ခန်းစာများ

### Lesson 4: `04_streams.cu` — CUDA Streams

- `cudaStream_t` → operations တွေကို queue စီရန်
- `cudaMemcpyAsync()` → non-blocking data transfer
- `cudaMallocHost()` → pinned (page-locked) memory, async transfer အတွက် လိုအပ်
- Computation + Data Transfer overlap → throughput တိုးမြင့်

### Lesson 5: `05_unified_memory.cu` — Unified Memory

- `cudaMallocManaged()` → CPU/GPU အလိုအလျောက် memory management
- `cudaMemcpy` manually မခေါ်ရတော့
- `cudaMemPrefetchAsync()` → performance hint
- Struct/class တွေကိုလည်း unified memory နဲ့ သုံးနိုင်

### Lesson 6: `06_atomics_reduction.cu` — Atomic Operations & Reduction

- `atomicAdd()`, `atomicMin()`, `atomicCAS()` → thread-safe operations
- Histogram computing (real-world use case)
- Tree-based parallel reduction → $O(\log n)$ steps
- `__shfl_down_sync()` → warp-level shuffle (register level, အမြန်ဆုံး)

### Lesson 7: `07_events_profiling.cu` — CUDA Events & Profiling

- `cudaEvent_t` → GPU timing (microsecond precision)
- `cudaGetDeviceProperties()` → GPU hardware info
- Memory bandwidth calculation
- Thread count tuning & loop unrolling optimization
- `nsys` / `ncu` profiling tools

### Lesson 8: `08_dynamic_parallelism.cu` — Dynamic Parallelism & Advanced Patterns

- GPU kernel ထဲကနေ kernel launch (recursive algorithms)
- Cooperative Groups (`cooperative_groups.h`) → flexible thread grouping
- Function pointers on GPU
- Compile: `nvcc -rdc=true file.cu -lcudadevrt`

---

## Compile & Run နည်း

```bash
cd ~/Desktop/cuda

# Basic lessons
nvcc 01_hello_cuda.cu -o 01_hello && ./01_hello
nvcc 02_vector_add.cu -o 02_vector_add && ./02_vector_add
nvcc 03_matrix_mul.cu -o 03_matrix_mul && ./03_matrix_mul

# Advanced lessons
nvcc 04_streams.cu -o 04_streams && ./04_streams
nvcc 05_unified_memory.cu -o 05_unified_memory && ./05_unified_memory
nvcc 06_atomics_reduction.cu -o 06_atomics_reduction && ./06_atomics_reduction
nvcc 07_events_profiling.cu -o 07_events_profiling && ./07_events_profiling
nvcc -rdc=true 08_dynamic_parallelism.cu -o 08_dynamic -lcudadevrt && ./08_dynamic
```

## Memory Hierarchy (မှတ်ထားရန်)

| Memory | Speed | Scope | Keyword |
|--------|-------|-------|---------|
| Register | အမြန်ဆုံး | Thread တစ်ခုတည်း | (auto) |
| Shared | မြန် | Block တစ်ခုလုံး | `__shared__` |
| Global | နှေး | GPU တစ်ခုလုံး | `cudaMalloc` |
| Unified | နှေး (auto-managed) | CPU + GPU | `cudaMallocManaged` |
| Pinned | Host memory (fast transfer) | CPU | `cudaMallocHost` |

## CUDA Execution Model

```
Grid
├── Block (0,0)
│   ├── Warp 0: Thread 0-31
│   ├── Warp 1: Thread 32-63
│   └── ...
├── Block (0,1)
│   └── ...
└── ...
```

- **Thread** → တစ်ခုချင်း အလုပ်လုပ်
- **Warp** → 32 threads (hardware scheduling unit)
- **Block** → threads group (shared memory share)
- **Grid** → blocks အားလုံး

## Optimization Tips

1. **Memory coalescing** — adjacent threads က adjacent memory ဖတ်ပါ
2. **Occupancy** — SM တစ်ခုမှာ active warps များများထားပါ
3. **Divergence ရှောင်ပါ** — warp ထဲက threads တွေ if/else ခွဲမသွားအောင်
4. **Shared memory** — global memory access ကို cache လုပ်ပါ
5. **Pinned memory** — host↔device transfer မြန်ဖို့ သုံးပါ
6. **Streams** — compute နဲ့ transfer overlap လုပ်ပါ
