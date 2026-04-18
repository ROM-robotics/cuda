# Lesson 1: Hello CUDA

## Program Flow

```mermaid
flowchart TD
    A["main() - CPU"] --> B["hello_kernel<<<2, 4>>>()"]
    B --> C["GPU Launch"]

    C --> D["Block 0"]
    C --> E["Block 1"]

    D --> D0["Thread 0"] & D1["Thread 1"] & D2["Thread 2"] & D3["Thread 3"]
    E --> E0["Thread 0"] & E1["Thread 1"] & E2["Thread 2"] & E3["Thread 3"]

    D0 & D1 & D2 & D3 --> F["printf: Hello from GPU!"]
    E0 & E1 & E2 & E3 --> F

    F --> G["cudaDeviceSynchronize()"]
    G --> H["CPU: GPU အလုပ်ပြီးပါပြီ!"]
```

## Kernel Launch Syntax

```mermaid
flowchart LR
    A["kernel<<<blocks, threads>>>()"] --> B["blocks = 2"]
    A --> C["threads = 4"]
    B --> D["Total threads = 2 × 4 = 8"]
    C --> D
```

## Key Concepts

```mermaid
graph LR
    subgraph Keywords
        A["__global__"] -->|"GPU ပေါ်မှာ run"| B["Kernel Function"]
        C["threadIdx.x"] -->|"Block ထဲက thread index"| D["0, 1, 2, 3"]
        E["blockIdx.x"] -->|"Grid ထဲက block index"| F["0, 1"]
    end
```

## Thread Indexing

| | Thread 0 | Thread 1 | Thread 2 | Thread 3 |
|---|---|---|---|---|
| **Block 0** | blockIdx=0, threadIdx=0 | blockIdx=0, threadIdx=1 | blockIdx=0, threadIdx=2 | blockIdx=0, threadIdx=3 |
| **Block 1** | blockIdx=1, threadIdx=0 | blockIdx=1, threadIdx=1 | blockIdx=1, threadIdx=2 | blockIdx=1, threadIdx=3 |
