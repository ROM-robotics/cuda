# Lesson 4: CUDA Streams

## Stream ဆိုတာ ဘာလဲ

```mermaid
flowchart LR
    subgraph Default["Default Stream (Sequential)"]
        direction LR
        A1["Copy H→D"] --> B1["Kernel"] --> C1["Copy D→H"]
        C1 --> A2["Copy H→D"] --> B2["Kernel"] --> C2["Copy D→H"]
    end
```

```mermaid
flowchart LR
    subgraph Streams["Multiple Streams (Overlapped)"]
        direction LR
        subgraph S0["Stream 0"]
            A1["Copy H→D"] --> B1["Kernel"] --> C1["Copy D→H"]
        end
        subgraph S1["Stream 1"]
            A2["Copy H→D"] --> B2["Kernel"] --> C2["Copy D→H"]
        end
        subgraph S2["Stream 2"]
            A3["Copy H→D"] --> B3["Kernel"] --> C3["Copy D→H"]
        end
    end
```

## Timeline Overlap

```mermaid
gantt
    title CUDA Stream Overlap Timeline
    dateFormat X
    axisFormat %s

    section Stream 0
    Copy H→D   :s0c1, 0, 2
    Kernel      :s0k, 2, 5
    Copy D→H   :s0c2, 5, 7

    section Stream 1
    Copy H→D   :s1c1, 1, 3
    Kernel      :s1k, 3, 6
    Copy D→H   :s1c2, 6, 8

    section Stream 2
    Copy H→D   :s2c1, 2, 4
    Kernel      :s2k, 4, 7
    Copy D→H   :s2c2, 7, 9

    section Stream 3
    Copy H→D   :s3c1, 3, 5
    Kernel      :s3k, 5, 8
    Copy D→H   :s3c2, 8, 10
```

## Pinned vs Pageable Memory

```mermaid
flowchart TD
    subgraph Pageable["Pageable Memory (malloc)"]
        P1["CPU RAM<br/>(pageable)"] -->|"1. copy to pinned buffer"| P2["Pinned Buffer<br/>(staging)"]
        P2 -->|"2. DMA transfer"| P3["GPU VRAM"]
    end

    subgraph Pinned["Pinned Memory (cudaMallocHost)"]
        Q1["CPU RAM<br/>(page-locked)"] -->|"DMA transfer တိုက်ရိုက်"| Q2["GPU VRAM"]
    end

    style Pageable fill:#ffebee
    style Pinned fill:#e8f5e9
```

## Program Flow

```mermaid
flowchart TD
    A["cudaMallocHost(&h_data)<br/>Pinned memory allocate"] --> B["cudaMalloc(&d_data)"]
    B --> C["Create 4 streams<br/>cudaStreamCreate()"]

    C --> D["Loop: i = 0..3"]
    D --> E["cudaMemcpyAsync(chunk i, H→D, stream[i])"]
    E --> F["kernel<<<..., stream[i]>>>(chunk i)"]
    F --> G["cudaMemcpyAsync(chunk i, D→H, stream[i])"]
    G -->|"next i"| D

    D -->|"done"| H["cudaDeviceSynchronize()"]
    H --> I["cudaStreamDestroy × 4"]
    I --> J["cudaFreeHost(h_data)"]
```

## Performance Impact

| Method | Data 1GB | Note |
|--------|----------|------|
| **Single stream** | ~100ms | Sequential |
| **4 streams** | ~40ms | Copy + Compute overlap |
| **Pinned + streams** | ~30ms | Fastest transfer + overlap |
