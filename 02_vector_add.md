# Lesson 2: Vector Addition

## CUDA Program ၇ ဆင့် Flow

```mermaid
flowchart TD
    A["Step 1: Host Memory Allocate<br/>malloc(h_a, h_b, h_c)"] --> B["Step 2: Device Memory Allocate<br/>cudaMalloc(d_a, d_b, d_c)"]
    B --> C["Step 3: CPU → GPU Copy<br/>cudaMemcpy(d_a ← h_a, HostToDevice)<br/>cudaMemcpy(d_b ← h_b, HostToDevice)"]
    C --> D["Step 4: Kernel Launch<br/>vector_add<<<blocks, 256>>>(d_a, d_b, d_c, n)"]
    D --> E["Step 5: GPU → CPU Copy<br/>cudaMemcpy(h_c ← d_c, DeviceToHost)"]
    E --> F["Step 6: Verify Result<br/>h_c[i] == h_a[i] + h_b[i] ?"]
    F --> G["Step 7: Free Memory<br/>cudaFree + free"]

    style A fill:#4CAF50,color:#fff
    style B fill:#2196F3,color:#fff
    style C fill:#FF9800,color:#fff
    style D fill:#f44336,color:#fff
    style E fill:#FF9800,color:#fff
    style F fill:#9C27B0,color:#fff
    style G fill:#607D8B,color:#fff
```

## Memory Layout

```mermaid
flowchart LR
    subgraph CPU["Host (CPU + RAM)"]
        h_a["h_a: 0, 1, 2, ..."]
        h_b["h_b: 0, 2, 4, ..."]
        h_c["h_c: result"]
    end

    subgraph GPU["Device (GPU + VRAM)"]
        d_a["d_a"]
        d_b["d_b"]
        d_c["d_c"]
    end

    h_a -->|"cudaMemcpy<br/>HostToDevice"| d_a
    h_b -->|"cudaMemcpy<br/>HostToDevice"| d_b
    d_c -->|"cudaMemcpy<br/>DeviceToHost"| h_c
```

## Thread-to-Element Mapping

```mermaid
flowchart TD
    subgraph Grid
        subgraph Block0["Block 0 (256 threads)"]
            T0["Thread 0<br/>i=0"] & T1["Thread 1<br/>i=1"] & T255["Thread 255<br/>i=255"]
        end
        subgraph Block1["Block 1 (256 threads)"]
            T256["Thread 0<br/>i=256"] & T257["Thread 1<br/>i=257"] & T511["Thread 255<br/>i=511"]
        end
    end

    T0 --> E0["c[0] = a[0] + b[0]"]
    T1 --> E1["c[1] = a[1] + b[1]"]
    T256 --> E256["c[256] = a[256] + b[256]"]
```

## Global Thread Index တွက်ပုံ

```mermaid
flowchart LR
    A["blockIdx.x"] -->|"× blockDim.x"| B["block offset"]
    C["threadIdx.x"] --> D["+"]
    B --> D
    D --> E["global index i"]
    E --> F["c[i] = a[i] + b[i]"]
```

> `i = blockIdx.x * blockDim.x + threadIdx.x`
>
> Block 1, Thread 3: `i = 1 * 256 + 3 = 259`
