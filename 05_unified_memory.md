# Lesson 5: Unified Memory

## Manual vs Unified Memory

```mermaid
flowchart TD
    subgraph Manual["Manual Memory Management"]
        M1["malloc(h_data)"] --> M2["cudaMalloc(&d_data)"]
        M2 --> M3["Initialize h_data on CPU"]
        M3 --> M4["cudaMemcpy(d ← h, H→D)"]
        M4 --> M5["kernel<<<...>>>(d_data)"]
        M5 --> M6["cudaMemcpy(h ← d, D→H)"]
        M6 --> M7["Read h_data on CPU"]
        M7 --> M8["free(h) + cudaFree(d)"]
    end

    subgraph Unified["Unified Memory"]
        U1["cudaMallocManaged(&data)"] --> U2["Initialize data on CPU"]
        U2 --> U3["kernel<<<...>>>(data)"]
        U3 --> U4["cudaDeviceSynchronize()"]
        U4 --> U5["Read data on CPU"]
        U5 --> U6["cudaFree(data)"]
    end

    style Manual fill:#ffebee
    style Unified fill:#e8f5e9
```

## Page Migration (Internal)

```mermaid
sequenceDiagram
    participant CPU
    participant Driver as CUDA Driver
    participant GPU

    CPU->>Driver: cudaMallocManaged(&data)
    Driver-->>CPU: data pointer (accessible by both)

    CPU->>CPU: data[i] = value (CPU page)

    CPU->>GPU: kernel<<<...>>>(data)
    Note over Driver: Page fault → migrate pages to GPU
    Driver->>GPU: Transfer pages

    GPU->>GPU: Kernel execution

    CPU->>CPU: cudaDeviceSynchronize()
    CPU->>CPU: read data[i]
    Note over Driver: Page fault → migrate back to CPU
    Driver->>CPU: Transfer pages back
```

## Prefetch Optimization

```mermaid
flowchart TD
    subgraph NoPrefetch["Without Prefetch"]
        N1["CPU writes data"] --> N2["Kernel launch"]
        N2 --> N3["⚠️ Page fault!<br/>Migrate on-demand<br/>(slow, many faults)"]
        N3 --> N4["Kernel runs"]
    end

    subgraph WithPrefetch["With Prefetch"]
        P1["CPU writes data"] --> P2["cudaMemPrefetchAsync<br/>(data → GPU)"]
        P2 --> P3["Kernel launch"]
        P3 --> P4["✅ Data already on GPU<br/>(No page faults)"]
    end

    style NoPrefetch fill:#fff3e0
    style WithPrefetch fill:#e8f5e9
```

## Particle Simulation Flow

```mermaid
flowchart TD
    A["cudaMallocManaged(&particles)<br/>Particle struct array"] --> B["CPU: Initialize positions & velocities"]
    B --> C["Simulation Loop × 100"]

    C --> D["cudaMemPrefetchAsync → GPU"]
    D --> E["update_particles<<<blocks, threads>>>"]
    E --> F["cudaDeviceSynchronize()"]
    F -->|"next step"| C

    C -->|"loop done"| G["cudaMemPrefetchAsync → CPU"]
    G --> H["CPU: Read final positions"]
    H --> I["cudaFree(particles)"]

    style E fill:#2196F3,color:#fff
```

## Unified Memory - Struct Support

```mermaid
classDiagram
    class Particle {
        float x, y, z
        float vx, vy, vz
    }

    note for Particle "cudaMallocManaged으로 allocate\nCPU/GPU 양쪽에서 직접 access"
```

> **When to use Unified Memory:**
> - Prototyping / rapid development
> - Complex data structures (linked lists, trees)
> - Data shared frequently between CPU & GPU
>
> **When NOT to use:**
> - Maximum performance needed (manual control better)
> - Predictable access patterns
