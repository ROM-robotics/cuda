# Lesson 7: CUDA Events & Performance Profiling

## CUDA Event Timing

```mermaid
sequenceDiagram
    participant CPU
    participant GPU

    CPU->>GPU: cudaEventRecord(start)
    Note over GPU: ⏱️ Start timestamp

    CPU->>GPU: kernel<<<...>>>() (run 1)
    CPU->>GPU: kernel<<<...>>>() (run 2)
    CPU->>GPU: kernel<<<...>>>() (run N)

    CPU->>GPU: cudaEventRecord(stop)
    Note over GPU: ⏱️ Stop timestamp

    CPU->>GPU: cudaEventSynchronize(stop)
    GPU-->>CPU: Done

    CPU->>CPU: cudaEventElapsedTime(&ms, start, stop)
    Note over CPU: ms = stop - start (milliseconds)
```

## Benchmarking Pattern

```mermaid
flowchart TD
    A["Setup: allocate memory, copy data"] --> B["Warmup Run<br/>(cache, TLB prepare)"]
    B --> C["cudaEventRecord(start)"]
    C --> D["Run kernel × N times"]
    D --> E["cudaEventRecord(stop)"]
    E --> F["cudaEventSynchronize(stop)"]
    F --> G["Calculate average time<br/>ms = elapsed / N"]
    G --> H["Calculate bandwidth<br/>GB/s = bytes / (ms × 1e6)"]

    style B fill:#FF9800,color:#fff
    style G fill:#4CAF50,color:#fff
    style H fill:#2196F3,color:#fff
```

## SAXPY: y = a*x + y

```mermaid
flowchart LR
    subgraph Normal["Normal SAXPY"]
        T0["Thread 0<br/>y[0] = a*x[0]+y[0]"]
        T1["Thread 1<br/>y[1] = a*x[1]+y[1]"]
        T2["Thread 2<br/>y[2] = a*x[2]+y[2]"]
        T3["Thread 3<br/>y[3] = a*x[3]+y[3]"]
    end

    subgraph Unrolled["Unrolled SAXPY (4x)"]
        U0["Thread 0<br/>y[0] = a*x[0]+y[0]<br/>y[1] = a*x[1]+y[1]<br/>y[2] = a*x[2]+y[2]<br/>y[3] = a*x[3]+y[3]"]
    end
```

## Bandwidth Calculation

```mermaid
flowchart TD
    A["SAXPY: y = a*x + y"] --> B["Memory Operations"]
    B --> C["Read x[i] → 1 read"]
    B --> D["Read y[i] → 1 read"]
    B --> E["Write y[i] → 1 write"]
    C & D & E --> F["Total: 3 × N × sizeof(float) bytes"]
    F --> G["Bandwidth = Total bytes / Time"]
    G --> H["GB/s = (3 × N × 4) / (ms × 10⁶)"]

    style H fill:#4CAF50,color:#fff
```

## Thread Count vs Performance

```mermaid
xychart-beta
    title "Threads per Block vs Performance"
    x-axis ["32", "64", "128", "256", "512", "1024"]
    y-axis "Relative Performance" 0 --> 100
    bar [40, 65, 85, 95, 100, 90]
```

## Device Properties (cudaGetDeviceProperties)

```mermaid
flowchart TD
    subgraph GPU["GPU Device"]
        subgraph Info["Device Info"]
            A["name: RTX 4090"]
            B["compute: 8.9"]
            C["multiProcessors: 128 SMs"]
        end
        subgraph Limits["Limits"]
            D["maxThreadsPerBlock: 1024"]
            E["sharedMemPerBlock: 48 KB"]
            F["totalGlobalMem: 24 GB"]
        end
        subgraph BW["Bandwidth"]
            G["memoryBusWidth: 384-bit"]
            H["memoryClockRate: 10.5 GHz"]
            I["Peak BW: 1 TB/s"]
        end
    end
```

## Profiling Tools

| Tool | Purpose | Command |
|------|---------|---------|
| **CUDA Events** | Kernel timing | Code ထဲမှာ ရေး |
| **nsys** | Timeline / system view | `nsys profile ./app` |
| **ncu** | Kernel-level analysis | `ncu --set full ./app` |
| **nvprof** | Legacy profiler | `nvprof ./app` |
