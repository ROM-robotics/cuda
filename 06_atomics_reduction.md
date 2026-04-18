# Lesson 6: Atomic Operations & Parallel Reduction

## Part 1: Race Condition Problem

```mermaid
sequenceDiagram
    participant T0 as Thread 0
    participant Mem as hist[3]
    participant T1 as Thread 1

    Note over Mem: hist[3] = 5

    T0->>Mem: Read hist[3] → 5
    T1->>Mem: Read hist[3] → 5
    T0->>Mem: Write hist[3] = 5+1 = 6
    T1->>Mem: Write hist[3] = 5+1 = 6

    Note over Mem: ❌ hist[3] = 6 (should be 7!)
```

## Atomic Operation Solution

```mermaid
sequenceDiagram
    participant T0 as Thread 0
    participant Mem as hist[3]
    participant T1 as Thread 1

    Note over Mem: hist[3] = 5

    T0->>Mem: atomicAdd(&hist[3], 1)
    Note over Mem: Lock → Read 5 → Write 6 → Unlock
    Note over Mem: hist[3] = 6

    T1->>Mem: atomicAdd(&hist[3], 1)
    Note over Mem: Lock → Read 6 → Write 7 → Unlock
    Note over Mem: ✅ hist[3] = 7
```

## Part 2: Tree-Based Parallel Reduction

```mermaid
flowchart TD
    subgraph Step0["Input (8 elements)"]
        A0["4"] & A1["2"] & A2["7"] & A3["1"] & A4["3"] & A5["5"] & A6["8"] & A7["6"]
    end

    subgraph Step1["Step 1: stride=4"]
        B0["4+3=7"] & B1["2+5=7"] & B2["7+8=15"] & B3["1+6=7"]
    end

    subgraph Step2["Step 2: stride=2"]
        C0["7+15=22"] & C1["7+7=14"]
    end

    subgraph Step3["Step 3: stride=1"]
        D0["22+14=36"]
    end

    A0 & A4 --> B0
    A1 & A5 --> B1
    A2 & A6 --> B2
    A3 & A7 --> B3

    B0 & B2 --> C0
    B1 & B3 --> C1

    C0 & C1 --> D0

    style D0 fill:#4CAF50,color:#fff
```

## Reduction in Shared Memory

```mermaid
flowchart TD
    A["Load input → shared memory sdata[]"] --> B["__syncthreads()"]

    B --> C["stride = blockDim.x / 2"]
    C --> D{"tid < stride?"}
    D -->|"Yes"| E["sdata[tid] += sdata[tid + stride]"]
    D -->|"No"| F["idle"]
    E --> G["__syncthreads()"]
    F --> G
    G --> H["stride >>= 1"]
    H --> I{"stride > 0?"}
    I -->|"Yes"| D
    I -->|"No"| J["tid == 0: atomicAdd(output, sdata[0])"]

    style J fill:#f44336,color:#fff
```

## Part 3: Warp Shuffle

```mermaid
flowchart LR
    subgraph Warp["Warp (32 threads) - Register Level"]
        direction TB
        subgraph S1["offset=16"]
            W0["T0: val"] ---|"+ T16"| W0r["T0: sum"]
        end
        subgraph S2["offset=8"]
            W1["T0: sum"] ---|"+ T8"| W1r["T0: sum"]
        end
        subgraph S3["offset=4"]
            W2["T0: sum"] ---|"+ T4"| W2r["T0: sum"]
        end
        subgraph S4["offset=2"]
            W3["T0: sum"] ---|"+ T2"| W3r["T0: sum"]
        end
        subgraph S5["offset=1"]
            W4["T0: sum"] ---|"+ T1"| W4r["T0: total"]
        end
    end
```

> `__shfl_down_sync(mask, val, offset)` — shared memory မလိုဘဲ warp ထဲ data ပေးပို့

## Atomic Functions Reference

| Function | Operation | Example |
|----------|-----------|---------|
| `atomicAdd` | a += b | Histogram, Reduction |
| `atomicSub` | a -= b | Counter decrement |
| `atomicMin` | a = min(a,b) | Find minimum |
| `atomicMax` | a = max(a,b) | Find maximum |
| `atomicCAS` | Compare & Swap | Custom atomic ops |
| `atomicExch` | a = b (atomic) | Lock implementation |
