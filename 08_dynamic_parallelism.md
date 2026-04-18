# Lesson 8: Dynamic Parallelism & Advanced Patterns

## Part 1: Dynamic Parallelism

```mermaid
flowchart TD
    subgraph Normal["Normal: CPU launches kernels"]
        CPU1["CPU"] -->|"launch"| K1["Kernel A"]
        CPU1 -->|"launch"| K2["Kernel B"]
        CPU1 -->|"launch"| K3["Kernel C"]
    end

    subgraph Dynamic["Dynamic: GPU launches kernels"]
        CPU2["CPU"] -->|"launch"| K4["Parent Kernel"]
        K4 -->|"GPU launch"| K5["Child Kernel 1"]
        K4 -->|"GPU launch"| K6["Child Kernel 2"]
        K5 -->|"GPU launch"| K7["Grandchild"]
    end

    style Dynamic fill:#e8f5e9
```

## GPU Quicksort (Recursive)

```mermaid
flowchart TD
    A["quicksort(data, 0, N-1)"] --> B["Partition around pivot"]
    B --> C["pivot_idx"]

    C --> D["quicksort<<<1,1>>><br/>(data, left, pivot-1)<br/>GPU Child Kernel!"]
    C --> E["quicksort<<<1,1>>><br/>(data, pivot+1, right)<br/>GPU Child Kernel!"]

    D --> F{"size < 32?"}
    E --> G{"size < 32?"}

    F -->|"Yes"| H["Insertion Sort<br/>(single thread)"]
    F -->|"No"| I["Partition → recurse"]

    G -->|"Yes"| J["Insertion Sort<br/>(single thread)"]
    G -->|"No"| K["Partition → recurse"]

    style D fill:#2196F3,color:#fff
    style E fill:#2196F3,color:#fff
```

## Partition Visualization

```mermaid
flowchart TD
    subgraph Before["Before Partition (pivot = 5)"]
        B1["8"] & B2["3"] & B3["7"] & B4["1"] & B5["5"]
    end

    subgraph After["After Partition"]
        A1["3"] & A2["1"] & A3["5"] & A4["8"] & A5["7"]
    end

    Before --> P["Partition"]
    P --> After

    A3 -.- Note["← pivot at index 2"]

    subgraph Left["Left: quicksort(0, 1)"]
        A1 & A2
    end
    subgraph Right["Right: quicksort(3, 4)"]
        A4 & A5
    end

    style A3 fill:#4CAF50,color:#fff
```

## Part 2: Cooperative Groups

```mermaid
flowchart TD
    subgraph Traditional["Traditional Sync"]
        T1["__syncthreads()"] --> T2["Block-level only<br/>⚠️ No sub-group sync"]
    end

    subgraph CG["Cooperative Groups"]
        C1["thread_block"] --> C2["Block sync<br/>block.sync()"]
        C1 --> C3["tiled_partition&lt;32&gt;"] --> C4["Warp sync<br/>warp.sync()"]
        C1 --> C5["tiled_partition&lt;16&gt;"] --> C6["Sub-warp sync"]
        C1 --> C7["tiled_partition&lt;4&gt;"] --> C8["4-thread group"]
    end

    style CG fill:#e8f5e9
```

## Cooperative Groups Reduce

```mermaid
flowchart TD
    A["cg::this_thread_block()"] --> B["cg::tiled_partition<32>(block)"]
    B --> C["Warp of 32 threads"]

    C --> D["val = data[i]"]
    D --> E["warp.shfl_down(val, 16)<br/>val += neighbor at offset 16"]
    E --> F["warp.shfl_down(val, 8)"]
    F --> G["warp.shfl_down(val, 4)"]
    G --> H["warp.shfl_down(val, 2)"]
    H --> I["warp.shfl_down(val, 1)"]
    I --> J{"warp.thread_rank() == 0?"}
    J -->|"Yes"| K["atomicAdd(result, val)"]
    J -->|"No"| L["done"]

    style K fill:#f44336,color:#fff
```

## Part 3: Function Pointers on GPU

```mermaid
flowchart LR
    subgraph Device["GPU Function Pointers"]
        OP["d_ops[] array"]
        OP --> ADD["gpu_add: a + b"]
        OP --> MUL["gpu_mul: a * b"]
        OP --> MAX["gpu_max: max(a,b)"]
    end

    K["apply_op<<<...>>><br/>(a, b, c, n, op_idx)"] --> OP
    IDX["op_idx = 0"] --> ADD
    IDX2["op_idx = 1"] --> MUL
    IDX3["op_idx = 2"] --> MAX
```

## Compile Requirements

```mermaid
flowchart LR
    A["nvcc"] --> B["-rdc=true<br/>Relocatable<br/>Device Code"]
    A --> C["file.cu"]
    A --> D["-lcudadevrt<br/>Device<br/>Runtime Lib"]
    A --> E["-o output"]

    B --> F["Dynamic Parallelism<br/>အတွက် မဖြစ်မနေ လို"]
    D --> F
```

## When to Use Dynamic Parallelism

| Use Case | Example |
|----------|---------|
| Recursive algorithms | Quicksort, tree traversal |
| Adaptive computation | AMR (Adaptive Mesh Refinement) |
| Irregular workloads | Graph algorithms |
| Nested parallelism | Fractal generation |

> ⚠️ **သတိ:** Dynamic Parallelism launch overhead ရှိသည်။ Simple cases တွင် CPU launch + streams ပိုမြန်နိုင်သည်။
