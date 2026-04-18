# Lesson 3: Tiled Matrix Multiplication

## Shared Memory ဘာလို့ သုံးလဲ

```mermaid
flowchart TD
    subgraph Without["❌ Without Tiling"]
        A1["Thread (row, col)"] -->|"row × N reads"| G1["Global Memory<br/>(နှေး ~400 cycles)"]
    end

    subgraph With["✅ With Tiling"]
        A2["Thread (row, col)"] -->|"fast read"| S1["Shared Memory<br/>(မြန် ~5 cycles)"]
        S1 -->|"tile တစ်ခုစီ load"| G2["Global Memory"]
    end
```

## Tiled Computation Flow

```mermaid
flowchart TD
    Start["C[row][col] = 0"] --> Loop["For each tile t = 0, 1, 2, ..."]

    Loop --> Load["1. Global → Shared Memory Load<br/>tile_A[ty][tx] = A[row][t*TILE + tx]<br/>tile_B[ty][tx] = B[t*TILE + ty][col]"]

    Load --> Sync1["2. __syncthreads()<br/>Threads အားလုံး load ပြီးအောင် စောင့်"]

    Sync1 --> Compute["3. Compute partial sum<br/>for k=0..TILE-1:<br/>  sum += tile_A[ty][k] * tile_B[k][tx]"]

    Compute --> Sync2["4. __syncthreads()<br/>Compute ပြီးမှ နောက် tile load"]

    Sync2 -->|"နောက် tile"| Loop
    Sync2 -->|"tiles ကုန်ပြီ"| Write["C[row][col] = sum"]

    style Load fill:#2196F3,color:#fff
    style Sync1 fill:#FF9800,color:#fff
    style Compute fill:#4CAF50,color:#fff
    style Sync2 fill:#FF9800,color:#fff
```

## Tile Loading Visualization (TILE_SIZE=4 ဥပမာ)

```mermaid
block-beta
    columns 8

    space:4 B["Matrix B"]:4
    space:4 b0["col 0"] b1["col 1"] b2["col 2"] b3["col 3"]

    A["Matrix A"]:1 a0["row 0"]:1 space:2 tb0["■"] tb1["□"] tb2["□"] tb3["□"]
    space:1 a1["row 1"]:1 space:2 tb4["□"] tb5["■"] tb6["□"] tb7["□"]
    space:1 a2["row 2"]:1 space:2 tb8["□"] tb9["□"] tb10["■"] tb11["□"]
    space:1 a3["row 3"]:1 space:2 tb12["□"] tb13["□"] tb14["□"] tb15["■"]
```

## 2D Grid/Block Configuration

```mermaid
flowchart TD
    subgraph Grid["Grid (2D blocks)"]
        subgraph B00["Block(0,0)<br/>16×16 threads"]
            T00["C[0..15][0..15]"]
        end
        subgraph B01["Block(0,1)<br/>16×16 threads"]
            T01["C[0..15][16..31]"]
        end
        subgraph B10["Block(1,0)<br/>16×16 threads"]
            T10["C[16..31][0..15]"]
        end
        subgraph B11["Block(1,1)<br/>16×16 threads"]
            T11["C[16..31][16..31]"]
        end
    end
```

## Memory Access Comparison

| | Global Memory Only | Tiled (Shared Memory) |
|---|---|---|
| **Global reads per element** | 2N | 2N / TILE_SIZE |
| **N=512, TILE=16** | 1024 reads | 64 reads |
| **Speedup** | 1× | **16×** |
