# CUDA Thread, Block, Tile and Memory Hierarchy

```mermaid
flowchart TB
    subgraph GPU [GPU Device]
        direction TB
        
        %% Global Memory Level
        GM[("Global Memory\n(Accessible by all Blocks & Threads)\nCapacity: GBs\nLatency: High")]
        
        subgraph Grid [Grid Level\nDimensions: gridDim.x, gridDim.y, gridDim.z]
            direction TB
            
            subgraph Block [Block Level\nDimensions: blockDim.x, blockDim.y, blockDim.z]
                direction TB
                
                %% Shared Memory Level
                SM[("Shared Memory\n(Accessible only by Threads in this Block)\nCapacity: KBs\nLatency: Low")]
                
                subgraph Tile [Tile Level\n(Logical sub-grouping of threads, often 2D)]
                    direction LR
                    T1(("Thread (0,0)\nRegisters\nLocal Memory"))
                    T2(("Thread (0,1)\nRegisters\nLocal Memory"))
                    T3(("Thread (1,0)\nRegisters\nLocal Memory"))
                end
                
                Tile <-->|Read / Write| SM
            end
            
            %% Multi-blocks to represent Grid
            subgraph Block2 [Other Block]
                direction TB
                SM2[("Shared Memory")]
                T_other(("Threads"))
            end
        end
        
        Block <-->|Read / Write| GM
        Block2 <-->|Read / Write| GM
    end
```

### ရှင်းလင်းချက်-
1. **Global Memory:**
   - **တည်နေရာ:** GPU ရဲ့ VRAM (Video RAM) ပေါ်မှာ ရှိပါတယ်။
   - **Dimension / Access:** Grid ထဲက ဘယ် Block, ဘယ် Thread မဆို လှမ်းသုံးလို့ရပါတယ်။ Storage အများဆုံးဖြစ်ပေမယ့် Latency အမြင့်ဆုံး ဖြစ်ပါတယ်။

2. **Grid & Block:**
   - **Grid Dimension:** `gridDim.x, gridDim.y, gridDim.z` ဆိုပြီး 1D, 2D, 3D အနေနဲ့ သတ်မှတ်လို့ရပါတယ်။
   - **Block Dimension:** `blockDim.x, blockDim.y, blockDim.z` ဆိုပြီး 1D, 2D, 3D အနေနဲ့ ထပ်ခွဲထားပါတယ်။
   - Grid တစ်ခုထဲမှာ Block ပေါင်းများစွာ ပါဝင်ပါတယ်။

3. **Shared Memory:**
   - **တည်နေရာ:** GPU ရဲ့ Streaming Multiprocessor (SM) ပေါ်မှာ on-chip အနေနဲ့ ရှိပါတယ်။
   - **Dimension / Access:** သက်ဆိုင်ရာ **Block တစ်ခုတည်း** မှာရှိတဲ့ Thread တွေပဲ မျှဝေသုံးစွဲလို့ရပါတယ်။ Global Memory ထက် အများကြီး ပိုမြန်ပါတယ်။

4. **Tile:**
   - Matrix Multiplication လိုမျိုး Algorithm တွေရေးတဲ့အခါ Block ကြီးတစ်ခုလုံးကို Sub-matrix (ဥပမာ 16x16, 32x32) အပိုင်းလေးတွေ Logical ထပ်ခွဲတာကို ခေါ်တာဖြစ်ပါတယ်။ (Hardware အရ သီးသန့်မရှိပါဘူး၊ Shared Memory ကို ထိရောက်စွာသုံးနိုင်ဖို့ Software ပိုင်းကနေ Logical ခွဲခြားတဲ့ သဘောတရားပါ။)

5. **Thread:**
   - **တည်နေရာ:** အသေးဆုံး execution unit ဖြစ်ပါတယ်။
   - **Memory:** Thread တစ်ခုချင်းစီမှာ ကိုယ်ပိုင်အသုံးပြုဖို့ **Registers** တွေနဲ့ **Local Memory** တွေ ပါရှိပါတယ်။ ယင်းတို့ကို အခြား thread တွေက လှမ်းယူကြည့်လို့ မရပါဘူး။
