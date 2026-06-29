# Lecture 12: AI与数据中心映射优化

**Stanford CS149, Fall 2025**

---

## 目录

1. [课程主题与概述](#1-课程主题与概述)
2. [专用AI硬件架构](#2-专用ai硬件架构)
3. [内存技术：HBM与3D堆叠](#3-内存技术hbm与3d堆叠)
4. [数据流架构与编程模型](#4-数据流架构与编程模型)
5. [SambaNova SN40L RDU架构](#5-sambanova-sn40l-rdu架构)
6. [元流水线（Metapipelining）](#6-元流水线metapipelining)
7. [FlashAttention与核融合](#7-flashattention与核融合)
8. [数据中心架构：DGX SuperPOD](#8-数据中心架构dgx-superpod)
9. [消息传递通信原语](#9-消息传递通信原语)
10. [AI模型中的并行策略](#10-ai模型中的并行策略)
11. [分布式矩阵乘法与计算-通信重叠](#11-分布式矩阵乘法与计算-通信重叠)
12. [流水线并行与训练](#12-流水线并行与训练)
13. [能效优化与数据移动成本](#13-能效优化与数据移动成本)
14. [DRAM工作原理深入](#14-dram工作原理深入)
15. [总结：内存瓶颈与未来方向](#15-总结内存瓶颈与未来方向)

---

## 1. 课程主题与概述

本节课围绕以下核心问题展开：

- **如何为深度神经网络（DNN）设计专用硬件？**
- **如何为专用硬件编程？**

涉及的三种代表性硬件架构：

| 硬件 | 核心特征 |
|------|----------|
| **Google TPU** | 高效稠密矩阵乘法 => 脉动阵列（Systolic Array） |
| **NVIDIA H100 / B100** | 异步计算与内存机制 => 编程复杂，使用Thunderkittens DSL简化 |
| **SambaNova SN40L** | 数据流架构，编程模型基于分块（Tiling）和流式处理（Streaming）+ 元流水线 |

---

## 2. 专用AI硬件架构

### 2.1 AI模型本质：数据流图

AI模型本质上是**数据流图（Dataflow Graph）**：

```
Input Sample --> GEMM 1 --> Pool --> GEMM 2 --> SoftMax --> Sum --> Output
                   ^                                                      ^
                Weights                                               Weights
```

数据流图由以下基本模式组成：
- **map**（映射）
- **filter**（过滤）
- **reduce**（归约）
- **GEMM**（通用矩阵乘法）—— 这些属于**并行模式（Parallel Patterns）**

### 2.2 从数据流图到可重构数据流架构

可重构数据流架构（Reconfigurable Dataflow Architecture）将AI模型的数据流图直接映射到硬件上，核心组件包括：

- **PMU**（Pattern Memory Unit）：模式内存单元
- **PCU**（Pattern Compute Unit）：模式计算单元
- **S（Switch）**：片上交换网络

Plasticine（ISCA 2017，Prabhakar, Zhang等）是此类架构的代表作。

### 2.3 可重构数据流架构 vs 理想加速器

| 特性 | 为什么重要？ |
|------|-------------|
| 无指令开销 | 无指令取指/解码开销 |
| **分块张量**（Tiled tensors，如 16x16, 32x32） | 最大化GEMM的TFLOPS性能 |
| **异步计算**（Asynchronous compute） | 重叠计算与内存访问 |
| **异步内存访问**（Asynchronous memory access） | 重叠计算与内存访问 |
| **异步芯片间通信**（Asynchronous chip-to-chip comm.） | 重叠计算、内存与通信 |
| **计算单元间通信**（CU-to-CU comm.） | 支持算子融合与流水线化 |
| **流式数据流**（Streaming Dataflow） | 极端异步性：无需顺序指令执行 |

---

## 3. 内存技术：HBM与3D堆叠

### 3.1 CPU vs GPU 内存对比

```
CPU:  64-bit 内存总线 --> 传统DRAM
GPU:  1024-bit 内存总线 --> HBM（高带宽内存）
```

### 3.2 3D堆叠技术

**核心技术**：通过硅通孔（Through-Silicon-Vias, TSVs）实现DRAM芯片的3D垂直堆叠。

架构层次：
1. **逻辑层（Logic Layer）**：位于堆叠底部，作为内存控制器，处理来自处理器的请求
2. **硅中介层（Silicon Interposer）**：作为DRAM堆叠与处理器之间的高带宽互连

**代表技术**：
- **Micron/Intel HMC**（Hybrid Memory Cube，混合内存立方体）
- **HBM**（High-Bandwidth Memory）：1024-bit接口连接堆叠

### 3.3 HBM三大优势

1. **更高带宽**：1024-bit超宽接口
2. **高能效比**：数据移动距离更短，功耗更低
3. **小尺寸**：3D堆叠大幅减少封装面积

### 3.4 GPU采用HBM的演进历程

| GPU型号（年份） | 内存接口 | 峰值带宽 | 容量 |
|----------------|---------|---------|------|
| AMD Radeon Fury（2015） | 4 x HBM = 4096-bit | 512 GB/s | - |
| NVIDIA P100（2016） | 4 x HBM2 = 4096-bit | 720 GB/s | 4 x 4GB = 16GB |
| NVIDIA H100（2022） | 6 x HBM3 = 6144-bit | **3.2 TB/s** | 80GB |

### 3.5 HBM4：自定义逻辑芯片

HBM4进一步引入自定义逻辑芯片（Custom Logic Die），具备：

- LPDDR接口
- I/O接口（Ethernet、PCI）
- 可能的计算能力：SRAM缓存、KV缓存压缩

---

## 4. 数据流架构与编程模型

### 4.1 能否用更简单的编程模型实现异步？

**核心理念**：采用**以数据为中心的视角（data-centric view）**。

### 4.2 数据流编程 = 数据并行模式

编程流程：

1. **分块（Tiling）**：将大张量分解为小分块
2. **并行化（Parallelization）**：并行执行各分块
3. **元流水线（Metapipelining）**：流水线化分块执行
4. **布局与布线（Place & Route）**：将计算映射到硬件
5. **代码生成（Codegen）**：生成最终执行代码

**可组合的计算原语**：MM（矩阵乘法）、Map、Zip、Reduce、Gather、Scatter等

这些原语支持在**空间和时间上的灵活调度**，实现**空间执行（spatial execution）**。

### 4.3 流式数据流 => 核融合（Kernel Fusion）

在GPU上，Attention算法通常被分解为多个独立的kernel调用，导致：
- 核融合程度低
- 数据局部性差
- 大量的kernel启动和同步开销

在数据流架构上，可以将整个Attention操作融合为一个数据流图，实现**粗粒度流水线**，大幅减少数据在片外存储和片上存储之间的传输。

---

## 5. SambaNova SN40L RDU架构

### 5.1 硬件规格

| 规格 | 数值 |
|------|------|
| PCU + PMU 总数 | **1,040个** |
| 算力（bf16） | **638 TFLOPS** |
| 片上SRAM | **520 MB** |
| HBM | **64 GB** |
| DDR | **1.5 TB** |

### 5.2 核心组件详解

- **PCU（Pattern Compute Unit）**：模式计算单元
  - 包含脉动阵列和SIMD计算单元（16 x 8 bf16格式）
  
- **PMU（Pattern Memory Unit）**：模式内存单元
  - 高地址生成灵活性和带宽（0.5 MB）
  
- **S（Mesh Switches）**：网格交换开关
  - 提供高片上互连灵活性和带宽
  
- **AGCU（Address Generator and Coalescing Unit）**：地址生成与合并单元
  - 片外内存和IO的门户

### 5.3 关键优势：520MB vs 100MB SRAM

SN40L片上SRAM（520MB）是H100（约100MB）的**5倍**，这为核融合提供了巨大优势：
- 数据流融合可以消除**GB级别的片外中间结果流量**
- 更高的数据局部性
- 零额外的kernel启动开销

---

## 6. 元流水线（Metapipelining）

### 6.1 基本概念

**元流水线**是一种**分层粗粒度流水线**（"流水线的流水线"），核心思想：
- 将并行模式（循环）转换为流式流水线
- 在循环体内插入流水线阶段
- 各流水线阶段并行执行
- 重叠执行多个循环迭代

### 6.2 关键特性

- 阶段间中间数据存储在**双缓冲区（Double Buffers）**中
- 能够处理**执行时间不均衡的阶段**
- 配合分块（Tiling）和融合（Fusion）效果很好
- 缓冲区可用于改变数据访问模式（如数据转置）
- **元流水线可以在融合（fusion）不可行的情况下工作**

### 6.3 矩阵乘法元流水线示例

```cpp
auto format = DataFormat::kBF16;
int64_t M = args::M.getValue();
int64_t N = args::N.getValue();
int64_t K = args::K.getValue();
auto A = INPUT_REGION("A", (M, K), format);
auto B = INPUT_REGION("B", (K, N), format);
auto C = OUTPUT_REGION("C", (M, N), format);
auto MM = 256; // M方向的Tile大小，假设能整除M
auto NN =  64; // N方向的Tile大小，假设能整除N
auto a_tile_shape = std::vector<int64_t>({MM, K});
auto b_tile_shape = std::vector<int64_t>({K, NN});
auto c_tile_shape = std::vector<int64_t>({MM, NN});

METAPIPE(M / MM, [&]() {
    auto a_tile = LOAD_TILE(A, a_tile_shape);
    METAPIPE(N / NN, [&]() {
        auto b_tile = LOAD_TILE(B, b_tile_shape, row_par = 4);
        auto c = MAT_MUL(a_tile, b_tile);
        auto c_tile = BUFFER(c);
        STORE_TILE(C, c_tile);
    });
});
```

### 6.4 元流水线的执行流程

```
METAPIPE(M, MM) {
   a_tile = LOAD_TILE(A, a_tile_shape)     // 从片外加载A的分块
   METAPIPE(N, NN) {
      b_tile = LOAD_TILE(B, b_tile_shape)   // 从片外加载B的分块
      c = MAT_MUL(a_tile, b_tile, row_par = 4)  // 计算矩阵乘法
      c_tile = BUFFER(c)                    // 缓冲结果
      STORE_TILE(C, c_tile)                 // 结果写回片外
   }
}
```

**数据流映射**：
- `a_tile` 由AGCU从片外加载到PMU
- `b_tile` 由AGCU从片外加载到PMU
- PCU执行矩阵乘法
- `c_tile` 通过AGCU写回片外

### 6.5 GDA（高斯判别分析）元流水线示例

元流水线的一个具体应用是GDA算法的加速：

```
map(N) { r =>
    row = matrix.slice(r)
    diff = map(D) { i => row(i) - sub(i) }
    vprod = map(D, D) { (i, j) => diff(i) * diff(j) }
}
```

该四阶段流水线（Pipe1-Pipe4）将：
- 数据加载（ld）
- 差值计算（diff/sub）
- 外积计算（vprod）
- 结果存储（st）

全部流水线化执行。

---

## 7. FlashAttention与核融合

### 7.1 Llama 3.1 8B 解码器结构

每个Decoder层包含：

```
RMS Norm -> Q GEMM | K GEMM | V GEMM
         -> QK matmul (transpose) -> Scale -> Maskfill -> Softmax
         -> PV matmul -> O GEMM
         -> RMS Norm -> Gate GEMM | Up GEMM
         -> SilU * Mul -> Down GEMM -> Add
```

层间通过 **AllReduce** 进行通信。

### 7.2 GPU上的限制核融合

使用TensorRT-LLM在GPU上运行Llama 3.1 8B：
- **低核融合**：每个操作对应独立的kernel调用
- **低数据局部性**：频繁的HBM读写
- **高启动和同步开销**：每个Decoder约800次kernel调用

### 7.3 RDU上的全核融合

在SN40L RDU上：
- **整个Decoder融合为一个kernel**：一次调用完成所有操作
- **高数据局部性**：520MB SRAM优势（vs H100的100MB）
- **零额外kernel启动开销**：每个token仅3次调用（vs GPU的约800次）
- **Kernel调用减少100倍**
- 数据流融合消除GB级别的片外中间结果流量

### 7.4 Kernel Loop优化

在传统GPU上，每个Decoder需要：
- 一次kernel启动
- 权重加载
- 计算
- 同步

在RDU上，所有Decoder合并为一个kernel循环：
- **将异步内存和计算完全重叠**
- HBM带宽持续被利用（权重加载与计算完全重叠）
- 无需频繁的kernel启动和同步

### 7.5 FlashAttention在RDU上的实现

FlashAttention采用分块（Tiling）策略，在RDU上通过数据流执行：
- Q和K的分块通过PMU加载
- PCU执行 QK^T 矩阵乘法
- 通过PMU传递Mask、Softmax、Dropout等中间结果
- 最终 PV 乘法得到输出

**关键特点**：通过token控制的流式数据流执行，实现**无锁同步**。

### 7.6 数据流带来的高性能

通过数据流架构可以实现：
- 计算、内存访问、芯片间通信的**完全重叠**
- **AllReduce与权重加载和计算完全重叠**
- **AllReduce不消耗HBM容量或带宽**

---

## 8. 数据中心架构：DGX SuperPOD

### 8.1 模组化架构

**1K GPU SuperPOD集群**配置：
- **140个DGX A100节点**（共1,120个GPU）
- **第一层快速存储**：DDN AI400x with Lustre
- **Mellanox HDR 200Gb/s InfiniBand**：完整Fat-tree拓扑
- 网络针对AI和HPC优化

### 8.2 DGX A100单节点配置

- 2x AMD 7742 EPYC CPU + 8x A100 GPU
- **NVLINK 3.0全连接交换机**
- 8个计算端口 + 2个存储端口的HDR InfiniBand

### 8.3 互连拓扑

Fat-tree结构由以下层次组成：
1. **Leaf Switches（叶交换机）**：直接连接DGX节点
2. **Spine Switches（脊交换机）**：连接叶交换机
3. **Distributed Core Switches（分布式核心交换机）**：连接脊交换机
4. **存储网络**独立于计算网络

支持**自适应路由**和**SharpV2**（用于集合通信卸载）。

### 8.4 Scale-Up vs Scale-Out

- **Scale-Up（垂直扩展）**：在单个节点内通过NVLINK等高带宽互连扩展
- **Scale-Out（水平扩展）**：通过InfiniBand等网络连接更多节点

---

## 9. 消息传递通信原语

### 9.1 AllReduce

AllReduce是分布式训练中最关键的集合通信操作，可以被分解为两个步骤：

```
AllReduce = ReduceScatter + AllGather
```

其中 `rank = accelerator node`（加速器节点编号）。

**工作原理**：
1. **ReduceScatter**：每个节点贡献部分数据，各节点获得最终归约结果的一部分
2. **AllGather**：每个节点将自己的部分结果广播给所有节点

### 9.2 All-to-All

All-to-All是另一种重要的通信模式：

```
输入（4个rank，每个有4个分片）:
rank 0: [A0, A1, A2, A3]
rank 1: [B0, B1, B2, B3]
rank 2: [C0, C1, C2, C3]
rank 3: [D0, D1, D2, D3]

输出（转置后）:
rank 0: [A0, B0, C0, D0]
rank 1: [A1, B1, C1, D1]
rank 2: [A2, B2, C2, D2]
rank 3: [A3, B3, C3, D3]
```

All-to-All在**专家并行（Expert Parallelism）**中广泛使用。

---

## 10. AI模型中的并行策略

### 10.1 并行维度总览

Transformer模型的张量具有以下维度：
- **batch_dim**（批次维度）
- **sequence_dim**（序列维度）
- **hidden_dim**（隐藏层维度）

激活张量和权重张量分布在多个维度上。

### 10.2 六种并行策略

| 并行策略 | 英文名称 | 分片维度 |
|----------|----------|---------|
| **数据并行（DP）** | Data Parallel | batch_dim |
| **张量并行（TP）** | Tensor Parallel | hidden_dim |
| **流水线并行（PP）** | Pipeline Parallel | layer_dim |
| **序列并行（SP）** | Sequence Parallel | sequence_dim |
| **上下文并行（CP）** | Context Parallel | sequence_dim |
| **专家并行（EP）** | Expert Parallel | MoE expert维度 |

### 10.3 并行策略与通信原语的对应关系

```
数据并行（DP）     --> ReduceScatter + AllGather，或 AllReduce
张量并行（TP）     --> ReduceScatter + AllGather，或 AllReduce
流水线并行（PP）   --> Send/Receive（点对点通信）
专家并行（EP）     --> All-to-All
```

- DP和TP属于**模型并行（Model Parallelism）**范畴
- PP和EP使用不同的通信模式

### 10.4 MoE（混合专家）模型的并行策略

MoE模型的并行包括：
- **专家并行（EP）**：将不同专家分布在不同的加速器上
- 计算和通信模式与传统的密集模型有所不同
- 使用All-to-All通信将token发送到对应的专家

---

## 11. 分布式矩阵乘法与计算-通信重叠

### 11.1 分布式矩阵乘法示例

**问题描述**：
- 矩阵乘法：`inputA[M x K] * inputB[K x N] = out[M x N]`
- 具体维度：BS=16, M=24576, K=131072, N=8192
- 在S个RDU上分布式计算

**映射策略**：沿K维度分配到S个RDU

```
每个RDU的计算量：
  本地的A: [M x K/S]
  本地的B: [K/S x N]
  本地输出: [M x N]（部分结果）

最终：S个 [M x N] 的部分结果通过 ReduceScatter 合并
```

**以4个RDU为例**：

```
A [M x K] 和 B [K x N]
     ↓
Split B along K dimension:
  B_0 [K/4 x N], B_1 [K/4 x N], B_2 [K/4 x N], B_3 [K/4 x N]
     ↓
GEMM on each RDU:
  RDU 0: A [M x K/4] * B_0 [K/4 x N] => out_0 [M x N]
  RDU 1: A [M x K/4] * B_1 [K/4 x N] => out_1 [M x N]
  RDU 2: A [M x K/4] * B_2 [K/4 x N] => out_2 [M x N]
  RDU 3: A [M x K/4] * B_3 [K/4 x N] => out_3 [M x N]
     ↓
Reduce-Scatter => out [M x N]（分配给各RDU）
```

### 11.2 计算-通信重叠：Llama 3.1 8B 示例

在RDU上：
- **高算子融合**：所有Decoder合并为一个kernel调用
- **零kernel启动和同步开销**
- **高数据局部性**
- **流水线化的AllReduce与计算完全重叠，不产生HBM流量**

具体实现流程：
```
Down GEMM --> Buffer --> Add --> AllReduce（通过AGCU/PCU/PMU流水线）
                                    ↓
                              与RDU1通信
```

在每个RDU内部，计算和通信通过AGCU、PCU、PMU的流水线结构完全重叠。

### 11.3 重叠的重要性：概念分析

- 不使用重叠（GPU模式）：通信时间随节点数增加而增加，最终通信成为瓶颈
- 使用重叠（RDU模式）：计算和通信并发执行，最大化资源利用率

随着RDU数量增加：
- **8节点**：无重叠时理论利用率为 88.5%
- **16节点**：无重叠时理论利用率为 77%
- **32节点**：无重叠时理论利用率为 52%（通信严重成为瓶颈）

### 11.4 重叠的重要性：量化分析

**基准测试**：BS=16, M=24576, K=131072, N=8192
**总计算量**：844.44 TFLOPs（总基准）

| RDU数量 | 系统总算力 | @100%计算时间 | ReduceScatter时间 @100%链路 | 无重叠理论利用率 | 有重叠实测利用率 |
|---------|-----------|--------------|---------------------------|-----------------|-----------------|
| 8       | 12,744 TF | 66.3 ms      | 8.6 ms                    | 88.5%           | **72%**         |
| 16      | 25,488 TF | 33.1 ms      | 9.7 ms                    | 77%             | **75%**         |
| 32      | 50,976 TF | 16.5 ms      | 15.0 ms                   | 52%             | **79%**         |

**关键发现**：由于计算与通信的重叠，即使在32个RDU的规模下，仍能维持**70%以上的利用率**。

---

## 12. 流水线并行与训练

### 12.1 流水线并行的挑战

传统的流水线并行存在：
- 计算资源**利用率不足**（pipeline bubble）
- 整体**吞吐量低**

### 12.2 细粒度流水线并行

**优化策略**：
1. 将**mini-batch**（每个迭代处理的样本数）划分为多个**micro-batch**
2. 在micro-batch之间对**前向计算（forward）和后向计算（backward）**进行流水线化

这样做的效果：
- 填充流水线气泡（pipeline bubble）
- 提高计算资源利用率
- 增加整体吞吐量

### 12.3 大规模训练的并行配置

以下为不同模型规模下的并行配置参考（序列长度2048，词汇表大小51200）：

| 模型大小 | 注意力头数 | 隐藏维度 | 层数 | TP | PP | MP | DP | GPU数 | 批大小 | %峰值算力 |
|----------|-----------|---------|------|----|----|----|----|-------|--------|----------|
| 1.7B     | 24        | 2304    | 24   | 1  | 1  | 1  | 32 | 32    | 512    | 44%      |
| 3.6B     | 32        | 3072    | 30   | 2  | 1  | 2  | 32 | 64    | 512    | 42%      |
| 7.5B     | 32        | 4096    | 36   | 4  | 1  | 4  | 32 | 128   | 512    | 41%      |
| 18B      | 48        | 6144    | 40   | 8  | 1  | 8  | 32 | 256   | 1024   | 41%      |
| 39B      | 64        | 8192    | 48   | 8  | 2  | 16 | 32 | 512   | 1536   | 41%      |
| 76B      | 80        | 10240   | 60   | 8  | 4  | 32 | 32 | 1024  | 1792   | 43%      |
| 145B     | 96        | 12288   | 80   | 8  | 8  | 64 | 24 | 1536  | 2304   | 44%      |
| 291B     | 128       | 16384   | 90   | 8  | 18 | 144| 15 | 2160  | 2430   | 45%      |
| 530B     | 128       | 20480   | 105  | 8  | 35 | 280| 9  | 2520  | 2520   | 49%      |
| 1T       | 160       | 25600   | 128  | 8  | 64 | 512| 6  | 3072  | 3072   | 49%      |

**缩写说明**：
- **TP** = Tensor Parallel（张量并行）
- **PP** = Pipeline Parallel（流水线并行）
- **MP** = Model Parallel = TP x PP（模型并行）
- **DP** = Data Parallel（数据并行）

**影响因素**：所有这些并行配置（流水线深度、张量和数据并行的程度、全局批大小、流水线调度策略、微批大小）都会影响通信量、流水线气泡大小和内存占用。

---

## 13. 能效优化与数据移动成本

### 13.1 降低能耗的两个核心思路

1. **使用专用处理器**：为特定工作负载选择合适的处理器
2. **减少数据移动**：尽可能减少数据传输

### 13.2 数据访问的能耗代价

**经验法则（移动系统设计）**：始终致力于减少从内存传输的数据量。

| 操作 | 能量消耗 | 相对比值 |
|------|---------|---------|
| 整数运算 | ~1 pJ | 1x |
| 浮点运算 | ~20 pJ | 20x |
| 从近端SRAM读取64-bit（片上1mm距离） | ~26 pJ | 26x |
| 从LPDDR（低功耗移动DRAM）读取64-bit | ~1200 pJ | 1200x |

**实际影响**：
- 从内存读取 10 GB/sec = 约1.6瓦
- 移动GPU的整个功耗预算仅约1瓦
- iPhone 16电池容量约14瓦时（MacBook Pro约99瓦时）

**重要推论**：

> **重新计算值（recomputing values）可能比存储并重新加载它们更节能！**

这在优化能耗时是一个反直觉但关键的洞察。

### 13.3 数据移动成本的双重问题

**问题1：数据移动限制性能**
- 更多处理单元 => 更高总内存请求速率
- 需要更多内存带宽
- 结果：受带宽限制的执行（bandwidth-limited execution）

**问题2：数据移动高能耗**
| 操作 | 能量消耗（45nm CMOS） |
|------|---------------------|
| 32-bit浮点数学运算 | ~0.9 pJ |
| 本地SRAM（片上）数据访问 | ~5 pJ |
| 从LPDDR加载32-bit | ~640 pJ |

数据来源：Han, ICLR 2016

---

## 14. DRAM工作原理深入

### 14.1 内存系统架构

```
CPU Core --> 发出load/store指令
        --> LLC (末级缓存)
        --> 内存控制器 -> 向DRAM发送命令
        --> 64-bit内存总线
        --> DRAM
```

### 14.2 DRAM阵列结构

- 每bit由一个晶体管 + 一个电容器组成
- 每行2K bits
- 数据引脚：8 bits
- **Row Buffer**（行缓冲区）：2K bits，作为缓存放最近访问的行

### 14.3 DRAM操作步骤（加载一个字节）

1. **预充电（Precharge, ~10ns）**：准备好bit lines
2. **行激活（Row Activation, ~10ns）**：将目标行传输到行缓冲区
3. **列选择（Column Selection）**：选择目标列
4. **数据传输到总线（~10ns）**：将数据传回内存控制器

### 14.4 DRAM访问延迟不固定

**最佳情况**：从已激活的行读取（行命中）
- 只需要：列访问时间（CAS）

**最坏情况**：bit lines未准备好，需要从新行读取
- 需要：预充电（PRE）+ 行激活（RAS）+ 列访问（CAS）

**两个关键问题**：
1. **何时执行预充电？** 每次列访问后？还是仅在访问新行时？
   - 答案：预充电将行缓冲区内容写回DRAM阵列（因为读取是破坏性的），仅在需要访问新行时执行
2. **如何处理DRAM访问延迟？**

### 14.5 引脚利用率问题

由于DRAM访问延迟，数据引脚仅在很短的时间内被使用（红色部分），这是严重的资源浪费。

### 14.6 DRAM突发模式（Burst Mode）

**解决方案**：将延迟分摊到更大的传输上。

每个DRAM命令描述一个批量传输，位在连续的时钟周期中被放置在输出引脚上。这样显著提高了引脚利用率。

### 14.7 多Bank架构

DRAM芯片由多个Bank组成：
- 所有Bank共享相同的引脚（同一时间只能有一个传输）
- Bank允许**内存请求的流水线化**：
  - 在一个Bank上执行预充电/行激活/列地址传输的同时
  - 从另一个Bank传输数据
- 实现高数据引脚利用率

### 14.8 DIMM：将多个芯片组织为一个模块

**示例**：8个DRAM芯片组成一个64-bit内存总线的DIMM

- DIMM对内存控制器来说表现为一个更大容量、更宽接口的DRAM模块
- 更高总带宽，但最小传输粒度为64 bits

### 14.9 物理地址交错（Interleaving）

**错误方式**：连续的物理地址映射到同一个DRAM芯片的同一行
- 导致所有数据由同一芯片串行服务

**正确方式**：物理地址按**字节粒度跨DRAM芯片交错**
- 8个芯片同时并行传输前64 bits
- 后续64 bits通过Burst模式连续传输
- 大幅提高有效带宽

### 14.10 内存控制器：内存请求调度器

内存控制器接收来自LLC的load/store请求，面临冲突的调度目标：
- 最大化吞吐量
- 最小化延迟
- 最小化能耗

**常见调度策略：FR-FCFS**
（First-Ready, First-Come-First-Serve）
- 首先服务当前打开行的请求（最大化行局部性）
- 其他行的请求按FIFO顺序服务
- 控制器可能合并多个小请求为大的连续请求（利用DRAM突发模式）

每个Bank有独立的请求队列：
```
bank 0 请求队列
bank 1 请求队列
bank 2 请求队列
bank 3 请求队列
```

### 14.11 双通道内存系统

通过增加内存通道来增加吞吐量（实际上加宽总线）：
- 每个通道可以独立发出命令
- 不同通道可以读取不同的行/列
- 更简单的设置：使用单个控制器驱动相同的命令到多个通道

### 14.12 DDR4 内存示例

**DDR4 2400** 规格：
- 64-bit内存总线 x 1.2GHz x 2次传输/时钟 = **19.2 GB/s 每通道**
- 2个通道 = **38.4 GB/s**
- CAS延迟约13纳秒

*注：DDR = Double Data Rate（双倍数据速率），每个时钟周期传输2次数据*

**处理器**：Intel Core i7-7700K（Myth集群）

### 14.13 DRAM总结

DRAM访问延迟取决于多个底层因素：
- DRAM芯片状态：行命中/行缺失？是否需要预充电？
- 内存控制器中的请求缓冲/重排序

现代多核处理器大量复杂性已转移到内存控制器设计中：
- 负责调度数十到数百个待处理的内存请求
- 负责将物理地址映射到DRAM的几何结构（Bank、Row、Column）
- **内存控制器设计是活跃的计算机体系结构研究领域**

---

## 15. 总结：内存瓶颈与未来方向

### 15.1 现代架构挑战：提升内存性能

**核心策略**：通过将内存放置在更靠近处理器的位置来**减少数据必须移动的距离**
- 实现更短但更宽的接口
- 3D堆叠（HBM）是实现这一目标的关键技术

### 15.2 内存瓶颈的多层次解决方案

**应用程序员层面**：
- 调度计算以最大化局部性（最小化所需的数据移动）
- 重新计算而不是存储并重新加载（能耗优化）

**硬件架构层面**：
- 智能DRAM请求调度（FR-FCFS等策略）
- 将数据更靠近处理器（深层缓存层次结构、3D堆叠）
- 增加带宽（更宽的内存系统，如HBM的1024-bit接口）
- **内存内/近内存计算**（Processing in/near Memory，活跃研究方向）
- **硬件加速压缩**（减少数据传输量）

**通用设计原则**：
1. **将数据存储靠近处理器**（Locate data storage near processor）
2. **将计算移动到数据存储位置**（Move computation to data storage）
3. **数据压缩**（Data compression）：用额外计算换取更少的数据传输

### 15.3 专用AI硬件的编程总结

| 硬件平台 | 编程模型 | 核心挑战 |
|----------|---------|---------|
| **Google TPU** | 高级框架（JAX/TensorFlow） | 脉动阵列调度 |
| **NVIDIA H100** | CUDA + ThunderKittens DSL | 异步计算/内存机制的复杂性 |
| **SambaNova SN40L** | 数据流编程 + 元流水线 | 编译器优化和硬件映射 |

**关键原则**：
- **最小化同步开销**是获得高性能的关键
- 在AI数据中心的模拟中，**AI进展依赖于硬件改进**

### 15.4 关键数据：Tensor Core的重要性

自2016年到2024年，GPU中**Tensor Core占据的计算能力比例从89%增长到98%**。几乎所有TFLOPS都来自Tensor Core。

---

## 附录：关键概念速查

| 概念 | 英文 | 简要说明 |
|------|------|---------|
| 脉动阵列 | Systolic Array | Google TPU的核心计算单元，高效执行稠密矩阵乘法 |
| 高带宽内存 | HBM | 通过3D堆叠和1024-bit接口实现高带宽、低功耗的内存技术 |
| 硅通孔 | TSV | Through-Silicon-Via，实现DRAM芯片3D堆叠的关键技术 |
| 模式计算单元 | PCU | SambaNova RDU的计算组件，包含脉动阵列和SIMD单元 |
| 模式内存单元 | PMU | SambaNova RDU的内存组件，提供灵活地址生成 |
| 元流水线 | Metapipelining | 分层粗粒度流水线编程模型，将循环转换为流式流水线 |
| 核融合 | Kernel Fusion | 将多个操作合并为一个kernel以减少启动开销和提高数据局部性 |
| FlashAttention | - | 分块注意力算法，减少内存读写 |
| AllReduce | - | 集合通信原语，所有节点归约并广播结果 |
| ReduceScatter | - | AllReduce的第一步，每个节点获得归约结果的一部分 |
| AllGather | - | AllReduce的第二步，每个节点广播自己的部分结果 |
| All-to-All | - | 每个节点向所有其他节点发送不同数据，用于专家并行 |
| FR-FCFS | - | DRAM内存控制器调度策略，优先服务已打开行的请求 |
| 数据并行 | Data Parallelism | 沿batch维度分片，每设备有完整模型副本 |
| 张量并行 | Tensor Parallelism | 沿hidden_dim分片权重张量 |
| 流水线并行 | Pipeline Parallelism | 沿层维度分片，不同设备负责不同层 |
| 专家并行 | Expert Parallelism | MoE模型中沿专家维度分片 |
