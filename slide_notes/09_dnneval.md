# Lecture 9: 深度神经网络推理优化

> Stanford CS149, Fall 2025
> Lecture 9: Efficiently Evaluating DNNs

---

## 一、课程回顾：并行计算性能基础

### 1.1 流水线化：重叠数据传输与计算

回顾之前课程的核心概念——通过流水线（pipelining）将数据移动与计算重叠执行：

```
time →
=  Arithmetic operations（算术运算）
=  Load data（加载数据）
=  Store result（存储结果）
```

**关键问题：**
- **问题1**：该程序是计算受限（compute bound）还是带宽受限（BW bound）？
- **问题2**：重叠数据移动与计算需要的片上存储代价是什么？（提示：通常称为"双缓冲"——double buffering）

### 1.2 Roofline 性能模型

Roofline 曲线是分析程序性能瓶颈的核心工具：

- **横轴**：算术强度（Arithmetic Intensity, Ops/BW）——每字节数据传输对应的浮点运算次数
- **纵轴**：吞吐量（Throughput, Ops/sec）

**Roofline 曲线的两个区域：**
- **带宽受限区（BW-bound regime）**：算术强度较低时，程序性能受限于内存带宽
- **计算受限区（Compute-bound regime）**：算术强度足够高时，程序性能受限于处理器的峰值计算能力

**硬件变化对性能瓶颈的影响：**
- 相同内存系统但更高峰值计算能力的机器 → 程序更容易变成**带宽受限**
- 更高带宽内存系统 + 更高峰值计算能力的机器 → 整体性能提升

### 1.3 循环融合（Loop Fusion）变换

循环融合是提高算术强度的关键优化技术——将多个循环合并为一个。

**程序1（未融合）：**
```c
void add(int n, float* A, float* B, float* C) {
    for (int i=0; i<n; i++)
       C[i] = A[i] + B[i];      // 算术强度 = 1/3
}
void mul(int n, float* A, float* B, float* C) {
    for (int i=0; i<n; i++)
       C[i] = A[i] * B[i];      // 算术强度 = 1/3
}

float* A, *B, *C, *D, *E, *tmp1, *tmp2;
// 计算 E = D + ((A + B) * C)
add(n, A, B, tmp1);
mul(n, tmp1, C, tmp2);
add(n, tmp2, D, E);
// 整体算术强度 = 1/3（每次数学操作对应两次加载、一次存储）
```

**程序2（融合后）：**
```c
void fused(int n, float* A, float* B, float* C, float* D, float* E) {
    for (int i=0; i<n; i++)
       E[i] = D[i] + (A[i] + B[i]) * C[i];
}
// 计算 E = D + (A + B) * C
fused(n, A, B, C, D, E);
// 算术强度 = 3/5（三次数学操作对应四次加载、一次存储）
```

**总结：** 循环融合通过减少中间结果的写回和重新读取，显著提高了算术强度。

### 1.4 核心回顾总结

- 当通信与计算重叠时，机器的能力（运算吞吐量和通信带宽）以及程序的算术强度共同决定程序是带宽受限还是计算受限
- 重叠通信与计算需要额外的片上存储空间（双缓冲），因为需要同时维护正在处理的数据和正在传输的数据
- 提高算术处理能力（更快的硬件）→ 程序更容易变成带宽受限
- 提高程序的算术强度（程序变换）→ 程序更容易变成计算受限

**关键认识：** 以上知识几乎涵盖了现代AI性能优化的所有软件方面的核心内容。硬件方面则归结为：(1) 数据移动消耗能量，(2)片上存储资源与计算资源是互斥的，应尽可能减少缓冲区占用。

---

## 二、深度神经网络基础

### 2.1 什么是深度神经网络（DNN）？

DNN 本质上是一个**电路（circuit）**——由基本计算单元组成的大规模计算图。

**基本单元：神经元（Neuron）**

一个有 n 个输入的神经元由 n+1 个参数描述（n 个权重 + 1 个偏置）：

```
output = f(Σ(xi * wi) + b)
```

其中 f 是激活函数，常见的包括：
- **ReLU（Rectified Linear Unit）**：`f(x) = max(0, x)`
- **Sigmoid**：`f(x) = 1 / (1 + e^(-x))` —— 用于二分类，输出可解释为概率

**机器学习解释**：神经元 = 二分类器，输出可解释为某一类的概率
**计算解释**：神经元 = 电路中的基本计算单元

### 2.2 DNN的拓扑结构

**全连接层（Fully Connected Layer）：**
- 每个输出单元接收所有输入节点的输入
- 可表示为矩阵-向量乘积

**稀疏连接/局部连接层（Sparsely/Locally Connected Layer）：**
- 每个单元只接收有限范围内的输入节点
- 卷积层是典型的局部连接结构

### 2.3 全连接层 = 矩阵-向量乘积

全连接层的计算本质上就是矩阵-向量乘法，然后加上非线性激活函数：

```
输出 = f(W * x + b)
```

其中 W 是权重矩阵，x 是输入向量，b 是偏置向量，f 是逐元素的激活函数（如 ReLU）。

---

## 三、卷积神经网络（CNN）基础

### 3.1 2D卷积的实现

以下C代码展示了一个3x3卷积（图像模糊）的基本实现：

```c
int WIDTH = 1024;
int HEIGHT = 1024;
float input[(WIDTH+2) * (HEIGHT+2)];
float output[WIDTH * HEIGHT];
float weights[] = {1.f/9, 1.f/9, 1.f/9,
                   1.f/9, 1.f/9, 1.f/9,
                   1.f/9, 1.f/9, 1.f/9};

for (int j=0; j<HEIGHT; j++) {
  for (int i=0; i<WIDTH; i++) {
    float tmp = 0.f;
    for (int jj=0; jj<3; jj++)
      for (int ii=0; ii<3; ii++)
        tmp += input[(j+jj)*(WIDTH+2) + (i+ii)] * weights[jj*3 + ii];
    output[j*WIDTH + i] = tmp;
  }
}
```

这段代码对1024x1024图像执行3x3的均值滤波（每个权重为1/9）。

### 3.2 卷积层的特点

卷积层有两个重要特性：

1. **局部连接**：每个输出像素只与输入的一个局部区域（如3x3）连接
2. **参数共享**：层中所有单元共享同一组权重参数（权重 + 偏置）

这意味着卷积层可以看作将一组可学习的"滤波器"（filter）应用于输入图像。

### 3.3 梯度检测滤波器

卷积滤波器可以被理解为一种"模式检测器"：
- 特定的滤波器权重可以检测**水平梯度**（水平边缘）
- 另一些权重检测**垂直梯度**（垂直边缘）
- 输出图像中每个像素的值表示"滤波器对该位置的响应强度"

### 3.4 同时应用多个滤波器

输入为多通道图像（如RGB: W x H x 3），可以应用多组滤波器：

```
输入: 图像（单通道） W x H
权重: 3x3 x num_filters
输出: 滤波器响应图 W x H x num_filters
```

每个滤波器由独特的3x3权重参数描述，对不同的图像特征（边缘、纹理、颜色等）做出响应。

### 3.5 完整的CNN层次结构

一个典型的CNN由以下层次组成：

1. **卷积层（Conv）**：应用多个滤波器，输出 W x H x num_filters
2. **ReLU激活层**：逐元素非线性变换 `max(0, x)`
3. **池化层（Pool）**：下采样，如 2x2 最大池化将分辨率减半 (W/2 x H/2)

池化层实现了数据降维，减少了后续层的计算量。

---

## 四、卷积层的高效实现

### 4.1 直接卷积实现（批处理版本）

```c
float input[IMAGE_BATCH_SIZE][INPUT_HEIGHT][INPUT_WIDTH][INPUT_DEPTH];
float output[IMAGE_BATCH_SIZE][INPUT_HEIGHT][INPUT_WIDTH][LAYER_NUM_FILTERS];
float layer_weights[LAYER_NUM_FILTERS][LAYER_CONVY][LAYER_CONVX][INPUT_DEPTH];
float layer_biases[LAYER_NUM_FILTERS];

// 假设卷积步长（stride）= 1
for (int img=0; img<IMAGE_BATCH_SIZE; img++)
   for (int j=0; j<INPUT_HEIGHT; j++)
      for (int i=0; i<INPUT_WIDTH; i++)
         for (int f=0; f<LAYER_NUM_FILTERS; f++) {
            float tmp = layer_biases[LAYER_NUM_FILTERS];
            for (int kk=0; kk<INPUT_DEPTH; kk++)         // 多通道求和
               for (int jj=0; jj<LAYER_FILTER_Y; jj++)   // 空间卷积(Y)
                  for (int ii=0; ii<LAYER_FILTER_X; ii++) // 空间卷积(X)
                     tmp += layer_weights[f][jj][ii][kk] * input[img][j+jj][i+ii][kk];
            output[img][j][i][f] = tmp;
         }
```

**七层循环嵌套**，具有显著的数据重用机会：
- 卷积过程中对**滤波器权重的重用**
- 跨不同滤波器时对**输入值**的重用

### 4.2 卷积转为矩阵乘法（Explicit GEMM）

将3x3卷积重新组织为矩阵-向量乘积：

- 将每个3x3的输入区域展开为长度为9的行向量
- 构建一个 **(WxH) x 9** 的矩阵（需要0填充处理边界）
- 权重向量为 **[w0, w1, ..., w8]**
- 输出 = 输入矩阵 * 权重向量

**注意：** 对于N个元素的滤波器，这种方法的存储开销为 O(N)，且需要显式构建输入数据矩阵。

### 4.3 多通道多滤波器卷积的矩阵形式

对于多输入通道的卷积，需要扩展矩阵构造：
- 每个输入通道的3x3区域展平为9个元素
- 对于C个输入通道：每行有 **9 x C** 个元素
- 权重矩阵的大小为 **(9 x C) x num_filters**
- 等价于对 **(W x H x C)** 的输入数据进行 **(3 x 3 x C)** 卷积

### 4.4 卷积到显式GEMM的映射

符号说明：
- **R x S**：滤波器的空间支持域
- **C**：输入通道数
- **K**：滤波器数量（输出通道数）
- **N**：批处理大小

卷积层可以通过将输入激活张量重组为矩阵，将滤波器权重重组为另一个矩阵，将卷积计算转化为高效的GEMM操作。

---

## 五、矩阵乘法（GEMM）在现代AI中的核心地位

### 5.1 GEMM的应用场景

密集矩阵-矩阵乘法是现代AI的**核心计算原语**，广泛应用于：

- **全连接层（Fully-connected layers）**
- **卷积层（Convolutional layers）**
- **Transformer架构中的Attention（注意力）模块**

Transformer 架构的核心计算流程：
```
输入序列 tokens → 矩阵乘法（Attention计算）→ 输出序列 tokens
```

具体而言，Attention 计算涉及 `Q * K^T`、`softmax`、`P * V` 等矩阵操作，全部以矩阵乘法为基础。

### 5.2 Attention模块的数学定义

设：
- **Q** 为 N x d 矩阵（Query）
- **K** 为 N x d 矩阵（Key）
- **V** 为 N x d 矩阵（Value）
- N 为输入序列长度
- d 为特征嵌入维度

**标准 Attention 计算：**

```
S = Q * K^T    （N x N 矩阵 —— 需要 N² 空间！）
P = softmax(S) （N x N 矩阵）
O = P * V      （N x d 矩阵）
```

**内存挑战：** 当序列长度 N 很大（如数千）时，S 和 P 矩阵需要 N² 的存储空间，这是一个严重的扩展性问题。

### 5.3 朴素矩阵乘法的实现问题

```c
float A[M][K];
float B[K][N];
float C[M][N];

// 计算 C += A * B
#pragma omp parallel for
for (int j=0; j<M; j++)
  for (int i=0; i<N; i++)
     for (int k=0; k<K; k++)
         C[j][i] += A[j][k] * B[k][i];
```

**问题：** 该实现算术强度很低，因为它没有利用对A和B访问的时间局部性（temporal locality）。每次迭代都需要重新从内存中加载数据。

---

## 六、通过分块（Blocking）提高算术强度

### 6.1 基本分块矩阵乘法

```c
float A[M][K];
float B[K][N];
float C[M][N];

// 计算 C += A * B
#pragma omp parallel for
for (int jblock=0; jblock<M; jblock+=BLOCKSIZE_J)
  for (int iblock=0; iblock<N; iblock+=BLOCKSIZE_I)
     for (int kblock=0; kblock<K; kblock+=BLOCKSIZE_K)
        for (int j=0; j<BLOCKSIZE_J; j++)
           for (int i=0; i<BLOCKSIZE_I; i++)
              for (int k=0; k<BLOCKSIZE_K; k++)
                 C[jblock+j][iblock+i] += A[jblock+j][kblock+k] * B[kblock+k][iblock+i];
```

**核心思想：** 在A和B的所需分块仍然驻留在缓存中时，先计算C分块的部分结果。

**自检问题：** 分块大小是否越大越好？——需要权衡：分块太大可能超出缓存容量，分块太小则无法充分利用数据重用。

### 6.2 层次化分块矩阵乘法

```c
float A[M][K];
float B[K][N];
float C[M][N];

// 计算 C += A * B
#pragma omp parallel for
for (int jblock2=0; jblock2<M; jblock2+=L2_BLOCKSIZE_J)
  for (int iblock2=0; iblock2<N; iblock2+=L2_BLOCKSIZE_I)
     for (int kblock2=0; kblock2<K; kblock2+=L2_BLOCKSIZE_K)
        for (int jblock1=0; jblock1<L2_BLOCKSIZE_J; jblock1+=L1_BLOCKSIZE_J)
           for (int iblock1=0; iblock1<L2_BLOCKSIZE_I; iblock1+=L1_BLOCKSIZE_I)
              for (int kblock1=0; kblock1<L2_BLOCKSIZE_K; kblock1+=L1_BLOCKSIZE_K)
                  for (int j=0; j<L1_BLOCKSIZE_J; j++)
                     for (int i=0; i<L1_BLOCKSIZE_I; i++)
                        for (int k=0; k<L1_BLOCKSIZE_K; k++)
                           // ... 最内层还可以有寄存器级分块
```

**设计原理：** 利用多级存储层次（L1缓存、L2缓存、L3缓存、寄存器），为每一级存储层次匹配合适的分块大小，在各层次上同时提高算术强度。

### 6.3 向量化的分块矩阵乘法（方案1 —— 向量化i循环）

```c
for (int j=0; j<BLOCKSIZE_J; j++) {
   for (int i=0; i<BLOCKSIZE_I; i+=SIMD_WIDTH) {
      simd_vec C_accum = vec_load(&C[jblock+j][iblock+i]);
      for (int k=0; k<BLOCKSIZE_K; k++) {
         simd_vec A_val = splat(&A[jblock+j][kblock+k]);  // 广播单个元素
         simd_muladd(A_val, vec_load(&B[kblock+k][iblock+i]), C_accum);
      }
      vec_store(&C[jblock+j][iblock+i], C_accum);
   }
}
```

**优点：** 提高了对B的空间局部性
**缺点：** 工作集增加了SIMD_WIDTH倍，仍然以大跨度访问B

### 6.4 向量化的分块矩阵乘法（方案2 —— 使用预转置）

```c
for (int j=0; j<BLOCKSIZE_J; j++)
   for (int i=0; i<BLOCKSIZE_I; i++) {
      float C_scalar = C[jblock+j][iblock+i];
      // C_scalar += dot(A的一行, B的一行)
      for (int k=0; k<BLOCKSIZE_K; k+=SIMD_WIDTH) {
        C_scalar += simd_dot(vec_load(&A[jblock+j][kblock+k]),
                              vec_load(&Btrans[iblock+i][kblock+k]));
      }
      C[jblock+j][iblock+i] = C_scalar;
   }
```

**适用场景：** 当i维度较小时使用。预先将B的分块转置存储到临时缓冲区中，然后对最内层循环进行向量化。

### 6.5 向量化的分块矩阵乘法（方案3 —— 最优化版本）

```c
// 假设A和C的分块已预转置为Atrans和Ctrans
for (int j=0; j<BLOCKSIZE_J; j+=SIMD_WIDTH) {
   for (int i=0; i<BLOCKSIZE_I; i+=SIMD_WIDTH) {
      simd_vec C_accum[SIMD_WIDTH];
      for (int k=0; k<SIMD_WIDTH; k++)
         C_accum[k] = vec_load(&Ctrans[iblock+i+k][jblock+j]);
      for (int k=0; k<BLOCKSIZE_K; k++) {
        simd_vec bvec = vec_load(&B[kblock+k][iblock+i]);
        for (int kk=0; kk<SIMD_WIDTH; kk++)  // 内层循环无依赖
            simd_muladd(vec_load(&Atrans[kblock+k][jblock+j]),
                        splat(bvec[kk]), C_accum[kk]);
      }
      for (int k=0; k<SIMD_WIDTH; k++)
        vec_store(&Ctrans[iblock+i+k][jblock+j], C_accum[k]);
   }
}
```

这是在SIMD向量化、分块和数据布局优化方面最激进的实现版本。

---

## 七、Implicit GEMM 与 GPU 实现

### 7.1 显式 GEMM 的代价

使用现成的GEMM库（如cuBLAS）处理卷积层需要**显式物化输入矩阵**：
- 将输入激活张量重组为"卷积矩阵"，数据量膨胀 **R x S 倍**（其中R x S是滤波器的空间尺寸）
- 显著增加了DRAM流量
- 需要大量的额外存储空间

### 7.2 Implicit GEMM —— 不显式构建完整矩阵

**朴素做法（Naive Implicit GEMM）：** 直接在循环中索引输入权重和激活张量，但不做分块优化。

**更好的做法（优化的 Implicit GEMM）：**
- 每次只在GPU片上共享内存（shared memory）中物化卷积矩阵的**一个子块**
- 不需要额外的片外存储
- 不增加DRAM流量
- 使用经过高度调优的**共享内存GEMM**例程来执行子块GEMM

符号说明：P x Q 为输出空间尺寸，R x S 为滤波器尺寸，C 为输入通道数，K 为输出通道数，N 为批大小。

### 7.3 不同层需要不同的调度策略

以MobileNet为例，网络中不同层具有极不相同的数据规模：

| 层类型 | 权重形状 | 输入尺寸 |
|--------|----------|----------|
| Conv | 3x3x3x32 | 224x224x3 |
| Conv dw | 3x3x32 dw | 112x112x32 |
| Conv (1x1) | 1x1x32x64 | 112x112x32 |
| Conv dw | 3x3x128 dw | 56x56x128 |
| FC | 1024x1000 | 1x1x1024 |

**关键问题：** 注意网络中权重和激活张量的不同尺寸，需要针对不同维度选择合适的调度策略。这对库的实现者来说是个重大挑战。

---

## 八、NVIDIA GPU 与 CUDA 编程生态

### 8.1 NVIDIA V100 GPU 架构回顾

- 80个流式多处理器（SM）
- L2缓存：6 MB
- GPU内存（HBM）：16 GB
- 内存带宽：900 GB/sec（4096位内存接口）
- 大量张量核心（Tensor Cores）

**核心挑战：** 需要"足够多的并行工作"才能填满整个机器的计算资源。

### 8.2 性能与工作量

- **小工作负载**（N=1, P=Q=64）：64 x 64 x 128 x 1 = 524K个输出 = 约2 MB输出数据（float32）
- **大工作负载**（N=32, P=Q=256）：256 x 256 x 128 x 32 = 256M个输出 = 约1 GB输出数据（float32）

更大的工作负载通常能获得更高的GPU利用率，因为GPU需要大量并行任务才能充分利用其计算能力。

### 8.3 直接实现 vs 库实现

对于大多数DNN评估需求，可以直接使用七重循环嵌套直接实现卷积：

```c
for (int img=0; img<IMAGE_BATCH_SIZE; img++)             // 批处理
   for (int j=0; j<INPUT_HEIGHT; j++)                      // 输出行
      for (int i=0; i<INPUT_WIDTH; i++)                   // 输出列
         for (int f=0; f<LAYER_NUM_FILTERS; f++) {       // 输出通道
            float tmp = layer_biases[LAYER_NUM_FILTERS];
            for (int kk=0; kk<INPUT_DEPTH; kk++)         // 输入通道求和
               for (int jj=0; jj<LAYER_FILTER_Y; jj++)   // 空间卷积Y
                  for (int ii=0; ii<LAYER_FILTER_X; ii++) // 空间卷积X
                     tmp += layer_weights[f][jj][ii][kk] * input[img][j+jj][i+ii][kk];
            output[img][j][i][f] = tmp;
         }
```

然而，对于追求极致性能的场景，应该使用经过调优的库。

### 8.4 关键CUDA库与框架

#### NVIDIA CUTLASS
- 基本原语/构建块，用于实现自定义高性能DNN层
- 适用于cuDNN未充分调优的非标准尺寸
- 核心功能：快速共享内存GEMM、快速Warp级GEMM、用于快速分块加载/张量索引的迭代器、张量归约等

#### Triton
- 语言级别支持张量的加载/存储操作
- 将数据"分块"加载到GPU共享内存
- 对分块执行数据并行操作
- 支持两层分块的矩阵乘法（完整参考实现）

#### Thunderkittens
- 基于分块（tile-based）编程原语的CUDA库
- 面向高级开发者，提高编写分块代码的生产力
- 支持异步分块加载/存储
- 支持高级内存布局（分块tile、交叉元素等）

#### cuDNN（NVIDIA深度神经网络库）
- 提供高度优化的DNN层实现
- 单个卷积操作可能有**多种实现算法**可供选择（根据问题参数自动选择最优算法）
- cuDNN 后端支持自动融合多个操作

#### AWS NKI
- 针对AWS Trainium/Inferentia芯片的底层库
- 提供关键DNN层的高性能实现

---

## 九、融合优化（Fusion）进阶

### 9.1 操作间内存流量的代价

考虑以下典型操作序列：

```
Conv → Scale/Bias → Max Pool
```

**朴素实现的问题：**
- 将1 GB的卷积输出写入内存
- 重新读取所有数据进行简单的Scale操作
- 再次读取所有数据进行Pool操作
- 巨大的带宽浪费！

**更好的解决方案——融合操作：**
- Scale+Bias 操作可以在每个卷积输出元素计算完毕后立即逐元素执行
- Max Pool 的输出可以在每计算完一个2x2区域后立即产生

融合后的数据流：
```
Conv + Scale/Bias + Max Pool:
N x H x W x C → N x H/2 x W/2 x K  （一步完成，无需中间存储）
```

### 9.2 Scale/Bias与卷积的融合实现

```c
for (int img=0; img<IMAGE_BATCH_SIZE; img++)
   for (int j=0; j<INPUT_HEIGHT; j++)
      for (int i=0; i<INPUT_WIDTH; i++)
         for (int f=0; f<LAYER_NUM_FILTERS; f++) {
            float tmp = 0.0f;
            for (int kk=0; kk<INPUT_DEPTH; kk++)
               for (int jj=0; jj<LAYER_FILTER_Y; jj++)
                  for (int ii=0; ii<LAYER_FILTER_X; ii++)
                     tmp += layer_weights[f][jj][ii][kk] * input[img][j+jj][i+ii][kk];
            output[img][j][i][f] = tmp * scale + bias;  // 融合！
         }
```

**课堂练习：** 如何进一步"融合"一个Max Pool操作（对2x2输出块取最大值）？提示：如何对标记为黄色的循环进行分块？

### 9.3 Softmax 优化

**朴素 Softmax 实现的问题：**
- 对于矩阵的行进行softmax计算
- `softmax(x) = f(x) / l(x)`，其中 `m(x) = max(xi)`，`f(x) = [e^(xi-m(x)), ...]`，`l(x) = sum(f(x))`
- 需要多次读/写整个 M x N 矩阵，算术强度很低
- 读取 5MN + 2M 个元素，写入 3MN + 2M 个元素

**融合后的实现：**
- 逐行处理：加载一行 → 完整计算该行softmax → 存储结果
- 前提是单行的工作集能够放入片上存储
- 读取 MN 个元素，写入 MN 个元素

### 9.4 Flash Attention —— 避免物化 N² 矩阵

**核心思想：** softmax 可以**分块计算**。

将向量 x 拆分为两个块 x(1) 和 x(2)：

```
m(x) = max(m(x(1)), m(x(2)))
f(x) = [e^(m(x1)-m(x)) * f(x(1)), e^(m(x2)-m(x)) * f(x(2))]
l(x) = e^(m(x(1))-m(x)) * l(x(1)) + e^(m(x(2))-m(x)) * l(x(2))
```

**融合 Attention 算法：**

```
Q: N x d
K^T: d x N
V: N x d
O = PV: N x d  （输出）

外层循环 j：（遍历 K、V 的块）
  内层循环 i：（遍历 Q 的块）
    加载块 Qi, K^T_j, V_j, O_i
    计算 S_ij = Qi * K^T_j
    计算 M_ij = m(S_ij), P_ij = f(S_ij), l_ij = l(S_ij)
    计算 P_ij * V_j 并累积到 O_i（使用前一页的数学公式进行适当的缩放）
```

**融合 Attention 的优势：**
- **内存占用：** 从不物化 N² 大小的矩阵
- **内存带宽：** 高算术强度
  - 读取3个块（来自Q, K, V）
  - 执行两次矩阵乘法和若干行求和
  - 累积到O块（保持在缓存中）
- **代价：** 相比原始版本有额外计算——每次i循环迭代时需要对O的先前值进行重新缩放

**Flash-Attention** 使用了 Thunderkittens 库在高性能GPU上实现了这一算法。

---

## 十、现代DNN框架中的融合

### 10.1 传统方法：库编写者硬编码融合操作

早期的深度学习框架（如 TensorFlow）中，库编写者手动编写少数几种"融合"操作的实现。这种方式不够灵活，无法覆盖所有可能的操作组合。

### 10.2 灵活融合：cuDNN 后端

编译器自动生成新的实现，将多个操作"融合"为单个执行节点，消除了运行时开销和通过内存传递中间结果的需要。

### 10.3 基于编译器的自动调度

多个基于编译器的工作致力于自动调度关键的DNN操作：
- **torch.compile**：PyTorch的JIT编译器，自动融合和优化计算图
- 研究重点：自动搜索最佳的循环分块、融合和调度策略

---

## 十一、低精度量化与模型压缩

### 11.1 使用低精度数值

降低DNN权重和中间激活的数值精度是提升效率的重要技术：

- **16位（FP16/BF16）** 和 **8位（INT8）** 精度已广泛使用
- 目前正发展到 **4位精度**
- 极端情况：**1位（二值化网络）**

**效果：** 低精度直接降低存储需求和内存带宽压力，在专门的硬件（如Tensor Core）上还能获得更高的计算吞吐量。

### 11.2 DNN优化技术全景

| 类别 | 关键技术 | 说明 |
|------|----------|------|
| **更好的算法** | 模型拓扑设计 | 调整网络深度、滤波器宽度、每层滤波器数量、卷积步长等参数 |
| | 自动拓扑搜索（NAS） | 自动搜索高效的网络拓扑结构 |
| **软件优化** | 循环分块/平铺（Loop blocking/tiling） | 利用缓存局部性 |
| | 融合（Fusion） | 减少中间数据的内存流量 |
| | 调度策略 | 通常由人工手动优化，但也有大量自动调度研究 |
| **近似方法** | 模型压缩 | 降低比特精度、剪枝、知识蒸馏等 |

---

## 十二、GPU 作为 DNN 计算平台的评估

### 12.1 GPU 的优势

- DNN的**高算术强度**矩阵计算能充分利用GPU丰富的浮点运算能力
- 存在高度优化的GPU内核库（cuDNN, CUTLASS等）
- 例如 NVIDIA A100 等现代GPU具有专门的张量核心，提供极高的矩阵运算吞吐量

### 12.2 GPU 的潜在劣势

需要思考的核心问题：**真的需要通用处理器（general purpose processor）吗？**

- GPU仍然是一种通用计算平台，包含了DNN计算中不必要的灵活性
- 通用处理器的开销（指令解码、控制流、缓存层次等）在专用的DNN加速场景中可能成为浪费
- 数据移动消耗大量能量

---

## 十三、下一讲预告：专用硬件加速

下一讲将讨论**通过专用硬件加速最大化DNN推理/训练的能效**，涵盖的硬件平台包括：

- Google TPUv3（张量处理单元）
- 华为麒麟NPU（神经网络处理单元）
- Apple Neural Engine（苹果神经引擎）
- GraphCore IPU（智能处理单元）
- Ampere GPU with Tensor Cores（带张量核心的安培GPU）
- Intel Deep Learning Inference Accelerator（英特尔深度学习推理加速器）
- Cerebras Wafer Scale Engine（晶圆级引擎）
- SambaNova Cardinal SN10

---

## 十四、关键要点总结

1. **算术强度**是决定DNN计算性能瓶颈的核心指标——它是浮点运算数与数据传输量的比值
2. **Roofline模型**是分析和优化DNN性能的基本工具
3. **循环分块（Blocking/Tiling）**是利用多级缓存层次提高GEMM算术强度的关键技术
4. **卷积到GEMM的映射**是现代DNN加速的基石——使卷积能够利用高度优化后的矩阵乘法实现
5. **Implicit GEMM**避免显式构建完整卷积矩阵，减少存储开销和内存流量
6. **操作融合（Fusion）**是减少内存带宽压力的关键优化——将多个操作合并为一次执行
7. **Flash Attention**是对Transformer Attention的融合优化，完全避免了O(N²)中间矩阵的物化
8. **低精度量化**是模型压缩的重要手段——从32位浮点到16位、8位、4位甚至1位
9. **不同层需要不同的优化策略**——网络中各层的数据维度差异很大，需要定制化的调度
10. **专业硬件加速器**比通用GPU更适合DNN推理，因为它们消除了通用处理的不必要开销
