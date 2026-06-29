# Lecture 7: GPU架构与CUDA编程

**课程**: Stanford CS149, Fall 2025 — Parallel Computing

---

## 一、本讲概要

本讲涵盖三个主要方面：

1. **GPU历史**：图形处理器如何从最初为3D游戏加速而设计，演变为适用于深度学习、计算机视觉和科学计算等广泛应用的高性能并行计算引擎
2. **CUDA编程**：使用CUDA语言在GPU上进行编程
3. **GPU架构深入**：详细分析现代GPU的硬件架构

---

## 二、GPU基础架构回顾（源自Lecture 2）

### 基本结构

- **内存**：DDR5 DRAM（数GB容量），高端GPU带宽约150-300 GB/s
- **GPU芯片**：多核芯片
  - 单个核心内进行SIMD执行（多个执行单元执行相同指令）
  - 单个核心上多线程执行（一个核心同时执行多个线程）

### 为什么GPU有这么多高吞吐量核心？

GPU的设计初心是高效执行**着色器程序（shader programs）**——对大量数据（顶点流、片元流、像素流）并行执行相同的计算。这正是**数据并行（data-parallelism）**的核心思想。每年GPU变得更快，因为更多的晶体管意味着更多的并行性。

---

## 三、GPU历史：从图形到通用计算

### 3.1 3D渲染基础

**渲染任务的定义**：计算3D网格中的每个三角形如何影响图像中每个像素的外观。

**图形管线输入**：
- 3D表面几何（如三角形网格）
- 表面材质
- 灯光
- 相机参数

**图形管线输出**：场景的最终图像

### 3.2 实时图形的基本元素（Primitives）

- **顶点（Vertices）**：空间中的点
- **图元（Primitives）**：三角形、点、线
- 表面通过3D三角形网格来表示

### 3.3 图形工作负载概述

1. 给定一个三角形，根据虚拟相机的位置确定它在屏幕上的位置
2. 对于三角形覆盖的所有输出图像像素，计算该像素处表面的颜色

### 3.4 着色器程序（Shader Program）示例

```glsl
uniform sampler2D myTexture;       // 只读全局变量（纹理贴图）
uniform float3 lightDir;           // 只读全局变量（光源方向）
varying vec3 norm;                 // 逐像素变化的输入（法线）
varying vec2 uv;                   // 逐像素变化的输入（纹理坐标）

void myShader()
{
  vec3 kd = texture2D(myTexture, uv);        // 采样纹理
  kd *= clamp(dot(lightDir, norm), 0.0, 1.0); // 简单光照计算
  return vec4(kd, 1.0);                       // 输出RGBA颜色
}
```

**关键要点**：
- 着色器是一个**纯函数（pure function）**，在输入流上被调用
- 对每个被三角形覆盖的像素（片元）运行一次
- 这是一个天然的**数据并行**计算模式

---

## 四、GPGPU的诞生

### 4.1 早期探索（约2001-2003）

研究人员意识到：GPU是非常快速的处理器，能够在大规模数据集合上并行执行相同的计算（着色器程序）。这与90年代超级计算机中的数据并行思想高度一致。

**早期技巧（Hack）**：将输出图像尺寸设置为数组大小（如512x512），渲染两个恰好覆盖屏幕的三角形，使得每个像素对应一个数组元素的计算。这样就可以将GPU当作数据并行编程系统来使用。

**代表性早期工作**：
- Coupled Map Lattice Simulation [Harris 02]
- Ray Tracing on Programmable Graphics Hardware [Purcell 02]
- Sparse Matrix Solvers [Bolz 03]

GPGPU = "General Purpose" computation on GPUs（GPU通用计算）

### 4.2 Brook流编程语言（2004）

斯坦福图形实验室的研究项目，将GPU硬件抽象为数据并行处理器。

```brook
kernel void scale(float amount, float a<>, out float b<>)
{
   b = amount * a;
}

float scale_amount;
float input_stream<1000>;   // 流声明
float output_stream<1000>;  // 流声明

// 将kernel函数映射到流上
scale(scale_amount, input_stream, output_stream);
```

Brook编译器将通用流程序翻译为图形命令（如drawTriangles）和可在当时GPU上运行的图形着色器程序。

### 4.3 CPU vs GPU 代码执行方式对比

**CPU上运行程序**：
- OS将程序文本加载到内存
- OS选择CPU执行上下文
- OS中断处理器，准备执行上下文（设置寄存器、程序计数器等）
- 处理器开始从执行上下文中执行指令

**2007年前的GPU**：
- 应用程序通过图形驱动提供GPU着色器程序二进制
- 应用程序设置图形管线参数（如输出图像大小）
- 应用程序提供顶点缓冲区
- 应用程序发送"draw"命令：`drawPrimitives(vertex_buffer)`
- 这是唯一的GPU硬件接口——GPU只能执行图形管线计算

---

## 五、NVIDIA Tesla架构与CUDA（2007）

### 5.1 Tesla架构的革命

Tesla架构提供了**第一个非图形的"计算模式"接口**：

- 应用程序可以在GPU内存中分配缓冲区并拷贝数据
- 应用程序通过图形驱动提供单个kernel程序二进制
- 应用程序以SPMD方式告诉GPU运行kernel：`launch(myKernel, N)`（运行N个kernel实例）

有趣的是，这比图形操作`drawPrimitives()`简单得多。

### 5.2 CUDA编程语言

- 2007年随NVIDIA Tesla架构引入
- 类似C语言，用于表达在GPU上运行的程序
- **设计目标**：保持低抽象距离——CUDA的抽象与现代GPU的能力/性能特征紧密匹配
- 相对底层：抽象层非常接近硬件实际

---

## 六、CUDA编程模型

### 6.1 线程层次结构

CUDA程序由**层次化的并发线程**组成。线程ID最多可以是三维的（2D示例如下），多维线程ID便于处理天然的N维问题。

**两级层次结构**：
- **Grid（网格）**：由多个thread block组成
- **Thread Block（线程块）**：由多个CUDA thread组成

#### 矩阵加法示例

```cuda
const int Nx = 12;
const int Ny = 6;

dim3 threadsPerBlock(4, 3);
dim3 numBlocks(Nx/threadsPerBlock.x, Ny/threadsPerBlock.y);

// 假设A, B, C是已分配的Nx x Ny浮点数组
// 此调用启动72个CUDA线程：6个线程块，每块12个线程
matrixAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
```

**每个线程**根据其在block中的位置（`threadIdx`）和其block在grid中的位置（`blockIdx`）计算其全局grid线程ID：

```cuda
// kernel定义（在GPU上运行）
__global__ void matrixAdd(float A[Ny][Nx],
                          float B[Ny][Nx],
                          float C[Ny][Nx])
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y * blockDim.y + threadIdx.y;
   C[j][i] = A[j][i] + B[j][i];
}
```

**关键语法**：
- `__global__`：表示CUDA kernel函数，在GPU上执行，从CPU端调用
- `<<<numBlocks, threadsPerBlock>>>`：批量启动CUDA线程（"启动一个CUDA线程块网格"）
- 调用在所有线程终止后返回

### 6.2 Host代码与Device代码的分离

CUDA程序在**静态层面**由程序员明确区分host（CPU）代码和device（GPU）代码。

```cuda
// ===== Host代码：在CPU上串行执行 =====
const int Nx = 12;
const int Ny = 6;
dim3 threadsPerBlock(4, 3);
dim3 numBlocks(Nx/threadsPerBlock.x, Ny/threadsPerBlock.y);
matrixAddDoubleB<<<numBlocks, threadsPerBlock>>>(A, B, C);

// ===== Device代码：在GPU上SPMD执行 =====
__device__ float doubleValue(float x)
{
   return 2 * x;
}

// kernel定义
__global__ void matrixAddDoubleB(float A[Ny][Nx],
                                 float B[Ny][Nx],
                                 float C[Ny][Nx])
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y * blockDim.y + threadIdx.y;
   C[j][i] = A[j][i] + doubleValue(B[j][i]);
}
```

- `__global__`：kernel函数，GPU上执行
- `__device__`：device端辅助函数，只能从GPU代码调用

### 6.3 显式指定线程数量

CUDA kernel启动的线程数量不由数据集合大小决定（不像图形着色器编程中的`map(kernel, collection)`那样）。线程数量在程序中**显式指定**。

当数据大小不是线程块大小的整数倍时，需要向上取整并添加边界检查：

```cuda
const int Nx = 11;  // 不是threadsPerBlock.x的倍数
const int Ny = 5;   // 不是threadsPerBlock.y的倍数

dim3 threadsPerBlock(4, 3);
dim3 numBlocks((Nx+threadsPerBlock.x-1)/threadsPerBlock.x,
               (Ny+threadsPerBlock.y-1)/threadsPerBlock.y);

// 此调用启动72个线程：6个块，每块12个线程
matrixAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
```

```cuda
__global__ void matrixAdd(float A[Ny][Nx],
                          float B[Ny][Nx],
                          float C[Ny][Nx])
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y * blockDim.y + threadIdx.y;

   // 越界保护
   if (i < Nx && j < Ny)
      C[j][i] = A[j][i] + B[j][i];
}
```

### 6.4 CUDA执行模型

```
Host (串行执行)          CUDA Device (SPMD执行)
实现: CPU               实现: GPU
```

Host和Device是分离的执行实体。Host在CPU上串行执行，当调用kernel时，GPU以SPMD方式并行执行大量线程。

---

## 七、CUDA内存模型

### 7.1 Host与Device地址空间分离

```
Host内存地址空间  <--cudaMemcpy-->  Device全局内存地址空间
    (CPU)                                (GPU)
```

Host和Device有**截然不同的地址空间**。需要使用`cudaMemcpy`在两者之间移动数据。

#### cudaMemcpy示例

```cuda
float* A = new float[N];  // 在host内存中分配缓冲区

// 初始化host端数据
for (int i=0; i<N; i++)
   A[i] = (float)i;

int bytes = sizeof(float) * N;
float* deviceA;                 // 在device地址空间分配缓冲区
cudaMalloc(&deviceA, bytes);

// 将数据从host拷贝到device
cudaMemcpy(deviceA, A, bytes, cudaMemcpyHostToDevice);

// 注意：在host端直接访问deviceA[i]是无效操作
// 因为deviceA不是host地址空间的指针
```

### 7.2 Device端三种内存类型

| 内存类型 | 可见性 | 特点 |
|---------|--------|------|
| **Per-thread private memory**（每线程私有内存） | 仅该线程可读写 | 寄存器/线程局部变量 |
| **Per-block shared memory**（每块共享内存） | 块内所有线程可读写 | 片上高速内存，低延迟 |
| **Device global memory**（设备全局内存） | 所有线程可读写 | 设备DRAM，容量大但延迟高 |

这三种不同的地址空间反映了程序中不同的局部性区域——这对GPU实现CUDA的效率有重要影响。

**核心思考**：如果预先知道某些线程访问相同的变量，应该如何调度这些线程？（答案：将它们放入同一个thread block，利用shared memory。）

---

## 八、CUDA代码示例：1D卷积

### 问题描述

```
output[i] = (input[i] + input[i+1] + input[i+2]) / 3.f;
```

每个输出元素需要访问3个输入元素，相邻输出元素之间共享输入数据。

### 版本1：直接使用全局内存

**策略**：每个线程计算一个输出元素，直接从全局内存读取输入。

```cuda
#define THREADS_PER_BLK 128

__global__ void convolve(int N, float* input, float* output) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;  // 线程局部变量
   float result = 0.0f;  // 线程局部变量
   for (int i=0; i<3; i++)
     result += input[index + i];    // 从全局内存读取
   output[index] = result / 3.f;     // 写入全局内存
}
```

```cuda
// Host代码
int N = 1024 * 1024;
cudaMalloc(&devInput, sizeof(float) * (N+2));   // 分配输入数组
cudaMalloc(&devOutput, sizeof(float) * N);       // 分配输出数组
// 正确初始化devInput的内容...

convolve<<<N/THREADS_PER_BLK, THREADS_PER_BLK>>>(N, devInput, devOutput);
```

**问题**：每个线程从全局内存进行3次读取，共产生 3 * 128 = 384 次全局内存加载。相邻线程会重复加载相同的输入元素，存在大量冗余访问。

### 版本2：利用Shared Memory优化

**策略**：将块所需的输入数据协作加载到shared memory中，大幅减少全局内存访问。

```cuda
#define THREADS_PER_BLK 128

__global__ void convolve(int N, float* input, float* output) {
   __shared__ float support[THREADS_PER_BLK+2];        // 每块shared memory分配
   int index = blockIdx.x * blockDim.x + threadIdx.x;  // 线程局部变量

   // 所有线程协作地将块的support区域从全局内存加载到shared memory
   support[threadIdx.x] = input[index];
   if (threadIdx.x < 2) {
      support[THREADS_PER_BLK + threadIdx.x] = input[index+THREADS_PER_BLK];
   }

   __syncthreads();  // 屏障：块内所有线程在此同步

   float result = 0.0f;  // 线程局部变量
   for (int i=0; i<3; i++)
     result += support[threadIdx.x + i];  // 从shared memory读取

   output[index] = result / 3.f;           // 写入全局内存
}
```

**优化分析**：
- 原来需要 3 * 128 = 384 次全局内存加载
- 现在总共只需要 130 次加载指令（128个线程各加载1个 + 2个额外线程加载边界元素）
- shared memory在片上，访问速度远快于全局内存

---

## 九、CUDA同步机制

### 9.1 __syncthreads()

- **屏障（Barrier）**：等待块内所有线程到达此点
- 用于协调块内线程的协作（如shared memory的数据加载和消费）
- 仅在thread block范围内有效

### 9.2 原子操作（Atomic Operations）

- 例如：`float atomicAdd(float* addr, float amount)`
- CUDA提供了对**全局内存地址**和**每块shared memory地址**的原子操作

### 9.3 Host/Device同步

- Kernel返回时存在**隐式屏障**：所有线程完成后才返回host

---

## 十、CUDA抽象总结

| 方面 | 描述 |
|------|------|
| **执行：线程层次** | 批量启动大量线程；两级层次：线程分组为线程块 |
| **分布式地址空间** | 内置memcpy在host/device地址空间之间拷贝；device端三种地址空间：每线程（local）、每块（shared）、每程序（global） |
| **屏障同步** | `__syncthreads()`用于线程块内线程同步 |
| **原子操作** | 用于shared和global变量的额外同步 |

---

## 十一、考虑CUDA语义：资源分配

### 关键问题

启动超过100万个CUDA线程（超过8K个线程块）：

```cuda
convolve<<<N/THREADS_PER_BLK, THREADS_PER_BLK>>>(N, devInput, devOutput);
```

**问题**：运行这个CUDA程序真的会创建100万个局部变量/每线程栈的实例吗？8000个shared变量的实例？

**对比pthread**：`pthread_create()`需要分配：
- 线程栈空间
- 控制块（供OS调度线程使用）

CUDA的实现方式与此**完全不同**——这就是为什么CUDA线程与pthread虽然抽象相似，但实现差异巨大。

---

## 十二、工作分配与可移植性

### 设计理念

CUDA程序应该能在任何规模的GPU上运行而无需修改：

- **高端GPU**：16核
- **中端GPU**：6核

CUDA程序中没有`num_cores`的概念。CUDA线程启动在精神上类似于数据并行模型中的`forall`循环——将问题分解为足够多的独立工作单元，由系统负责将这些工作单元调度到可用的核心上。

---

## 十三、CUDA编译

### 编译后的CUDA device binary包含

```cuda
#define THREADS_PER_BLK 128

__global__ void convolve(int N, float* input, float* output) {
   __shared__ float support[THREADS_PER_BLK+2];  // 每块分配
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   support[threadIdx.x] = input[index];
   if (threadIdx.x < 2) {
      support[THREADS_PER_BLK+threadIdx.x] = input[index+THREADS_PER_BLK];
   }
   __syncthreads();
   float result = 0.0f;
   for (int i=0; i<3; i++)
     result += support[threadIdx.x + i];
   output[index] = result;
}
```

启动8K个线程块时，编译后的binary包含：

1. **程序文本**（指令）
2. **资源需求信息**：
   - 每块128个线程
   - 每线程B字节的local data
   - 每块 128+2 = 130个float（520字节）的shared space

---

## 十四、CUDA线程块调度

### 调度机制

```
Kernel启动命令 ──> 线程块调度器 ──> 将线程块分配到各核心
launch(blockDim, convolve)
    |
    v
 8K个convolve线程块组成的Grid
```

**块的资源需求**（来自编译后的kernel binary）：
- 128个线程
- 520字节shared memory
- 128 * B字节local memory

### 关键假设

**CUDA的核心假设**：线程块的执行可以以**任意顺序**进行（块之间没有依赖关系）。

GPU实现使用**动态调度策略**将线程块（"工作"）映射到核心上，同时尊重资源需求约束。

### 设计模式：工作线程池

这是另一种常见的并行设计模式实例：

```
问题 ──分解──> 子问题（"tasks", "work"） ──分配──> 工作线程池
```

其他例子：
- ISPC的任务启动实现
- Web服务器中的线程池（线程数量是核心数的函数，而非请求数的函数）

---

## 十五、NVIDIA V100 GPU架构详解

### 15.1 V100 SM "子核心"（Sub-core）

每个V100 SM包含4个子核心，每个子核心包含：

| 功能单元 | 描述 |
|---------|------|
| **SIMD fp32单元** | 控制共享于16个单元，每时钟16x乘加（32-wide SIMD操作每2时钟完成） |
| **SIMD int单元** | 控制共享于16个单元，每时钟16x乘/加 |
| **SIMD fp64单元** | 控制共享于8个单元（32-wide SIMD操作每4时钟完成） |
| **Load/Store单元** | 内存访问 |
| **Tensor Core单元** | 深度学习加速 |
| **Warp Selector** | 选择可运行的warp执行下一条指令 |
| **Fetch/Decode** | 指令获取和解码 |

### 15.2 Warp的概念

**Warp = 线程块中连续的32个线程**

```
线程块中的线程0-31   -> Warp 0
线程块中的线程32-63  -> Warp 1
线程块中的线程64-95  -> Warp 2
...
```

- 一个包含256个CUDA线程的线程块映射到**8个warp**（256 / 32 = 8）
- V100的每个子核心可以调度和交错执行最多**16个warp**
- 每个子核心有**64KB寄存器**，整个SM有**256KB寄存器**（4个子核心 x 64KB）
- 寄存器在SM最多64个warp之间分配

### 15.3 SIMT执行模型

**SIMT = Single Instruction, Multiple Thread（单指令，多线程）**

- Warp中的线程如果共享相同的指令，则以**SIMD方式**执行
- 如果32个CUDA线程不共享相同指令，会因**分支发散（divergent execution）**导致性能下降

**关键区别**：
- **ISPC gang**：程序编译为SIMD指令，一个gang中的实例**必须**执行相同指令
- **NVIDIA GPU warp**：硬件**动态检查**32个独立CUDA线程是否共享指令，如果是则SIMD执行所有32个线程。CUDA程序**不编译为SIMD指令**。

warp不是CUDA编程模型的一部分，但它是理解现代NVIDIA GPU上CUDA实现性能的**重要实现细节**。

### 15.4 指令执行示意

```
指令流（warp中所有CUDA线程运行相同的指令流）:
00  fp32  mul r0 r1 r2
01  int32 add r3 r4 r5
02  fp32  mul r6 r7 r8
...

时间（时钟周期）:
时钟0: Fetch -> fp32执行指令00（16个ALU处理前16个线程）
时钟1: fp32执行指令00（16个ALU处理后16个线程）
时钟2: Fetch -> int32执行指令01（16个ALU处理前16个线程）
时钟3: int32执行指令01（16个ALU处理后16个线程）
...
```

由于只有16个ALU，为整个warp（32个线程）运行一条指令需要**两个时钟周期**。

### 15.5 V100 SM 完整结构

```
                    V100 SM (Streaming Multiprocessor)
    ┌─────────────────────────────────────────────────────────┐
    │           Shared Memory + L1 Cache (128 KB)              │
    ├────────────┬────────────┬────────────┬──────────────────┤
    │  Sub-core 0│  Sub-core 1│  Sub-core 2│   Sub-core 3     │
    │  64KB regs │  64KB regs │  64KB regs │   64KB regs     │
    │ 16 warps   │  16 warps  │  16 warps  │   16 warps      │
    ├────────────┴────────────┴────────────┴──────────────────┤
    │       总共: 256KB寄存器, 最多64个warp/SM                  │
    └─────────────────────────────────────────────────────────┘
```

每个SM每个时钟周期的操作：
- 每个子核心从其分区中的16个warp中选择一个可运行的warp
- 每个子核心为warp中的CUDA线程运行下一条指令（根据发散情况可能适用于全部或部分线程）

---

## 十六、V100 GPU整体规格

```
        L2 Cache (6 MB)
             |
    ┌────────┴────────┐
    |   80 个 SM 核心   |
    |                  |
    └────────┬────────┘
             |
     GPU Memory (HBM, 16 GB)
     带宽: 900 GB/sec (4096-bit接口)
```

### V100关键参数

| 参数 | 数值 |
|------|------|
| 时钟频率 | 1.245 GHz |
| SM数量 | 80 |
| fp32乘加ALU | 80 x 4 x 16 = 5,120 个 |
| 峰值fp32性能 | 5,120 x 1.245G x 2 = 约12.7 TFLOPs |
| 最大并发warp | 80 x 64 = 5,120 warps |
| 最大并发CUDA线程 | 163,840 |

*乘加（mul-add）计为2次浮点操作（FLOPs）

---

## 十七、SIMT发散执行详解

现代GPU执行使用纯标量指令流的硬件线程。GPU核心检测不同硬件线程何时执行相同指令，并使用SIMD ALU实现最多SIMD-width个线程的同时执行。

```
执行上下文0 ─┐
执行上下文1 ─┤
执行上下文2 ─┤  (相同指令) --> ALU 0-6 同时执行
执行上下文3 ─┤
执行上下文4 ─┤
执行上下文5 ─┤
执行上下文6 ─┤
执行上下文7 ──  (不同指令) --> ALU 7 被屏蔽（masked off）
```

如果线程6执行与其他线程不同的指令，ALU 6将被"屏蔽掉"——这就是**分支发散**导致性能损失的硬件原因。

---

## 十八、CUDA Kernel运行逐步分析

### 场景设定

假设在一个**虚构的双核GPU**上运行convolve kernel：
- 每个核心有 384个CUDA线程（12个warp）的执行上下文容量
- 每个核心有 1.5 KB shared memory 存储

Kernel需求：
- 每块 128 个CUDA线程（4个warp）
- 每块 520 字节 shared memory

### 逐步执行过程

**Step 1**：Host向GPU发送命令：
```
EXECUTE:    convolve
ARGS:       N, input_array, output_array
NUM_BLOCKS: 1000
```

**Step 2**：调度器将Block 0映射到Core 0（预留128个线程的执行上下文 + 520字节shared storage），NEXT=1

**Step 3**：调度器继续将block映射到可用的执行上下文（交错映射）。Core 0上Block 0和Block 1分别占用Context 0-127；Core 1上类似。

**Step 3（继续）**：每个核心最多容纳2个block。第三个block放不下，因为3 x 520字节 > 1.5 KB shared memory。

**Step 4**：Block 0在Core 0上完成执行。

**Step 5**：Block 4被调度到Core 0（映射到Context 0-127）。

**Step 6**：Block 2在Core 0上完成。

**Step 7**：Block 5被调度到Core 0（映射到Context 128-255）。

这个过程持续进行直到所有1000个block完成。关键观察：
- **线程块之间以任意顺序调度**
- **资源限制决定同时能运行的block数量**（这里是shared memory成为瓶颈）
- **动态调度：一旦block完成，立即为新block释放资源**

---

## 十九、为什么必须为block中所有线程分配执行上下文？

### 问题

如果有一个256线程的block，但SM只有128线程的执行能力。为什么不先运行线程0-127到完成，再运行线程128-255？

### 答案

CUDA kernel可能在block内的线程之间创建依赖关系。最简单的例子是`__syncthreads()`。

```cuda
__syncthreads();  // 屏障：所有256个线程必须都到达此处
```

- **CUDA语义**：block中的线程**确实在并发运行**。如果block中一个线程可运行，它最终就会被执行（不会死锁）。
- 当存在依赖关系时，block中的线程不能被系统以任何顺序执行。
- 因此，block内的**所有线程必须同时存在**并拥有寄存器状态。

---

## 二十、CUDA抽象的实现

| 层面 | 语义 |
|------|------|
| **线程块之间** | 系统假设块之间无依赖；可以任意顺序调度；逻辑上并发（类似于ISPC tasks） |
| **线程块内部** | 块开始执行时，所有线程都存在并已分配寄存器状态；线程块本身就是SPMD程序（类似ISPC gang）；块内线程是并发、协作的"工作者" |

### GPU实现细节

- NVIDIA GPU的**warp**性能特征类似ISPC gang的实例（但warp概念不存在于编程模型中）
- **同一线程块的所有warp被调度到同一个SM上**，允许通过shared memory进行高带宽/低延迟通信
- 当块内所有线程完成时，块资源（shared memory分配、warp执行上下文）被释放给下一个块

---

## 二十一、线程块间的原子操作

### 例子：直方图

```cuda
// 全局内存中的直方图计数器
int counts[10];

// 所有CUDA线程原子地更新全局内存中的共享变量
int* A = {0, 3, 4, 1, 9, 2, ..., 8, 4, 1};  // 0-9之间的整数数组

// 在线程块内:
atomicAdd(&counts[A[i]], 1);
```

这是**有效的CUDA代码**。原子操作用于互斥访问，不影响系统以任意顺序调度block的能力。

### 不合理的CUDA代码示例

```cuda
// 全局内存
int myFlag;  // 假设初始化为0

// Block 0:
// 做一些工作...
atomicAdd(&myFlag, 1);
while(atomicAdd(&myFlag, 0) == 0)  // 忙等待
     { }
// 做更多工作...
```

**问题**：在只有一个线程块的GPU上，这可能导致死锁！因为：
- 如果Block 1先运行并在flag上等待
- 但Block 0永远得不到运行机会
- Block 1将永远等待

这就是为什么**线程块之间不应该有依赖关系**。

---

## 二十二、Bonus："持久线程"（Persistent Thread）编程风格

这种风格引入对GPU实现细节的依赖（不推荐用于可移植代码）：

```cuda
#define THREADS_PER_BLK 128
#define BLOCKS_PER_CHIP 80 * (32*64/128) // 特定于V100 GPU

__device__ int workCounter = 0;          // 全局内存变量

__global__ void convolve(int N, float* input, float* output) {
  __shared__ int startingIndex;
  __shared__ float support[THREADS_PER_BLK+2];

  while (1) {
     if (threadIdx.x == 0)
        startingIndex = atomicInc(workCounter, THREADS_PER_BLK);
     __syncthreads();

     if (startingIndex >= N)
        break;

     int index = startingIndex + threadIdx.x;
     support[threadIdx.x] = input[index];
     if (threadIdx.x < 2)
        support[THREADS_PER_BLK+threadIdx.x] = input[index+THREADS_PER_BLK];
     __syncthreads();

     float result = 0.0f;
     for (int i=0; i<3; i++)
       result += support[threadIdx.x + i];
     output[index] = result;
     __syncthreads();
  }
}

// Host代码
int N = 1024 * 1024;
cudaMalloc(&devInput, N+2);
cudaMalloc(&devOutput, N);
convolve<<<BLOCKS_PER_CHIP, THREADS_PER_BLK>>>(N, devInput, devOutput);
```

**核心思想**：
- 程序员启动恰好能填满GPU的线程块数量
- 工作分配完全由应用程序实现（通过`atomicInc`获取工作）
- 绕过了GPU的线程块调度器
- 程序员的心理模型变为：**所有CUDA线程同时运行在GPU上**

**缺点**：程序对GPU实现做了假设（依赖具体的核心数和每核心支持的block数），牺牲了可移植性。

---

## 二十三、CUDA总结

### 执行语义

1. **问题分解为线程块**：体现了数据并行模型的精神（机器无关：系统将block调度到任意数量的核心上）
2. **线程块内的线程确实并发运行**（它们必须如此，因为它们需要协作）
3. **单个线程块内部**：SPMD + 共享地址空间编程
4. **三种不同但微妙的执行模型差异**需要理解清楚

### 内存语义

1. **分布式地址空间**：Host/Device内存分离
2. **Device内存内部**：线程局部 / 块共享 / 全局变量
3. Loads/Stores在它们之间移动数据（可以将local/shared/global视为不同的地址空间）

### 关键实现细节

1. **线程块内线程被调度到同一个GPU SM上**，以允许通过shared memory快速通信
2. **线程块内线程分组为warp**，在GPU硬件上进行SIMT执行

---

## 二十四、补充说明

1. 本讲讨论的是为GPU**可编程核心**编写CUDA程序。工作（由CUDA kernel启动描述）通过硬件工作调度器映射到核心上。

2. GPU中仍然存在图形管线接口用于驱动图形执行，以及大量为加速图形管线操作而存在的非可编程硬件功能——这些在运行CUDA程序时基本上被"关闭"。

3. 本讲**没有涉及**SM中用于深度学习的数百TFLOPS的**Tensor Core**——这是后续课程的主题。

---

## 关键术语对照

| 英文 | 中文 |
|------|------|
| SIMD (Single Instruction, Multiple Data) | 单指令多数据 |
| SIMT (Single Instruction, Multiple Thread) | 单指令多线程 |
| SPMD (Single Program, Multiple Data) | 单程序多数据 |
| SM (Streaming Multiprocessor) | 流式多处理器 |
| Warp | 线程束（32个线程为一组） |
| Thread Block | 线程块 |
| Grid | 网格 |
| Shared Memory | 共享内存 |
| Global Memory | 全局内存 |
| Occupancy | 占用率 |
| Shader | 着色器 |
| GPGPU | GPU通用计算 |
| HBM (High Bandwidth Memory) | 高带宽内存 |
| Kernel | 核函数 |
| Divergent Execution | 分支发散执行 |
| Persistent Thread | 持久线程 |
