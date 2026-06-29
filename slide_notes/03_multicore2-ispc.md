# Lecture 3: 多核编程与ISPC

## 课程信息

- **课程**：Parallel Computing (CS149, Stanford University)
- **学期**：Fall 2025
- **主题**：
  - 多核架构，第二部分（延迟/带宽问题）
  - 并行编程抽象

---

## 1. 上节课回顾与新话题

### 1.1 回顾：吞吐量计算的三大思想

上节课介绍了吞吐量计算硬件的三个核心思想：

1. **多核执行 (Multi-core execution)**
2. **SIMD 执行 (Single Instruction Multiple Data)**
3. **硬件多线程 (Hardware multi-threading)** —— 上节课未完成，本节课从这开始

---

## 2. 延迟与带宽 (Latency and Bandwidth)

### 2.1 思想实验：逐元素向量乘法

**任务**：对两个包含数百万个元素的向量 A 和 B 做逐元素乘法：

1. 加载输入 A[i]
2. 加载输入 B[i]
3. 计算 A[i] × B[i]
4. 将结果存入 C[i]

**思考问题**：这个应用在现代吞吐量导向的并行处理器上能高效运行吗？

### 2.2 NVIDIA V100 GPU 结构

V100 GPU 的内部结构：

- **80 个 SM（流式多处理器）核心**
- 每个 SM 有 64 个 fp32 ALU（算术逻辑单元）
- 总共：80 × 64 = **5120 个 ALU**
- L2 Cache：6 MB
- GPU 内存（HBM2）：16 GB
- 内存带宽：**900 GB/s**（4096 位接口）

**核心问题**：每个时钟周期都要为所有这些 ALU 提供数据，这是一个巨大的挑战。

### 2.3 高速公路类比：理解延迟与带宽

#### 2.3.1 基本概念

用旧金山到斯坦福的高速公路来类比：

| 参数 | 值 |
|------|-----|
| 距离 | ~50 km |
| 车速 | 100 km/hr |
| 延迟 | 0.5 小时（单程所需时间） |
| 吞吐量 | 2 辆车/小时（假设每次只有一辆车在路上） |

#### 2.3.2 提高吞吐量的两种方式

| 方式 | 说明 | 效果 |
|------|------|------|
| 方式1：加速行驶 | 车速提升到 200 km/hr | 吞吐量 = 4 辆车/小时 |
| 方式2：增加车道 | 4 条车道，每条车速 100 km/hr | 吞吐量 = 8 辆车/小时（每条车道 2 辆/小时） |

**关键洞察**：增加车道（并行资源）比单纯加速更有效地提高了吞吐量。

#### 2.3.3 更高效地使用高速公路

- **稀疏排列**：车辆间隔 1 km → 吞吐量 = 100 辆车/小时
- **密集排列**：同一速度下缩小车间距 → 吞吐量 = 400 辆车/小时

**类比对应**：
- 速度 → 不可改变的物理限制（如光速）
- 车道数 → 并行硬件资源
- 车间距 → 流水线中的利用率

### 2.4 术语定义

| 术语 | 定义 | 示例 |
|------|------|------|
| **内存带宽 (Memory Bandwidth)** | 内存系统向处理器提供数据的速率 | 20 GB/s（约 4 项/秒） |
| **延迟 (Latency)** | 传输单个数据项所需的时间 | ~2 秒（每个项） |

**重要关系**：带宽和延迟是独立的属性。可以提高带宽而不降低延迟（如增加车道而不改变车速）。

### 2.5 洗衣服示例：理解流水线化

**单次洗衣操作**：
- 洗衣机：45 分钟
- 烘干机：60 分钟
- 折叠：15 分钟
- **一次洗衣的延迟**：2 小时

#### 提高吞吐量的两种方案

**方案一：复制执行资源**
- 使用两台洗衣机、两台烘干机、叫上一个朋友
- 两批衣物的延迟：仍为 2 小时
- 吞吐量翻倍：1 批/小时
- 代价：资源翻倍

**方案二：流水线化 (Pipelining)**
- 在一批衣物进入烘干机的同时，另一批进入洗衣机
- 延迟：单批仍为 2 小时
- 吞吐量：1 批/小时
- 资源：一台洗衣机、一台烘干机

**关键思想**：流水线化——用更少的资源达到相同的高吞吐量，代价是单次任务的延迟不变。

### 2.6 管道类比：瓶颈分析

两个连接的水管：
- 管道1：最大流量 100 升/秒
- 管道2：最大流量 50 升/秒

**结论**：串联系统的最大流量 = min(100, 50) = **50 升/秒**，即系统吞吐量受最窄管道的限制。

### 2.7 将概念应用到计算机

考虑以下程序模式：多个线程重复执行三个依赖指令：

```
1. X = load 64 bytes
2. Y = add x + x
3. Z = add x + y
```

**多线程核心假设**：
- 每时钟执行一个数学操作
- 可以同时发射加载指令和数学操作
- 每时钟从内存接收 8 字节
- 假设有足够的硬件线程来隐藏内存延迟

#### 2.7.1 时序分析

**核心参数**：
- 一次加载 64 字节需要 8 个时钟周期（每周期 8 字节）
- 最多 3 个未完成的加载请求
- 每周期可执行 1 次加法（可与加载并发射）

**时序图分析**（Page 18）：

| 时钟 | 活动 | 进行中的加载 |
|------|------|-------------|
| 1 | 发射加载请求 | 1 |
| 2 | 发射第二个加载请求 | 2 |
| 3 | 发射第三个加载请求 | 3 |
| 4-7 | 等待数据，发射加载请求 | 3 |
| 8 | 数据到达，执行加法 | 3（已满） |
| 后续 | 停滞等待数据... | ... |

#### 2.7.2 核心结论：内存带宽绑定

- **内存带宽绑定执行**！指令执行速率由内存提供数据的速率决定
- 红色区域（停滞）：核心在等待下一次指令所需的数据
- 内存 100% 时间在传输数据，无法更快地传送数据
- **关键洞察**：在稳态下，核心的未充分利用率仅取决于**指令吞吐量和内存吞吐量**的函数，而**不是**内存延迟或未完成内存请求数量的函数

### 2.8 高带宽内存

- 现代 GPU 使用位于处理器附近的高带宽内存
- 示例：V100 使用 HBM2，带宽达 **900 GB/s**

### 2.9 回到思想实验：带宽限制分析

**计算分析（逐元素向量乘法）**：
- 三次内存操作（加载 A[i], 加载 B[i], 存储 C[i]）共 12 字节
- 每次 MUL（乘法）操作需要 12 字节

**V100 GPU 能力**：
- 每时钟可执行 5120 次 fp32 MUL（@ 1.6 GHz）
- 需要约 **98 TB/s** 的带宽才能让所有功能单元保持忙碌
- 实际带宽仅有 900 GB/s

**结论**：
- GPU 效率 < 1%（但比 8 核 CPU 快得多）
- 8 核 Xeon E5v4 CPU 运行此计算的效率约 3%（连接到 76 GB/s 内存总线）
- **此计算是带宽受限的 (bandwidth limited)！**

### 2.10 总结：带宽是关键资源

在现代计算中，**带宽是真正的瓶颈资源**。

高效的并行程序应该：
1. **减少从内存获取数据的频率**
   - 重用同一线程之前加载的数据（时间局部性优化）
   - 在线程间共享数据（线程间协作）
2. **倾向于执行额外的算术运算，而不是存储/重新加载值**
   - 数学运算是"免费的"（相对于内存访问而言）
3. **核心原则**：程序必须不频繁地访问内存，才能高效利用现代处理器

### 2.11 指令流水线示例

四级指令流水线：

| 阶段 | 名称 | 说明 |
|------|------|------|
| IF | 指令获取 (Instruction Fetch) | 从内存获取指令 |
| D | 指令解码+寄存器读取 (Decode) | 解码指令并读取寄存器 |
| EX | 执行操作 (Execute) | 执行算术/逻辑操作 |
| WB | 写回 (Write Back) | 将结果写回寄存器 |

**性能指标**：
- **延迟**：1 条指令需要 4 个周期完成
- **吞吐量**：每周期 1 条指令（因为流水线每个阶段同时处理不同的指令）

**重要说明**：
- 当说处理器"每周期完成一次乘法"时，指的是**指令吞吐量**，而不是**延迟**
- 单条乘法的延迟可能远超一个周期，但由于流水线化，可以每周期启动一条新指令
- 现代 CPU 的指令流水线可以深达约 20 级，且实际流水线深度因指令类型而异

---

## 3. 抽象 vs. 实现 (Abstraction vs. Implementation)

### 3.1 核心概念

这是本节课后半部分的主题，也是整个课程中反复出现的重要思想。

| 概念 | 定义 | 关键问题 |
|------|------|----------|
| **语义 (Semantics)** | 编程模型提供的操作的含义是什么？ | 给定程序和操作的语义，程序将计算出什么答案？ |
| **实现 (Implementation)** | 如何在并行机器上计算答案？（也称为调度 scheduling） | 程序的每个部分在并行计算机上如何执行？哪些操作由哪个线程/执行单元/向量通道执行？ |

### 3.2 学习目标

作为学生，你的目标应该是：

> 给定一个程序，并了解并行编程模型的实现方式，能够在脑中"追踪"并行计算机的每个部分在程序执行的每一步都在做什么。

**将编程抽象的语义与其实现细节混淆是本课程中常见的困惑来源。**

---

## 4. ISPC 编程 (Intel SPMD Program Compiler)

### 4.1 ISPC 简介

- **ISPC** = Intel SPMD Program Compiler
- **SPMD** = Single Program Multiple Data（单程序多数据）
- 官网：http://ispc.github.com/
- 推荐阅读：[The Story of ISPC](https://pharr.org/matt/blog/2018/04/30/ispc-all.html) (by Matt Pharr)

### 4.2 示例程序：Taylor 展开计算 sin(x)

#### 4.2.1 原始 C++ 串行版本

使用泰勒展开式计算 sin(x)：sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...

```c
void sinx(int N, int terms, float* x, float* result)
{
   for (int i=0; i<N; i++)
   {
    float value = x[i];
    float numer = x[i] * x[i] * x[i];
    int denom = 6;  // 3!
    int sign = -1;
    for (int j=1; j<=terms; j++)
    {
       value += sign * numer / denom;
       numer *= x[i] * x[i];
       denom *= (2*j+2) * (2*j+3);
       sign *= -1;
      }
      result[i] = value;
   }
}
```

#### 4.2.2 C++ 调用端代码

```c
#include "sinx.h"
int main(int argc, void** argv) {
  int N = 1024;
  int terms = 5;
  float* x = new float[N];
  float* result = new float[N];
  // initialize x here
  sinx(N, terms, x, result);
  return 0;
}
```

**串行执行流程**：main() → 调用 sinx() → 串行执行 for 循环 → 返回 main()

### 4.3 ISPC 版本：交错分配 (Interleaved Assignment)

#### 4.3.1 ISPC 代码

```c
export void ispc_sinx(
   uniform int N,
   uniform int terms,
   uniform float* x,
   uniform float* result)
{
   // assume N % programCount = 0
   for (uniform int i=0; i<N; i+=programCount)
   {
    int idx = i + programIndex;
    float value = x[idx];
    float numer = x[idx] * x[idx] * x[idx];
    uniform int denom = 6;  // 3!
    uniform int sign = -1;
    for (uniform int j=1; j<=terms; j++)
    {
       value += sign * numer / denom;
       numer *= x[idx] * x[idx];
       denom *= (2*j+2) * (2*j+3);
       sign *= -1;
      }
      result[idx] = value;
   }
}
```

#### 4.3.2 ISPC 关键语言概念

| 关键字/概念 | 含义 | 说明 |
|-------------|------|------|
| `programCount` | gang 中同时执行的程序实例数量 | uniform 值，所有实例相同 |
| `programIndex` | 当前实例在 gang 中的 ID（从 0 开始） | varying 值，每个实例不同 |
| `uniform` | 类型修饰符，所有实例的值相同 | 纯粹用于优化，非正确性必需 |

#### 4.3.3 SPMD 编程抽象

**调用 ISPC 函数的执行流程**：

1. **串行执行**：C++ 代码在 main() 中运行
2. **调用 ISPC 函数**：`ispc_sinx(N, terms, x, result);`
3. **SPMD 执行**：生成一个由 `programCount` 个 ISPC "程序实例" 组成的"gang"
   - 所有实例**并发**运行 ISPC 代码
   - 每个实例有自己的本地变量副本（蓝色变量）
   - `uniform` 变量在所有实例间共享同一值
4. **返回**：所有实例完成后，控制权返回串行 C++ 代码

#### 4.3.4 交错分配图解

当 `programCount = 8` 时，数组元素的分配方式如下：

```
实例 0 (programIndex=0): 处理元素 0, 8, 16, 24, ...
实例 1 (programIndex=1): 处理元素 1, 9, 17, 25, ...
实例 2 (programIndex=2): 处理元素 2, 10, 18, 26, ...
...
实例 7 (programIndex=7): 处理元素 7, 15, 23, 31, ...
```

循环结构：`for (i=0; i<N; i+=programCount)` 中，`idx = i + programIndex`。因此第一次迭代时，实例 0 处理 idx=0，实例 1 处理 idx=1，依此类推。下一次迭代（i+=8），实例 0 处理 idx=8，实例 1 处理 idx=9，以此类推。

#### 4.3.5 ISPC 的 SIMD 实现

- ISPC 编译器生成 SIMD 指令实现 gang 抽象
- gang 中的实例数量即硬件的 SIMD 宽度（或其小整数倍）
- ISPC 编译器生成的 `.o` 文件中包含 SIMD 指令
- C++ 代码像往常一样链接到生成的目标文件

**交错分配的优势**：`float value = x[idx];` 对于所有程序实例，8 个值在内存中是**连续的**，可以用**单条打包向量加载指令**（vmovaps / `_mm256_load_ps()`）高效实现。

### 4.4 ISPC 版本 2：分块分配 (Blocked Assignment)

#### 4.4.1 ISPC 代码

```c
export void ispc_sinx_v2(
   uniform int N,
   uniform int terms,
   uniform float* x,
   uniform float* result)
{
   // assume N % programCount = 0
   uniform int count = N / programCount;
   int start = programIndex * count;
   for (uniform int i=0; i<count; i++)
   {
    int idx = start + i;
    float value = x[idx];
    float numer = x[idx] * x[idx] * x[idx];
    uniform int denom = 6;  // 3!
    uniform int sign = -1;
    for (uniform int j=1; j<=terms; j++)
    {
       value += sign * numer / denom;
       numer *= x[idx] * x[idx];
       denom *= (j+3) * (j+4);
       sign *= -1;
      }
      result[idx] = value;
   }
}
```

#### 4.4.2 分块分配图解

当 `programCount = 8` 且 `N = 64` 时：

```
count = 64/8 = 8（每个实例处理 8 个连续元素）
start = programIndex * 8

实例 0 (programIndex=0): start=0,  处理元素 0, 1, 2, 3, 4, 5, 6, 7
实例 1 (programIndex=1): start=8,  处理元素 8, 9, 10, 11, 12, 13, 14, 15
实例 2 (programIndex=2): start=16, 处理元素 16, 17, 18, 19, 20, 21, 22, 23
...
实例 7 (programIndex=7): start=56, 处理元素 56, 57, 58, 59, 60, 61, 62, 63
```

#### 4.4.3 分块分配的性能问题

`float value = x[idx];` 对于所有程序实例，8 个值在内存中是**非连续的**。需要使用 **gather 指令**（vgatherdps / `_mm256_i32gather_ps()`）实现。

**Gather 指令**：
- 更复杂
- 更昂贵的 SIMD 指令
- 性能通常低于打包加载

### 4.5 交错分配 vs. 分块分配

| 特性 | 交错分配 (Interleaved) | 分块分配 (Blocked) |
|------|----------------------|-------------------|
| 实例间的数据访问模式 | 每个实例访问跨步数据 | 每个实例访问连续块 |
| SIMD 内存操作 | 打包向量加载 (vmovaps) | GATHER 指令 (vgatherdps) |
| 内存访问效率 | 更高效（连续内存访问） | 较低效（非连续内存访问） |
| 使用场景 | 当处理独立元素时，交错分配通常更好 | 当需要数据局部性或有特殊访问需求时 |

### 4.6 foreach 抽象：提高抽象层次

#### 4.6.1 foreach 语法

```c
export void ispc_sinx(
   uniform int N,
   uniform int terms,
   uniform float* x,
   uniform float* result)
{
   foreach (i = 0 ... N)
   {
    float value = x[i];
    float numer = x[i] * x[i] * x[i];
    uniform int denom = 6;  // 3!
    uniform int sign = -1;
    for (uniform int j=1; j<=terms; j++)
    {
       value += sign * numer / denom;
       numer *= x[i] * x[i];
       denom *= (2*j+2) * (2*j+3);
       sign *= -1;
      }
      result[i] = value;
   }
}
```

#### 4.6.2 foreach 的关键特性

- `foreach` 声明**并行循环迭代**
- 程序员说："这些是**整个 gang**（而非每个实例）必须执行的迭代"
- ISPC 实现负责将迭代分配给 gang 中的程序实例
- 程序员不需要手动处理 `programIndex` 和 `programCount`

#### 4.6.3 foreach 的实现方式

以下四种实现都可以正确执行相同的 foreach 代码：

**实现 1：程序实例 0 执行所有迭代**（当 programCount == 0 时）
```c
if (programCount == 0) {
   for (int i=0; i<N; i++) {
     // do work for iteration i here...
  }
}
```

**实现 2：交错分配迭代给程序实例**
```c
// assume N % programCount = 0
for (uniform int loop_i=0; loop_i<N; loop_i+=programCount) {
  int i = loop_i + programIndex;
  // do work for iteration i here...
}
```

**实现 3：分块分配迭代给程序实例**
```c
// assume N % programCount = 0
uniform int count = N / programCount;
int start = programIndex * count;
for (uniform int loop_i=0; loop_i<count; loop_i++) {
    int i = start + loop_i;
    // do work for iteration i here...
}
```

**实现 4：动态分配迭代给实例**
```c
uniform int nextIter;
if (programCount == 0)
  nextIter = 0;
int i = atomic_add_local(&nextIter, 1);
while (i < N) {
  // do work for iteration i here...
  i = atomic_add_local(&nextIter, 1);
}
```

**关键洞察**：使用 foreach 后，程序员"思考迭代，而不是并行执行"。在简单情况下，foreach 让程序员几乎可以像写串行程序一样表达程序。

### 4.7 foreach 示例：理解程序行为

#### 4.7.1 示例 1：绝对值 + 重复

```c
// ISPC code:
export void absolute_repeat(
   uniform int N,
   uniform float* x,
   uniform float* y)
{
   foreach (i = 0 ... N)
   {
     if (x[i] < 0)
        y[2*i] = -x[i];
     else
        y[2*i] = x[i];
     y[2*i+1] = y[2*i];
 }
}

// main C++ code:
const int N = 1024;
float* x = new float[N/2];
float* y = new float[N];
// initialize N/2 elements of x here
// call ISPC function
absolute_repeat(N/2, x, y);
```

**程序功能**：计算 x 元素的绝对值，然后在输出数组 y 中重复两次。

**安全说明**：此程序是安全的——每个 foreach 迭代写入不同位置，没有实例间冲突。

#### 4.7.2 示例 2：移位负数（有未定义行为）

```c
// ISPC code:
export void shift_negative(
   uniform int N,
   uniform float* x,
   uniform float* y)
{
   foreach (i = 0 ... N)
   {
       if (i >= 1 && x[i] < 0)
       y[i-1] = x[i];
     else
       y[i] = x[i];
 }
}

// main C++ code:
const int N = 1024;
float* x = new float[N];
float* y = new float[N];
// initialize N elements of x
// call ISPC function
shift_negative(N, x, y);
```

**程序问题**：**输出是未定义的！** 多个迭代的循环体可能**写入相同的内存位置**。

例如，当 `x[1] < 0` 时，迭代 i=1 写入 `y[0]`；同时迭代 i=0 也写入 `y[0]`。由于并行执行，两个实例可能同时写入同一位置，导致数据竞争。

**教训**：即使使用 foreach，程序员仍需注意并行安全性。ISPC 是低级语言——通过暴露 programIndex 和 programCount，程序员必须定义每个实例做什么工作、访问什么数据。

### 4.8 数组求和：正确与错误的写法

#### 4.8.1 错误版本 1：类型错误

```c
export uniform float sum_incorrect_1(
   uniform int N,
   uniform float* x)
{
   float sum = 0.0f;     // sum 是 varying 类型
   foreach (i = 0 ... N)
   {
      sum += x[i];        // 每个实例有不同的 sum 值
   }
   return sum;            // 返回类型期望 uniform float，但 sum 是 varying
}
```

**错误**：`sum` 是 `float` 类型（每个实例不同值），但返回类型是 `uniform float`。x[i] 在每个实例中有不同值，所以该操作将不同的值加到不同的 sum 副本上。

#### 4.8.2 错误版本 2：类型错误

```c
export uniform float sum_incorrect_2(
   uniform int N,
   uniform float* x)
{
   uniform float sum = 0.0f;
   foreach (i = 0 ... N)
   {
      sum += x[i];       // uniform 对象与 varying 对象运算
   }
   return sum;
}
```

**错误**：`sum` 是 `uniform float`（所有实例共享一个值），但 `x[i]` 是 `varying`（每个实例不同）。这种 uniform-varying 不匹配在 foreach 体内会产生歧义。两个版本都会产生**编译时类型错误**。

#### 4.8.3 正确版本：使用 reduce_add()

```c
export uniform float sum_array(
   uniform int N,
   uniform float* x)
{
   uniform float sum;
   float partial = 0.0f;
   foreach (i = 0 ... N)
   {
      partial += x[i];     // 每个实例积累私有部分和
   }
   // reduce_add() 是 ISPC 跨实例标准库函数
   sum = reduce_add(partial);
   return sum;
}
```

**执行逻辑**：
1. 每个实例使用自己的私有 `partial` 变量累加部分和（无通信）
2. 使用 `reduce_add()` 跨实例通信原语对所有部分和求和
3. 结果是所有程序实例的**同一总值**（`reduce_add()` 返回 uniform float）

**等价的 AVX 内在函数实现**：

```c
float sum_summary_AVX(int N, float* x) {
  float tmp[8];  // assume 16-byte alignment
  __m256 partial = _mm256_broadcast_ss(0.0f);
  for (int i=0; i<N; i+=8)
    partial = _mm256_add_ps(partial, _mm256_load_ps(&x[i]));
  _mm256_store_ps(tmp, partial);
  float sum = 0.f;
  for (int i=0; i<8; i++)
    sum += tmp[i];
  return sum;
}
```

**自测**：如果你理解了这个 AVX 实现为什么能正确实现 ISPC gang 抽象的语义，那么你就很好地掌握了 ISPC。

### 4.9 ISPC 跨实例操作 (Cross-Instance Operations)

ISPC 提供了以下跨程序实例通信原语：

| 操作 | 函数签名 | 说明 |
|------|----------|------|
| **归约求和** | `uniform int64 reduce_add(int32 x);` | 计算 gang 中所有实例变量值的和 |
| **归约求最小值** | `uniform int32 reduce_min(int32 a);` | 计算 gang 中所有值的最小值 |
| **广播** | `int32 broadcast(int32 value, uniform int index);` | 将指定实例的值广播到 gang 中所有实例 |
| **旋转/移位** | `int32 rotate(int32 value, uniform int offset);` | 对所有 i，将实例 i 的值传给实例 `(i+offset) % programCount` |

### 4.10 ISPC：抽象 vs. 实现

| 层面 | 概念 | 说明 |
|------|------|------|
| **编程抽象** | SPMD（单程序多数据） | 程序员"思考"：运行一个 gang 就是生成 programCount 个逻辑指令流，每个有不同的 programIndex |
| **实现** | SIMD（单指令多数据） | ISPC 编译器生成向量指令（AVX2、ARM NEON 等），执行 ISPC gang 的逻辑。编译器处理条件控制流到向量指令的映射（通过屏蔽向量通道等，类似 Assignment 1 中的手动操作） |

**ISPC 语义的复杂之处**：SPMD 抽象 + uniform 值（允许实现细节在一定程度上透过抽象"泄露"）。

### 4.11 SPMD 编程模型总结

```
单线程控制 → 调用 SPMD 函数 → 多实例并行执行 → SPMD 函数返回 → 恢复单线程控制
```

**SPMD 核心思想**：
- 定义一个函数
- 在**不同的输入参数**上**并行运行该函数的多个实例**
- 所有实例执行相同的程序，但处理不同的数据

### 4.12 ISPC Task：实现多核执行

- ISPC 的 **gang 抽象**通过在一个 x86 CPU 核心的单线程内执行的 SIMD 指令实现
- 前面展示的所有 ISPC 代码仅在**一个核心**上执行
- ISPC 还包含另一个抽象：**task（任务）**
- **task 用于实现多核执行**——将工作分配到多个 CPU 核心上
- 这将在 Assignment 1 中详细讨论

---

## 5. 更高级别的并行抽象

### 5.1 如果 ISPC 不是低级语言？

考虑完全隐藏 programIndex 和 programCount：

```c
export void ispc_function(
   int    N,
   float* x,
   float* y)
{
   // 所有逻辑在 foreach 外必须使用 uniform 值
   int twoN = 2 * N;
   foreach (i = 0 ... twoN)
   {
     float val = x[i];
     float result;
     // do work here to compute
     // result from val
     y[i] = result;
   }
}
```

**关键变化**：
- 不需要 `uniform` 关键字（所有 foreach 外的变量自动是 uniform）
- foreach 内的变量自动是 varying
- 程序员几乎不需要考虑程序实例

### 5.2 另一个选择：禁止数组索引

更进一步——完全不允许数组索引，只对"集合"数据结构的每个元素调用计算：

```python
# NumPy 风格
import numpy as np

X = np.arange(15)  # 创建数组 [0, 1, 2, 3, ..., 14]
Y = np.arange(15)
Z = X + Y;          # Z = [0, 2, 4, 6, ...]  —— 逐元素操作
```

这种模型对 NumPy、PyTorch 等库的使用者来说应该非常熟悉：
- 程序员不编写循环
- 不进行数据索引
- 操作应用于整个集合

---

## 6. 课程总结

1. **编程模型**（如 ISPC 的 SPMD）提供了一种组织并行程序的思维方式
2. 它们提供允许多种有效实现的**抽象**
3. **核心思想**：在课程的剩余部分，始终要思考**抽象 vs. 实现**的区别
   - **抽象（语义）**：程序"意味着"什么？它将计算什么答案？
   - **实现（调度）**：程序在并行机器上实际如何执行？

### 关键知识点回顾

| 主题 | 关键概念 |
|------|----------|
| **延迟与带宽** | 带宽是独立的资源；流水线化利用时间重叠提高吞吐量而不减少延迟 |
| **带宽瓶颈** | 逐元素向量乘法等简单计算是带宽受限的；需要约 98 TB/s 才能让现代 GPU 满负荷运转 |
| **优化策略** | 重用数据、共享数据、用计算换内存访问 |
| **SPMD 模型** | 单个程序，多个数据；ISPC 中调用函数时创建 gang 的程序实例 |
| **交错 vs. 分块** | 交错分配利用连续内存访问（打包加载）；分块分配需要 GATHER 指令 |
| **foreach** | 高级抽象，隐藏 programIndex/programCount 细节；程序员专注于迭代 |
| **跨实例操作** | reduce_add、broadcast、rotate 等原语实现实例间通信 |
| **抽象 vs. 实现** | SPMD（编程抽象） vs. SIMD（硬件实现）；理解两者关系至关重要 |
