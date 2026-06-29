# Lecture 4: 并行编程基础

> Stanford CS149, Fall 2025 — Lecture 4: Parallelizing Code: The Programming Thought Process

---

## 目录

1. [ISPC 回顾与进阶](#1-ispc-回顾与进阶)
2. [SPMD 编程模型](#2-spmd-编程模型)
3. [抽象与实现](#3-抽象与实现)
4. [创建并行程序的思维过程](#4-创建并行程序的思维过程)
5. [问题分解与 Amdahl 定律](#5-问题分解与-amdahl-定律)
6. [任务分配（Assignment）](#6-任务分配assignment)
7. [编排（Orchestration）与同步](#7-编排orchestration与同步)
8. [案例研究：2D 网格求解器](#8-案例研究2d-网格求解器)
9. [数据并行模型 vs 共享地址空间模型](#9-数据并行模型-vs-共享地址空间模型)
10. [总结](#10-总结)

---

## 1. ISPC 回顾与进阶

### 1.1 sinx() 示例（来自 Lecture 3）

回顾之前介绍的 ISPC 代码——计算 sin(x) 的泰勒展开：

```cpp
// ISPC 代码: sinx.ispc
export void ispc_sinx(
   uniform int N,
   uniform int terms,
   uniform float* x,
   uniform float* result)
{
   // 假设 N % programCount = 0
   for (uniform int i=0; i<N; i+=programCount)
   {
      int idx = i + programIndex;
      float value = x[idx];
      float numer = x[idx] * x[idx] * x[idx];
      uniform int denom = 6;  // 3!
      uniform int sign = -1;
      for (uniform int j=1; j<=terms; j++)
      {
         value += sign * numer / denom
         numer *= x[idx] * x[idx];
         denom *= (2*j+2) * (2*j+3);
         sign *= -1;
      }
      result[idx] = value;
   }
}
```

```cpp
// C++ 代码: main.cpp
#include "sinx_ispc.h"
int main(int argc, void** argv) {
  int N = 1024;
  int terms = 5;
  float* x = new float[N];
  float* result = new float[N];

  // 初始化 x
  // 执行 ISPC 代码
  ispc_sinx(N, terms, x, result);
  return 0;
}
```

SPMD 编程抽象的关键点：
- 调用 ISPC 函数会**产生（spawn）一个 "gang"（帮派）**，包含多个 ISPC "程序实例"（program instances）
- 所有实例**并发执行**同一段 ISPC 代码
- 每个实例有**自己的局部变量副本**（代码中的蓝色变量）
- 函数返回时，所有实例都已完成

### 1.2 ISPC 的关键概念

- **programCount**：gang 中同时执行的实例数量（是一个 uniform 值）
- **programIndex**：当前实例在 gang 中的 ID（是一个 varying 值，即每个实例不同）
- **uniform**：类型修饰符。所有实例在该变量上拥有相同的值。其使用纯粹是为了**优化**，并非正确性所必需。

---

## 2. SPMD 编程模型

### 2.1 SPMD 执行模型

SPMD = **Single Program, Multiple Data**（单程序，多数据）

执行流程：
```
单线程控制流 (C 代码)
    |
    v
调用 SPMD 函数 (ispc_sinx)
    |
    v
SPMD 执行：函数的多个实例并行运行
（多个逻辑线程）
    |
    v
SPMD 函数返回
    |
    v
恢复单线程控制流 (C 代码)
```

在这种模型中：
- 定义一个函数，以**不同的输入参数并行运行该函数的多个实例**
- ISPC 编译器生成的实现使用 **SIMD 指令**（如 AVX2、ARM NEON）来实现 gang 抽象
- gang 中实例的数量是硬件 SIMD 宽度的整数倍
- ISPC 编译器生成包含 SIMD 指令的 .o 目标文件，C++ 代码正常链接

### 2.2 交错分配 vs 块分配

**版本 1：交错（Interleaved）分配**

```cpp
for (uniform int i=0; i<N; i+=programCount)
{
    int idx = i + programIndex;  // 交错索引
    // ...
}
```

- 相邻的 program instance 处理相邻的数组元素
- 可以利用**打包向量加载指令**（如 `vmovaps`）高效实现，因为 8 个值在内存中连续
- 例如：instance 0 处理元素 0,8,16,...；instance 1 处理元素 1,9,17,...

**版本 2：块（Blocked）分配**

```cpp
uniform int count = N / programCount;
int start = programIndex * count;
for (uniform int i=0; i<count; i++)
{
    int idx = start + i;  // 块内连续索引
    // ...
}
```

- 每个 program instance 处理**连续的一块**数组元素
- 例如：instance 0 处理元素 0,1,2,...；instance 1 处理元素 8,9,10,...
- `float value = x[idx];` 此时访问的是**8 个非连续的内存位置**
- 需要使用 **gather 指令**（如 `vgatherdps`）实现，gather 是更复杂、更昂贵的 SIMD 指令

### 2.3 foreach 抽象

ISPC 提供 `foreach` 来提升抽象层次：

```cpp
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
         value += sign * numer / denom
         numer *= x[i] * x[i];
         denom *= (2*j+2) * (2*j+3);
         sign *= -1;
      }
      result[i] = value;
   }
}
```

**关键思想：**
- `foreach` 声明了**并行循环迭代**
- 程序员只需说：这些是整个 gang 需要执行的迭代
- ISPC 的实现负责将迭代分配给 gang 中的各个 program instance
- 程序员**不需要关心 programIndex 和 programCount**

**foreach 的可能实现方式：**

1. **实现 1**：只有 instance 0 执行所有迭代（退化情况）
2. **实现 2（交错分配）**：
   ```cpp
   for (uniform int loop_i=0; loop_i<N; loop_i+=programCount)
   {
       int i = loop_i + programIndex;
       // 执行迭代 i 的工作
   }
   ```
3. **实现 3（块分配）**：
   ```cpp
   uniform int count = N / programCount;
   int start = programIndex * count;
   for (uniform int loop_i=0; loop_i<count; loop_i++)
   {
       int i = start + loop_i;
       // 执行迭代 i 的工作
   }
   ```
4. **实现 4（动态分配）**：使用原子操作动态分配迭代给实例

使用 `foreach` 的关键优势：程序员可以像**编写顺序程序一样**表达并行计算，而不必思考并行执行的细节。

### 2.4 常见 ISPC 编程陷阱

**陷阱 1：未定义行为（数据竞争）**

```cpp
// ISPC 代码
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
```

**问题**：多个迭代可能写入相同的内存位置（如 `y[i-1]`），输出是**未定义**的！

**陷阱 2：类型错误——对 varying 变量赋值给 uniform**

```cpp
export uniform float sum_incorrect_1(
   uniform int N,
   uniform float* x)
{
   float sum = 0.0f;       // sum 是 varying 类型（每个实例有自己的副本）
   foreach (i = 0 ... N)
   {
      sum += x[i];          // x[i] 对每个实例不同
   }
   return sum;              // 编译时类型错误！
}
```

**问题**：`sum` 是 `varying float`，不能返回给期望 `uniform float` 的调用者。

**陷阱 3：对 uniform 变量使用 varying 计算**

```cpp
export uniform float sum_incorrect_2(
   uniform int N,
   uniform float* x)
{
   uniform float sum = 0.0f;  // sum 是 uniform 类型
   foreach (i = 0 ... N)
   {
      sum += x[i];              // x[i] 是 varying，赋值给 uniform 是编译错误
   }
   return sum;
}
```

**正确的数组求和实现：**

```cpp
export uniform float sum_array(
   uniform int N,
   uniform float* x)
{
   uniform float sum;
   float partial = 0.0f;          // 每个实例私有累积部分和
   foreach (i = 0 ... N)
   {
      partial += x[i];
   }
   // reduce_add() 是 ISPC 标准库的跨实例通信原语
   sum = reduce_add(partial);
   return sum;
}
```

**工作原理**：
- 每个实例在私有的 `partial` 变量中累积部分和（无通信）
- 使用 `reduce_add()` 跨实例通信原语将所有部分和相加
- 结果为所有实例都能获得的 `uniform float` 总值

### 2.5 ISPC 跨程序实例操作

| 操作 | 原型 | 功能 |
|------|------|------|
| 求和归约 | `uniform int64 reduce_add(int32 x)` | 计算 gang 中所有实例变量值的和 |
| 最小值归约 | `uniform int32 reduce_min(int32 a)` | 计算 gang 中所有值的最小值 |
| 广播 | `int32 broadcast(int32 value, uniform int index)` | 将某个实例的值广播到所有实例 |
| 轮转/移位 | `int32 rotate(int32 value, uniform int offset)` | 将值从实例 i 传给实例 i+offset |

**高级示例：用 shift 在 8 个实例中计算乘积（仅需 lg(8)=3 步）**

```cpp
export void vec8product(
   uniform float* x,
   uniform float* result)
{
   float val1 = x[programIndex];
   float val2 = shift(val1, 1);

   if (programIndex % 2 == 0)
       val1 = val1 * val2;
   val2 = shift(val1, 2);
   if (programIndex % 4 == 0)
       val1 = val1 * val2;
   val2 = shift(val1, 4);
   if (programIndex % 8 == 0) {
       *result = val1 * val2;
   }
}
```

这是一种**蝶形归约（butterfly reduction）**模式，利用 shift 操作实现高效并行。

---

## 3. 抽象与实现

### 3.1 ISPC 的两层抽象

| 层面 | 描述 |
|------|------|
| **编程抽象（SPMD）** | 程序员"认为"自己正在运行多个逻辑指令流，每个有不同的 programIndex |
| **实现（SIMD）** | ISPC 编译器生成向量指令（AVX2/ARM NEON），将 gang 的语义映射到 SIMD 硬件 |

**关键教训：**
- ISPC 虽然提供高层次的 `foreach` 抽象，但通过暴露 `programIndex` 和 `programCount`，仍然允许程序员进行低级控制
- 这可能导致**未定义行为**（如数据竞争）或**仅对特定 programCount 正确的程序**
- 更强的抽象（如完全隐藏 programIndex，甚至不允许数组索引）可以使程序更安全、更可移植

### 3.2 更高级的数据并行抽象

**方案 1：隐藏 programIndex，只使用 foreach**
```cpp
export void ispc_function(int N, float* x, float* y)
{
   int twoN = 2 * N;
   foreach (i = 0 ... twoN)
   {
      float val = x[i];
      float result;
      // 对 val 进行运算得到 result
      y[i] = result;
   }
}
```
- 不需要考虑 program instance
- foreach 之外的所有值必须是 uniform

**方案 2：完全不使用数组索引——使用 map 操作**
```cpp
float dowork(float x) {
  // 对 x 进行运算，返回 result
}

Collection x;  // 包含 N 个元素
Collection y = map(doWork, x);
```
- 这与 NumPy、PyTorch 等框架的编程模型非常相似
- 程序员不写循环，不进行数组索引
- 对集合中每个元素调用函数

### 3.3 ISPC Tasks（多核扩展）

- `gang` 抽象由 SIMD 指令实现，仅在**单个 CPU 核心**上运行
- ISPC 还提供 **task** 抽象来利用**多核**执行
- task 被 ISPC 运行时分配到线程池中的工作线程（worker threads）上
- task 分配可以是动态的：每个 worker 线程在完成当前 task 后自动取下一个未完成的 task

```cpp
void foo(uniform float* input,
         uniform float* output,
         uniform int N)
{
  // 创建 100 个 task
  launch[100] my_ispc_task(input, output, N);
}
```

---

## 4. 创建并行程序的思维过程

### 4.1 并行程序的三个步骤

创建并行程序的核心思考过程：

1. **识别**可以并行执行的工作
2. **划分**工作（以及与之相关的数据）
3. **管理**数据访问、通信和同步

常见目标是**最大化加速比**（speedup）：

$$Speedup(P) = \frac{Time(1)}{Time(P)}$$

### 4.2 并行程序创建的四个阶段

```
问题
  |
  v
分解 (Decomposition) --> 子问题（"任务"）
  |
  v
分配 (Assignment) --> 并行线程（"工作者"）
  |
  v
编排 (Orchestration) --> 并行程序（通信线程）
  |
  v
映射 (Mapping) --> 在并行机器上执行
```

这些职责可能由程序员、系统（编译器、运行时、硬件）或两者共同承担。

---

## 5. 问题分解与 Amdahl 定律

### 5.1 问题分解

- 将问题拆分为可以**并行执行的任务**
- 一般原则：创建**足够多的任务**以充分利用机器上的所有执行单元
- **分解的核心挑战：识别依赖关系**（或者说，识别缺乏依赖关系的地方）

### 5.2 Amdahl 定律

设 S = 程序中**固有串行部分**的比例（依赖关系阻止了并行执行）。

则并行执行的最大加速比：**Speedup <= 1/S**

**直观理解：**

| 串行比例 S | 理论上限加速比 |
|-----------|---------------|
| 0.01 (1%) | <= 100x |
| 0.05 (5%) | <= 20x |
| 0.1 (10%) | <= 10x |

**重要警告：** 即使是很小的串行区域，在大型并行机器上也会严重限制加速比。

例如：Summit 超级计算机有 148,635,648 个 ALU，但如果 0.1% 的应用是串行的，最大加速比也仅为 1000x。

### 5.3 分解的责任归属

- **绝大多数情况下**：由程序员负责将程序分解为独立任务
- **自动分解**（编译器自动并行化串行程序）仍是一个具有挑战性的研究问题：
  - 编译器需要分析程序、识别依赖关系
  - 如果依赖关系是**数据相关的**（编译时不可知）怎么办？
  - 目前只在简单循环嵌套上取得了有限的成功
  - 针对复杂通用代码的"魔法并行编译器"**尚未实现**

### 5.4 Amdahl 定律示例

考虑一个 N x N 图像上的两步计算：
1. 步骤 1：将所有像素亮度乘以 2（每个像素独立计算）
2. 步骤 2：计算所有像素值的平均值

**首次并行化尝试（P 个处理器）：**
- 步骤 1：并行执行，时间 = N^2/P
- 步骤 2：串行执行，时间 = N^2
- 总加速比 <= 2（不论 P 多大！）

**改进方案——并行化步骤 2：**
- 步骤 2 改进：并行计算部分和，然后串行合并结果
- 步骤 2 时间 = N^2/P + P
- 当 N >> P 时，加速比趋近于 P

---

## 6. 任务分配（Assignment）

### 6.1 分配的核心问题

- 将任务分配给工作者（线程、程序实例、向量通道等）
- 目标：实现**良好的负载均衡**，减少**通信成本**
- 可以是**静态分配**（运行前确定）或**动态分配**（运行时决定）

### 6.2 静态分配示例：C++11 线程

```cpp
void my_thread_start(int N, int terms, float* x, float* results) {
  sinx(N, terms, x, result); // 执行工作
}

void parallel_sinx(int N, int terms, float* x, float* result) {
    int half = N/2;

    // 启动线程处理前半部分数组
    std::thread t1(my_thread_start, half, terms, x, result);
    // 主线程处理后半部分数组
    sinx(N - half, terms, x + half, result + half);
    t1.join();
}
```

这个例子中：
- 分解：按循环迭代划分
- 分配：程序员手动将工作分配给 C++ 线程（块状分配）

### 6.3 ISPC 中的分配方式对比

**程序员管理的分配（静态交错）：**
```cpp
export void ispc_sinx_interleaved(...)
{
   // 手动实现交错分配
   for (uniform int i=0; i<N; i+=programCount)
   {
       int idx = i + programIndex;
       // ...
   }
}
```

**系统管理的分配（foreach）：**
```cpp
export void ispc_sinx_foreach(...)
{
   // foreach 将分配责任交给系统
   foreach (i = 0 ... N)
   {
       float value = x[i];
       // ...
   }
}
```

foreach 暴露独立工作给系统，系统负责将迭代（工作）分配给 ISPC 程序实例。抽象留下了动态分配的空间（不过当前 ISPC 实现仍使用静态方案）。

---

## 7. 编排（Orchestration）与同步

### 7.1 编排涉及的内容

- 结构化通信
- 添加同步以保护依赖关系
- 在内存中组织数据结构
- 调度任务

目标：减少通信/同步成本，保持数据引用的局部性，减少开销。

### 7.2 共享地址空间模型

**核心思想：** 线程通过**读写共享地址空间中的变量**来进行通信。

```cpp
// 共享变量
int x = 0;

// 线程 1
void foo(int* x) {
    x = 1;  // 写入共享变量
}

// 线程 2
void bar(int* x) {
    while (x == 0) {}  // 等待线程 1 的写入
    print x;
}
```

**比喻：** 共享地址空间就像一块**公告板（bulletin board）**，每个人都可以读和写。

### 7.3 同步原语

#### 锁（Lock）——互斥访问

```cpp
// 共享变量
int x = 0;
Lock my_lock;

// 线程 1
my_lock.lock();
x++;               // 临界区
my_lock.unlock();
print(x);

// 线程 2
my_lock.lock();
x++;               // 临界区
my_lock.unlock();
print(x);
```

**为什么需要互斥？**

在不加锁时，x++ 操作由三条指令组成：
1. `r1 <- x`（从内存加载 x 到寄存器）
2. `r1 <- r1 + r2`（加法）
3. `x <- r1`（将结果存回内存）

可能出现的时间交错（假设初始 x=0, r2=1）：

```
T1: r1 <- x         (T1 读到 0)
T2: r1 <- x         (T2 读到 0)
T1: r1 <- r1 + r2   (T1 的 r1 = 1)
T2: r1 <- r1 + r2   (T2 的 r1 = 1)
T1: x <- r1         (T1 写入 1)
T2: x <- r1         (T2 也写入 1！)
```

结果：两个线程各执行一次 x++，但 x 从 0 变成了 1（丢失了一次更新）！

**保持原子性的机制：**
- 互斥锁/解锁（Lock/Unlock mutex）
- 硬件支持的原子读-改-写操作（如 `atomicAdd`）
- 语言的原子块支持（如 `atomic { ... }`）

#### 屏障（Barrier）——阶段同步

```cpp
barrier(num_threads);
```

- 屏障是表达依赖关系的**保守方式**
- 屏障将计算划分为**阶段**
- **所有线程在屏障之前的所有计算**都必须在**任何线程在屏障之后的任何计算开始之前**完成
- 即：屏障之后的所有计算被认为依赖于屏障之前的所有计算

### 7.4 映射到硬件

将"线程"（工作者）分配给硬件执行单元的方式：

| 方式 | 示例 |
|------|------|
| 操作系统分配 | 将线程映射到 CPU 核心的硬件执行上下文 |
| 编译器分配 | 将 ISPC 程序实例映射到向量指令通道 |
| 硬件分配 | 将 CUDA 线程块映射到 GPU 核心 |

设计决策：
- 将**相关线程**放在同一核心上（最大化局部性和数据共享，最小化通信/同步成本）
- 将**不相关线程**放在同一核心上（一个可能受带宽限制，另一个受计算限制，充分利用机器）

---

## 8. 案例研究：2D 网格求解器

### 8.1 问题描述

在 (N+2) x (N+2) 的网格上求解偏微分方程（PDE），使用 Gauss-Seidel 迭代算法直到收敛：

```cpp
// 核心更新公式
A[i,j] = 0.2 * (A[i,j] + A[i,j-1] + A[i-1,j]
                      + A[i,j+1] + A[i+1,j]);
```

### 8.2 串行实现的伪代码

```cpp
const int n;
float* A;  // 假设已分配 (N+2) x (N+2) 的网格

void solve(float* A) {
  float diff, prev;
  bool done = false;

  while (!done) {                     // 最外层循环：迭代
    diff = 0.f;
    for (int i=1; i<n; i++) {         // 遍历非边界网格点
      for (int j=1; j<n; j++) {
        prev = A[i,j];
        A[i,j] = 0.2f * (A[i,j] + A[i,j-1] + A[i-1,j] +
                                  A[i,j+1] + A[i+1,j]);
        diff += fabs(A[i,j] - prev);  // 计算变化量
      }
    }
    if (diff/(n*n) < TOLERANCE)       // 收敛判断
      done = true;
  }
}
```

### 8.3 步骤 1：识别依赖关系

**Gauss-Seidel 原始算法的依赖关系：**
- 每一行的元素依赖于**左侧元素**（刚刚更新过的）
- 每一行依赖于**上一行**（刚刚更新过的）
- 沿对角线存在独立工作，但并行度不均匀，且需要频繁同步

**关键洞察：通过改变算法来获得更好的并行性！**

### 8.4 红黑着色（Red-Black Coloring）重排序

改变网格单元更新的顺序，使用类似国际象棋棋盘的**红黑着色方案**：

```
R B R B R ...
B R B R B ...
R B R B R ...
...
```

**执行策略：**
1. 并行更新所有**红色**单元格
2. 等待所有处理器完成红色更新
3. 并行更新所有**黑色**单元格（它们的数据依赖于红色邻居，而这些邻居已经更新完毕）
4. 重复直到收敛

**这种方法的特点：**
- 改变了收敛路径，但最终**收敛到相同解**（在误差阈值内）
- 浮点计算的具体值不同，但数学上等价
- 这是并行编程中的**常见技术**——为获得并行性而调整算法

### 8.5 不同分配方式下的通信成本

**块状分配 vs 交错分配：**

- **块状分配**（每个处理器负责一块连续的行）：边界通信数据较少
- **交错分配**（循环分配行给处理器）：每个处理器需要与更多邻居通信

选择取决于运行程序的系统架构。

### 8.6 数据并行表达

```cpp
const int n;
float* A = allocate(n+2, n+2);  // 分配网格

void solve(float* A) {
   bool done = false;
   float diff = 0.f;

   while (!done) {
     for_all (red cells (i,j)) {
         float prev = A[i,j];
         A[i,j] = 0.2f * (A[i-1,j] + A[i,j-1] + A[i,j] +
                          A[i+1,j] + A[i,j+1]);
         reduceAdd(diff, abs(A[i,j] - prev));
     }

     if (diff/(n*n) < TOLERANCE)
       done = true;
   }
}
```

- **分解**：每个网格元素的处理是独立工作
- **分配**：由系统处理
- **编排**：由系统处理（for_all 块末尾有隐式 barrier；reduceAdd 是内置通信原语）

### 8.7 共享地址空间表达（SPMD 线程）

```cpp
int     n;               // 网格大小
bool    done = false;
float   diff = 0.0;
LOCK    myLock;
BARRIER myBarrier;
float*  A = allocate(n+2, n+2);

void solve(float* A) {
   float myDiff;
   int threadId = getThreadId();
   int myMin = 1 + (threadId * n / NUM_PROCESSORS);
   int myMax = myMin + (n / NUM_PROCESSORS)

   while (!done) {
     float myDiff = 0.f;
     diff = 0.f;
     barrier(myBarrier, NUM_PROCESSORS);

     for (j=myMin to myMax) {
        for (i = red cells in this row) {
           float prev = A[i,j];
           A[i,j] = 0.2f * (A[i-1,j] + A[i,j-1] + A[i,j] +
                            A[i+1,j] + A[i,j+1]);
           myDiff += abs(A[i,j] - prev);
       }
     }

     // 性能优化：先累积本地部分差，最后一次性归约
     lock(myLock);
     diff += myDiff;
     unlock(myLock);

     barrier(myBarrier, NUM_PROCESSORS);

     if (diff/(n*n) < TOLERANCE)
         done = true;

     barrier(myBarrier, NUM_PROCESSORS);
   }
}
```

**代码中的关键设计点：**

1. **threadId** 每个 SPMD 实例不同，用于计算各自负责的行范围
2. **三个 barrier** 的作用：
   - 第一个 barrier：确保所有线程重置 diff 后，再开始新一轮计算
   - 第二个 barrier：确保所有线程完成 diff 累加后，再检查收敛条件
   - 第三个 barrier：确保所有线程读取到相同的 done 值后，再进入下一轮迭代（或退出）
3. **锁的作用**：保护对全局 `diff` 变量的互斥更新

### 8.8 性能优化：减少锁竞争

**初始版本（每次循环都加锁——性能差）：**
```cpp
for (j=myMin to myMax) {
    for (i = red cells in this row) {
        // ...
        LOCK(myLock);
        diff += abs(A[i,j] - prev);   // 每次循环都加锁
        UNLOCK(myLock);
    }
}
```

**优化版本（先累积本地值，最后一次性加锁）：**
```cpp
for (j=myMin to myMax) {
    for (i = red cells in this row) {
        // ...
        myDiff += abs(A[i,j] - prev);  // 先累积到线程局部变量
    }
}
lock(myLock);
diff += myDiff;   // 每个线程只加锁一次
unlock(myLock);
```

这是一个**将锁从内循环移到外循环**的常见优化模式——先计算本地部分和，再一次性全局归约。

### 8.9 高级优化：消除 Barrier 依赖

通过**以空间换时间**——使用多个 diff 变量解除迭代间的依赖：

```cpp
float diff[3];  // 使用 3 个副本代替 1 个
float *A = allocate(n+2, n+2);

void solve(float* A) {
  float myDiff;     // 线程局部变量
  int index = 0;    // 线程局部变量
  diff[0] = 0.0f;

  barrier(myBarrier, NUM_PROCESSORS);  // 仅一次性初始化 barrier

  while (!done) {
    myDiff = 0.0f;
    // 执行计算（累积到本地 myDiff）

    lock(myLock);
    diff[index] += myDiff;             // 原子更新全局 diff
    unlock(myLock);

    diff[(index+1) % 3] = 0.0f;       // 清除旧值
    barrier(myBarrier, NUM_PROCESSORS); // 仅需一个 barrier

    if (diff[index]/(n*n) < TOLERANCE)
      break;

    index = (index + 1) % 3;
  }
}
```

**技巧**：使用不同的 diff 变量给连续的循环迭代，消除数据依赖，减少需要的同步点。

---

## 9. 数据并行模型 vs 共享地址空间模型

### 9.1 两种模型的对比

| 特性 | 数据并行模型 | 共享地址空间模型 (SPMD) |
|------|-------------|----------------------|
| **同步** | 单一逻辑控制流；for_all 循环体末尾有隐式 barrier | 程序员负责同步（显式 lock/barrier） |
| **通信** | 通过内存读写隐式完成；内置 reduce 等复杂通信原语 | 通过共享变量的读写隐式完成 |
| **典型使用方式** | for_all, foreach, map 等高级抽象 | 共用同一地址空间的多个线程，每个执行相同程序 |
| **编程难度** | 相对简单，更接近顺序编程思维 | 需要显式管理同步和通信 |
| **灵活性** | 受限于语言/框架提供的抽象 | 更灵活，可精细控制 |

### 9.2 模型选择指南

- **数据并行**适合：规则的数据处理（如数组/图像处理）、每个元素的独立计算
- **共享地址空间**适合：需要精细控制通信和同步模式的场景、不规则的数据结构

---

## 10. 总结

本讲的核心要点：

1. **SPMD 编程模型**：单程序多数据——定义一个函数，在不同数据上并行运行多个实例。ISPC 将 SPMD 抽象映射到 SIMD 硬件实现。

2. **抽象层次**：
   - 低级：手动使用 `programIndex` 和 `programCount`（精细控制但容易出错）
   - 中级：使用 `foreach`（让系统管理分配）
   - 高级：使用 `map` 操作（完全不写循环，如 NumPy/PyTorch 风格）

3. **Amdahl 定律**：并行加速比受限于程序的串行部分。即使是 0.1% 的串行代码，在百万级处理器上也会将加速比限制在 1000 倍以内。

4. **创建并行程序的四个阶段**：
   - **分解**（Decomposition）：识别可并行的工作（核心挑战：找到缺乏依赖的地方）
   - **分配**（Assignment）：将任务分给工作者（静态或动态）
   - **编排**（Orchestration）：管理通信、同步、数据布局
   - **映射**（Mapping）：将线程映射到硬件资源

5. **关键同步原语**：
   - **锁（Lock）**：实现互斥，保护临界区
   - **屏障（Barrier）**：划分计算阶段，表达整体依赖关系
   - **原子操作**：硬件支持的不可分割读-改-写

6. **性能优化模式**：
   - 将锁从内循环移到外循环（先累积本地结果，再一次性全局归约）
   - 以空间换时间消除同步依赖（如使用多个变量副本）

7. **算法可以为了并行性而改变**：如红黑着色重排序使 Gauss-Seidel 的网格更新可以并行执行。收敛路径不同，但数学上等价——这是并行编程中的常见且合法的手段。
