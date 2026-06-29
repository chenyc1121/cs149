# Lecture 8: 数据并行编程模型

**课程**: Stanford CS149, Fall 2025 — Parallel Computing

---

## 1. 课程概述：数据并行的思维方式

### 核心理念转变

- 之前的课程：从"每个worker做什么工作"以及"如何分配工作给worker"的角度思考并行编程
- 本讲的核心：以**对数据序列（sequences）进行操作**的方式来描述算法
- 主要操作原语（primitives）：
  - `map`（映射）
  - `filter`（过滤）
  - `fold` / `reduce`（折叠/规约）
  - `scan` / `segmented scan`（扫描/分段扫描）
  - `sort`（排序）
  - `groupBy`（分组）
  - `join`（连接）
  - `partition` / `flatten`（分区/展平）

### 核心思想

这些操作原语存在**高性能的并行实现**，因此基于这些原语编写的程序通常可以在并行机器上高效运行——前提是能避免带宽瓶颈（bandwidth bound）。

---

## 2. 动机：为什么需要暴露大量并行性？

### 利用大量计算核心

- 高核心数量的机器（high-core count machines）
- 云环境中的多机集群
- SIMD 处理 + 多线程核心需要更多并行性
- **GPU 架构需要极其大量的并行性**

### 回顾：NVIDIA V100 GPU 的几何结构

- 时钟频率：1.245 GHz
- 每芯片 **80 个 SM 核心**
- **80 x 4 x 16 = 5,120 个 fp32 乘加 ALU**
- 算力 = **12.7 TFLOPS**（乘加计为 2 flops）
- 每芯片最多 80 x 64 = **5,120 个交错 warp**
- 最多 **163,840 个 CUDA 线程/芯片**
- L2 Cache: 6 MB
- GPU 内存: 16 GB, **900 GB/sec**

> **关键结论**：如果程序没有暴露大量并行性，且没有高算术强度（arithmetic intensity），在 GPU 上运行将不会高效。

---

## 3. 依赖关系是并行编程的关键

### 理解依赖关系

- 并行编程的关键部分是**理解操作之间何时存在依赖关系**
- **没有依赖关系 = 潜在的并行执行机会**

```c
x = a + b;
y = b * 7;
z = (x - y) * (x + y);
```

这段代码中的数据依赖关系通过数据流图（data flow graph）来表示，图中箭头表示值流动的方向。操作之间没有直接依赖关系的部分可以并行执行。

---

## 4. 数据并行模型

### 定义

将计算组织为**对元素序列的操作**（例如，对序列中的所有元素执行相同的函数）。

### 典型示例：NumPy

```python
C = A + B   # A, B, C 是长度相同的向量
```

这一操作表达了逐元素的并行计算，与顺序执行无关。

---

## 5. 核心数据类型：序列（Sequences）

### 定义

- 元素的有序集合
- 不同语言/框架中的表现形式：
  - C++: `Sequence<T>`
  - Scala: `List[T]`
  - Python: Pandas DataFrames
  - PyTorch / JAX: Tensors（N维序列）
  - Haskell: `seq T`

### 重要区分：序列 vs 数组

> **与数组不同，程序只能通过特定的操作来访问序列的元素，而不能直接随机访问元素（random access）**。这种抽象使得底层实现可以自由地进行重排序和并行化。

---

## 6. Map（映射）

### 定义

Map 是一个**高阶函数**（higher order function），即接受一个函数作为参数的函数。

- 将一个**无副作用**的单元函数 `f :: a -> b` 应用于输入序列的所有元素
- 生成一个长度相同的输出序列

```haskell
-- Haskell 类型签名
map :: (a -> b) -> seq a -> seq b
```

### C++ 实现

```cpp
// C++ 中的等价物：std::transform
template<class InputIt, class OutputIt, class UnaryOperation>
OutputIt transform(InputIt first1, InputIt last1, OutputIt d_first,
                   UnaryOperation unary_op);

// 使用示例
int f(int x) { return x + 10; }
int a[] = {3, 8, 4, 6, 3, 9, 2, 8};
int b[8];
std::transform(a, a+8, b, f);
// b = [13, 18, 14, 16, 13, 19, 12, 18]
```

- JAX 中的等价物：`vmap`

### Map 的并行化

由于 `f :: a -> b` 是一个**纯函数**（无副作用），对序列中所有元素应用 `f` 可以**以任意顺序执行**而不改变程序的输出。因此，`map` 的实现可以自由地进行重排序和并行化：

```
parallel_map(f, s):
    partition sequence s into P smaller sequences
    for each subsequence s_i (in parallel):
        out_i = map f s_i
    out = concatenate out_i's
```

实现策略：将序列分成 P 个子序列，并行处理每个子序列，然后将结果拼接。

---

## 7. Fold（折叠 / Reduce规约）

### Fold Left（左折叠）

将一个二元操作 `f` 应用于每个元素和一个累积值，由一个类型为 `b` 的初始值作为种子：

```
f :: (b, a) -> b
fold :: b -> ((b, a) -> b) -> seq a -> b
```

**Scala 示例**：

```scala
def foldLeft[A, B](init: B, f: (B, A) => B, l: List[A]): B
```

**图解**：输入序列 `[3, 8, 4, 6, 3, 9, 2, 8]`，初始值 `10`，操作 `+`：

```
fold_left 10 + [3, 8, 4, 6, 3, 9, 2, 8]
= ((((((((10 + 3) + 8) + 4) + 6) + 3) + 9) + 2) + 8)
= 53
```

### 并行 Fold

并行 fold 除了需要二元函数 `f` 之外，还需要一个额外的二元**组合函数（combiner function）** `*`：

```
f   :: (b, a) -> b
comb :: (b, b) -> b
fold_par :: b -> ((b, a) -> b) -> ((b, b) -> b) -> seq a -> b
```

- 初始种子值必须是 `f` 和 `comb` 的**单位元（identity element）**
- 如果 `f :: (b, b) -> b` 本身已经是**结合性（associative）的二元操作符**，则不需要单独的 combiner（此时可以自然地在树形结构中并行归约）

**并行化策略**：
1. 将序列分成多个子序列
2. 各自独立进行fold，得到部分结果（使用 identity 初始化）
3. 使用 combiner 将部分结果合并成最终结果

---

## 8. Scan（扫描 / 前缀和）

### 定义

```
f :: (a, a) -> a     (结合性二元操作符)
scan :: a -> ((a, a) -> a) -> seq a -> seq a
```

### 两种形式

- **Inclusive Scan（包含扫描）**: `out[i]` 是 `in[0]` 到 `in[i]` 的扫描结果
- **Exclusive Scan（排除扫描）**: `out[i]` 是 `in[0]` 到 `in[i-1]` 的扫描结果（即排除 `in[i]` 自身）

### 顺序实现

```cpp
float op(float a, float b) { /* ... */ }

void scan_inclusive(float* in, float* out, int N) {
    out[0] = in[0];
    for (int i = 1; i < N; i++)
        out[i] = op(out[i-1], in[i]);
}
```

**示例**：输入 `[3, 8, 4, 6, 3, 9, 2, 8]`，操作 `+`：
- inclusive scan: `[3, 11, 15, 21, 24, 33, 35, 43]`

### 数据并行 Scan 的数学表示

设 `A = [a0, a1, a2, a3, ..., an-1]`，`⊕` 是以 `I` 为单位元的结合性二元操作符：

- `scan_inclusive(⊕, A) = [a0, a0⊕a1, a0⊕a1⊕a2, ...]`
- `scan_exclusive(⊕, A) = [I, a0, a0⊕a1, ...]`

当操作为 `+` 时，`scan_inclusive(+, A)` 称为**前缀和（prefix sum）**：
- `prefix_sum(A) = [a0, a0+a1, a0+a1+a2, ...]`

---

## 9. 数据并行 Inclusive Scan 算法（O(N log N) work）

### 算法描述

该算法为一种**归约树（reduction tree）**方法：

1. 每一层步长加倍（1, 2, 4, 8, ...）
2. 每个元素累加模式逐渐扩大

**复杂度**：
- **Work（总操作量）**: O(N lg N) —— 比顺序算法 O(N) 更"低效"
- **Span（最长顺序依赖链）**: O(lg N) —— 但在并行环境下更优

算法通过多个阶段，每阶段有更多的并行性。使用 `ai-j` 表示 `a_i ⊕ ... ⊕ a_j` 的结果。

> **注意**：可以通过减去原始向量得到 exclusive scan 的结果。

---

## 10. 工作高效的并行 Exclusive Scan（O(N) work）

### 两阶段算法

**Phase 1: Up-sweep（上扫/归约阶段）**

```
for d = 0 to (log2(n) - 1):
    forall k = 0 to n-1 step 2^(d+1):
        a[k + 2^(d+1) - 1] = a[k + 2^d - 1] + a[k + 2^(d+1) - 1]
```

这一阶段在每一轮中构建更大的部分和，类似一棵向上传播的归约树。

**Phase 2: Down-sweep（下扫/分发阶段）**

```
x[n-1] = 0
for d = (log2(n) - 1) down to 0:
    forall k = 0 to n-1 step 2^(d+1):
        tmp = a[k + 2^d - 1]
        a[k + 2^d - 1] = a[k + 2^(d+1) - 1]
        a[k + 2^(d+1) - 1] = tmp + a[k + 2^(d+1) - 1]
```

**复杂度**：
- **Work**: O(N)
- **Span**: O(lg N)
- **局部性（Locality）**: 取决于实现（需要考虑常数因子）

---

## 11. 双核处理器上的 Scan 实现

### 分治策略

将序列分成两半：

1. **P1**: 对元素 `[0-7]` 进行顺序 scan
2. **P2**: 对元素 `[8-15]` 进行顺序 scan
3. 计算基准值 `base = a[0-7]`（即 P1 的最后一个部分和）
4. **P2**: 将 `base` 加到自己的部分结果上

**分析**：
- **Work**: O(N)，常数因子仅为 **1.5**（相比顺序算法）
- **数据访问**：
  - 非常高的**空间局部性**（连续内存访问）
  - 在 NUMA 系统上，P1 对 `a[8]` 到 `a[11]` 的访问可能开销更大
  - 但在小规模多核系统上，访问成本与 P2 本地访问相同

---

## 12. Exclusive Scan：SIMD 实现（CUDA Warp级别）

### CUDA warp-level scan

以下代码在 32-wide SIMD 执行单元上实现 exclusive scan。每个 CUDA 线程对数组 idx 处的元素返回其 exclusive scan 结果：

```cuda
__device__ int scan_warp(int *ptr, const unsigned int idx)
{
    const unsigned int lane = idx % 32; // 线程在warp中的索引 (0..31)

    for (int i = 0; i < 5; i++) {       // 5 步因为 2^5 = 32
        int shift = 1 << i;
        if (lane >= shift) {
            int tmp1 = ptr[idx - shift];
            int tmp2 = ptr[idx];
            __syncwarp();               // 同步warp内线程
            ptr[idx] = tmp1 + tmp2;
            __syncwarp();
        }
    }
    return (lane > 0) ? ptr[idx-1] : 0;
}
```

**关键设计决策**：
- 使用 **O(N lg N) work 的算法**（非 work-efficient 算法）
- **原因**：work-efficient 算法在 SIMD 上下文中反而需要多于 2 倍的指令数，且导致 **SIMD 利用率（SIMD utilization）低**
- 在这个层面，简单算法指令更少，更适合 SIMD 并行执行

---

## 13. 构建更大数组的 Scan

### 分层 Scan 策略

以 128 元素、4 个 warp 的线程块为例：

1. **第一层**：每个 warp 对自己的 32 个元素执行 SIMD scan
   - Warp 0: `a[0-31]`
   - Warp 1: `a[32-63]`
   - Warp 2: `a[64-95]`
   - Warp 3: `a[96-127]`

2. **第二层**：收集每个 warp 的部分和，构成一个 4 元素数组
   - 在 warp 0 上对 4 个部分和执行 SIMD scan → 得到基底值 `base[0..3]`

3. **第三层**：将基底值加回每个 warp
   - Warp 1: 加 `base[0]` 到所有元素
   - Warp 2: 加 `base[1]` 到所有元素
   - Warp 3: 加 `base[2]` 到所有元素

---

## 14. 多线程 SIMD CUDA 实现（Block级别 Scan）

```cuda
__device__ void scan_block(int* ptr, const unsigned int idx)
{
    const unsigned int lane = idx % 32;         // warp内线程索引
    const unsigned int warp_id = idx >> 5;      // block内warp索引

    int val = scan_warp(ptr, idx);              // Step 1: 每个warp内部分scan
                                                 // (所有线程执行，同warp线程
                                                 //  通过共享内存 'ptr' 通信)

    if (lane == 31) ptr[warp_id] = ptr[idx];    // Step 2: 每个warp的线程31
    __syncthreads();                            // 将部分scan结果写入block共享内存

    if (warp_id == 0) scan_warp(ptr, idx);      // Step 3: warp 0 对基底值scan
    __syncthreads();

    if (warp_id > 0)                            // Step 4: 将基底值加到所有元素
        val = val + ptr[warp_id-1];

    __syncthreads();
    ptr[idx] = val;
}
```

---

## 15. 扩展到百万元素级别的 Scan

### 多 Kernel Launch 策略

处理 100 万元素（1024 元素/block）：

1. **Kernel Launch 1**：每个 block 独立执行内部 scan
2. **Kernel Launch 2**：收集所有 block 的部分和，进行一次全局 scan
3. **Kernel Launch 3**：将全局基底值加回各 block 的元素

超过 100 万元素时，第二阶段（全局 scan）需要进一步分拆到多个 block 中。这体现了多级层次化（multi-level hierarchy）的设计思路。

---

## 16. Scan 实现的设计原则总结

### 并行性（Parallelism）
- Scan 算法有 O(N) 的并行 work
- 高效的实现只需利用**刚好足够填满机器执行资源**的并行度
- 目标：**减少 work 和减少通信/同步开销**

### 局部性（Locality）
- **多级实现**以匹配内存层次结构
- CUDA 示例：per-block 实现在本地共享内存中进行

### 异构性（Heterogeneity）
- 在机器的**不同层级使用不同的算法策略**
- CUDA 示例：
  - **intra-warp scan**：使用简单（O(N lg N)）但 SIMD 友好的算法
  - **inter-thread scan**：使用更高效的算法
- 低核心数 CPU 示例：主要基于顺序 scan

---

## 17. 并行分段扫描（Parallel Segmented Scan）

### 问题背景：不规则数据结构的处理

常见场景：对"序列的序列"进行操作：

- 对于图中的每个顶点 v
  - 对于连接到 v 的每条边 e
- 对于模拟中的每个粒子 p
  - 对于距离 p 在 D 内的每个粒子
- 对于文档集合中的每个文档 d
  - 对于 d 中的每个词

**挑战**：存在两级并行性可以利用，但问题是**不规则的（irregular）**——边的列表、粒子邻居列表、每个文档的单词数在不同顶点/粒子/文档之间差异很大。

### Segmented Scan 的定义

分段扫描是 scan 的**推广**：同时对输入序列的**连续分区（contiguous partitions）**执行扫描。

**示例**：

```
设 A = [[1,2], [6], [1,2,3,4]]
设 ⊕ = +
segmented_scan_exclusive(⊕, A) = [[0,1], [0], [0,1,3,6]]
```

### 表示方式：Start-Flag 表示法

用标记数组（flag array）来表示嵌套序列的边界：

```
嵌套序列 A = [[1,2,3], [4,5,6,7,8]]

flag:  1  0  0  1  0  0  0  0
data:  1  2  3  4  5  6  7  8
```

- `flag[i] = 1` 表示 data[i] 是一个新段的开始
- `flag[i] = 0` 表示 data[i] 属于当前段

---

## 18. 工作高效的分段扫描算法

### Up-sweep（上扫）

```
for d = 0 to (log2(n) - 1):
    forall k = 0 to n-1 step 2^(d+1):
        if flag[k + 2^(d+1) - 1] == 0:
            data[k + 2^(d+1) - 1] = data[k + 2^d - 1] + data[k + 2^(d+1) - 1]
        flag[k + 2^(d+1) - 1] = flag[k + 2^d - 1] || flag[k + 2^(d+1) - 1]
```

关键区别：只有当 flag 为 0（即两个元素在同一段内）时才进行累加；同时用逻辑或传播 flag 信息。

### Down-sweep（下扫）

```
data[n-1] = 0
for d = (log2(n) - 1) down to 0:
    forall k = 0 to n-1 step 2^(d+1):
        tmp = data[k + 2^d - 1]
        data[k + 2^d - 1] = data[k + 2^(d+1) - 1]
        if flag_original[k + 2^d] == 1:
            data[k + 2^(d+1) - 1] = 0              // 新段开始
        else if flag[k + 2^d - 1] == 1:
            data[k + 2^(d+1) - 1] = tmp
        else:
            data[k + 2^(d+1) - 1] = tmp + data[k + 2^(d+1) - 1]
        flag[k + 2^d - 1] = 0
```

关键区别：需要**保留原始 flag 的副本**（`flag_original`），在 down-sweep 阶段用来判断段边界，在段开始处插入 0。

---

## 19. Scan / Segmented Scan 总结

### Scan
- 理论：问题的**并行性**与元素数量成**线性关系**
- 实践：
  - 利用局部性
  - 仅使用刚好足以填满机器执行资源的并行度
  - 在机器的不同层级使用不同策略的绝佳案例

### Segmented Scan
- 以**规则的、数据并行的方式**表达和操作**不规则数据结构**（如 list of lists）

---

## 20. Gather 和 Scatter：关键数据并行操作

### Gather（收集）

```
gather(index, input, output):
    output[i] = input[index[i]]
```

- 根据索引序列从数据序列中收集数据
- `output[i]` 从 `input` 的第 `index[i]` 个位置获取数据

**图例**：根据 `index_seq = [3, 12, 4, 9, 9, 15, 13, 0]` 从数据序列中 gather 对应的值。

### Scatter（散布）

```
scatter(index, input, output):
    output[index[i]] = input[i]
```

- 根据索引序列将输入数据散射到输出序列的指定位置
- **注意**：如果多个元素写入同一个索引位置，需要处理冲突

---

## 21. Gather/Scatter 硬件指令支持

### 指令级支持

- **AVX2 (2013)**: 支持 SIMD `gather` 指令
  - `gather(R1, R0, mem_base)` —— 从内存 `mem_base` 中根据索引向量 `R0` gather 数据到结果向量 `R1`
- **AVX2 不支持 SIMD scatter**：必须用标量循环来实现
- **AVX512**: 存在 `scatter` 指令
- **GPU**: 硬件支持 gather/scatter（但仍是比连续向量加载/存储更昂贵的操作）

> **性能考量**：Gather/Scatter 仍然比顺序向量的 load/store 更昂贵，因为有随机内存访问的开销。

---

## 22. 应用案例：稀疏矩阵乘法（Sparse Matrix Multiplication）

### 问题描述

稀疏矩阵-向量乘：`y = A * x`，其中矩阵 A 中大部分值为零。

- 简单并行化：按行并行计算点积
- 但**每行工作量不同**（不规则的），这使得宽 SIMD 执行变得复杂

### 稀疏存储格式：CSR（Compressed Sparse Row）

```
values     = [[3, 1], [2], [4], ..., [2, 6, 8]]
cols       = [[0, 2], [1], [2], ...]
row_starts = [0, 2, 3, 4, ...]
```

### 使用 Scan 的并行方法

**示例数据**：
```
x = [x0, x1, x2, x3]
values = [[3,1], [2], [4], [2,6,8]]
cols   = [[0,2], [1], [2], [1,2,3]]
row_starts = [0, 2, 3, 4]
```

**步骤 1：Gather**——根据 cols 从 x 中收集数据：
```
gathered = [x0, x2, x1, x2, x1, x2, x3]
```

**步骤 2：Map**——对每个非零值执行乘法：
```
products = [3*x0, x2, 2*x1, 4*x2, 2*x1, 6*x2, 8*x3]
```

**步骤 3：创建 flags 向量**（从 row_starts）：
```
flags = [1, 0, 1, 1, 1, 0, 0]
```

**步骤 4：Inclusive Segmented Scan**——对 (products, flags) 使用加法：
```
[3x0, 3x0+x2, 2x1, 4x2, 2x1, 2x1+6x2, 2x1+6x2+8x3]
```

**步骤 5：取每段最后一个元素**：
```
y = [3x0+x2, 2x1, 4x2, 2x1+6x2+8x3]
```

这展示了如何将 gather、map、segmented scan 组合使用来解决实际的稀疏计算问题。

---

## 23. 将 Scatter 转化为 Sort/Gather

### 特殊情况的优化

当索引元素是唯一的且引用了所有索引位置时（即 scatter 是一个排列），可以将 scatter 转化为排序：

```
scatter(index, input, output):
    output = sort input sequence by values in index sequence
```

**示例**：
```
index: [0, 2, 1, 4, 3, 6, 7, 5]
input: [3, 8, 4, 6, 3, 9, 2, 8]
input (sorted by index): [3, 4, 8, 3, 6, 8, 9, 2]
```

### 实现带有冲突解决操作的 scatter（scatterOp）

当索引元素**不唯一**时，需要同步来保证原子性：

**示例**：`atomicAdd(output[index[i]], input[i])`
```
index: [1, 1, 0, 2, 0, 0]
```

**步骤 1：Sort**——根据索引值对输入序列排序：
```
Sorted index: [0, 0, 0, 1, 1, 2]
Input sorted by index: [input[2], input[4], input[5], input[0], input[1], input[3]]
```

**步骤 2：计算每段起始位置**：
```
starts: [1, 0, 0, 1, 0, 1]
```

**步骤 3：Segmented Scan**（使用操作 'op'）：
```
[op(op(input[2], input[4]), input[5]), op(input[0], input[1]), input[3]]
```

---

## 24. 更多序列操作

### Group by key

```
Seq (key, T) -> Seq (key, Seq T)
```

根据 key 将元素分组，创建一个序列的序列，每个子序列包含相同 key 的元素。

### Filter

从序列中移除不满足谓词条件的元素。

**示例**：filter 出所有偶数值元素。

### Sort

对序列进行排序，可以根据需要保持键值关联。

这些操作的组合使程序员可以用声明式（declarative）的方式描述复杂的并行计算。

---

## 25. 设计案例：在 GPU 上构建粒子网格数据结构

### 问题描述

将 100 万个点粒子根据其 2D 位置放入一个 16 格均匀网格中，构建 2D 数组的列表数据结构。

- **GPU 背景**：V100 每个 SM 核心最多 2048 个 CUDA 线程，共 80 个 SM 核心

### 方案 1：按粒子并行化（粒度竞态问题）

```c
list cell_list[16];    // 2维列表数组
lock cell_list_lock;

for each particle p:   // 并行执行
    c = compute cell containing p
    lock(cell_list_lock)
    append p to cell_list[c]
    unlock(cell_list_lock)
```

**问题**：**大量竞争（massive contention）**——数千个线程争夺同一个全局锁。

### 方案 2：使用更细粒度的锁

```c
list cell_list[16];
lock cell_list_lock[16];   // 每个cell一个锁

for each particle p:  // 并行执行
    c = compute cell containing p
    lock(cell_list_lock[c])
    append p to cell_list[c]
    unlock(cell_list_lock[c])
```

**效果**：假设粒子均匀分布，竞争减少约 **16 倍**。但仍然无法完全消除竞争。

### 方案 3：按 cell 并行化

```c
list cell_lists[16];

for each cell c:          // 并行执行
    for each particle p:  // 顺序执行
        if (p is within c)
            append p to cell_lists[c]
```

**问题**：
- **并行度不足**：只有 16 个并行任务，而 GPU 需要数千个
- **工作不高效**：比顺序算法多做 16 倍的 particle-in-cell 计算

### 方案 4：计算部分结果 + 合并

- 生成 N 个"部分"网格（并行），然后合并
- 每个线程块的所有线程更新同一个网格
- 优势：
  - 竞争减少 N 倍
  - 同步成本更低（使用 block-local 变量/CUDA 共享内存）
- 代价：
  - 需要额外的工作来合并 N 个网格
  - 需要额外的内存存储 N 个网格（而非 1 个）

### 方案 5：数据并行方法

**步骤 1：Map**
```c
// 并行计算每个粒子所在的cell
// 输入: 粒子位置数组
// 输出: grid_cell[i] = 粒子i所在的cell编号
```

**步骤 2：Sort**
- 按 cell 编号对 `(grid_cell, particle_index)` 对进行排序
- 排序后，同一 cell 中的粒子在内存中连续排列

**步骤 3：找每个 cell 的起止位置**
```c
this_cell = grid_cell[thread_index];
prev_cell = grid_cell[thread_index-1];

if (thread_index == 0)              // 特殊处理第一个cell
    cell_starts[this_cell] = 0;
else if (this_cell != prev_cell) {  // cell边界
    cell_starts[this_cell] = thread_index;
    cell_ends[prev_cell] = thread_index;
}
if (thread_index == numParticles-1)  // 特殊处理最后一个cell
    cell_ends[this_cell] = thread_index + 1;
```

**优势**：
- 保持大量并行度
- 消除对细粒度同步的需求

**代价**：
- 需要一次排序和对数据的多次遍历（额外的内存带宽开销）

---

## 26. 设计案例：并行直方图（Parallel Histogram）

### 问题描述

给定一个值序列，计算直方图：

```c
int f(float value);            // 将数组值映射到直方图bin ID
float input[N];                 
int histogram_bins[NUM_BINS];  // bin初始化为0

for (int i = 0; i < N; i++) {
    histogram_bins[f(input[i])]++;
}
```

**挑战**：仅使用 `map()` 和 `sort()` 操作来实现大规模并行的直方图。

### 数据并行直方图实现

**Part 1：计算 bin ID 并排序**

```cuda
// Step 1: 每个线程计算自己元素的bin ID
void compute_bin(float* input, int* bin_ids) {
    bin_ids[thread_index] = f(input[thread_index]);
}

// 中间步骤: 按 bin_ids 排序
sort(N, bin_ids, sorted_bin_ids);

// Step 2: 找到每个bin在排序列表中的起始位置
void find_starts(int* bin_ids, int* bin_starts) {
    if (thread_index == 0 || bin_ids[thread_index] != bin_ids[thread_index-1])
        bin_starts[bin_ids[thread_index]] = thread_index;
}
```

**Part 2：计算每个 bin 的大小**

```cuda
void bin_sizes(int* bin_starts, int* histogram_bins,
               int num_items, int num_bins) {
    if (bin_starts[thread_index] == -1) {
        histogram_bins[thread_index] = 0;    // 该bin没有元素
    } else {
        // 找下一个非空bin来确定当前bin的大小
        int next_idx = thread_index + 1;
        while (next_idx < num_bins && bin_starts[next_idx] == -1)
            next_idx++;

        if (next_idx < num_bins)
            histogram_bins[thread_index] =
                bin_starts[next_idx] - bin_starts[thread_index];
        else
            histogram_bins[thread_index] =
                num_items - bin_starts[thread_index];
    }
}
```

**算法流程总结**：
1. Map：为每个元素计算 bin ID
2. Sort：按 bin ID 排序元素
3. Map：找到每个 bin 在排序数组中的起始位置
4. Map：计算每个 bin 的大小（通过相邻起始位置之差）

**关键点**：处理空 bin 需要向后搜索找到下一个非空 bin，这是一个需要注意的边界情况。

---

## 27. N-Body 问题中的网格应用

### 问题背景

一个常见操作是计算与邻近粒子的相互作用。例如：给定一个粒子，找到在半径 R 内的所有粒子。

**解决方案**：将粒子组织成网格（cell 大小为 R），只需要检查周围网格单元中的粒子。这避免了 O(N^2) 的全局搜索。

---

## 28. 课后总结

### 数据并行思维的核心原则

1. **用对大数据集合的简单操作来实现算法**
   - 这些操作通常是广泛可并行化的
   - 存在高效的并行实现

2. **将不规则的并行性转化为规则的并行性**
   - 例如：通过排序 + segmented scan 处理不规则数据结构

3. **将细粒度同步转化为粗粒度同步**
   - 例如：用 sort 代替 atomic scatter，用一次全局 barrier 代替多次细粒度锁

4. **折中（trade-off）**：大多数数据并行方案需要对数据进行多次遍历 —— **对内存带宽需求高**

### 实际系统中的数据并行原语

数据并行原语是当今许多并行/分布式系统的基础：

- **CUDA Thrust Library**：GPU 上的 STL-like 并行算法库
- **Pandas DataFrames**：Python 中的表格数据处理
- **JAX**：支持 vmap、pmap 等变换的数值计算库
- **Apache Spark / Hadoop**：大规模分布式数据处理

这些框架都提供了 `map`、`reduce`、`filter`、`groupBy`、`join` 等数据并行操作的实现，使得开发者可以写出高效并行的程序，而不需要显式管理并行性和同步。

---

## 关键概念回顾

| 操作 | 功能 | 并行策略 |
|------|------|---------|
| **Map** | 对每个元素应用函数 | 完全独立，embarrassingly parallel |
| **Fold/Reduce** | 将序列归约为单个值 | 树形归约，需要结合性操作 |
| **Scan** | 前缀/累计操作 | 分层：warp-level SIMD → block-level → multi-kernel |
| **Segmented Scan** | 对不规则段执行 scan | 带 flag 传播的扩展 scan 算法 |
| **Gather** | 按索引收集数据 | 独立读取（可能有竞争） |
| **Scatter** | 按索引散布数据 | 需要原子操作或 sort-based 方法 |
| **Sort** | 排序 | 排序网络或并行排序算法 |
| **GroupBy** | 按键分组 | Sort + 找段边界 |

---

**本讲的核心信息**：通过将计算抽象为序列上的高阶操作（map、reduce、scan等），我们可以利用这些操作的成熟并行实现，编写出在从多核 CPU 到 GPU 再到分布式集群上都能高效运行的并行程序。关键是要理解每种操作的并行特性、代价模型以及它们如何组合来解决实际问题。
