# Lecture 5: 并行程序性能优化 (一)

> Stanford CS149, Fall 2025
>
> 主题：工作分配与调度 (Work Distribution and Scheduling)

---

## 一、课程概览

本节课内容分为三大部分：

1. **回顾 Lecture 4**：SPMD 网格求解器应用
2. **基本负载均衡技术**：静态分配 vs 动态分配，任务粒度选择
3. **Cilk 调度系统深度剖析**：fork-join 并行、work stealing、dequeue 实现、sync 机制

---

## 二、高性能并行编程的核心矛盾

编写高性能并行程序是一个**不断迭代优化**的过程，需要在三个相互矛盾的维度之间权衡：

1. **负载均衡 (Workload Balance)**：将工作均匀分配到所有可用的执行资源上，避免某些核心空闲等待。
2. **减少通信 (Reduce Communication)**：尽量避免线程之间的同步等待（stall），减少数据传输。
3. **减少额外开销 (Reduce Overhead)**：为增加并行性、管理工作分配、减少通信而引入的"额外工作"。

**Tip #1**：始终先实现最简单的串行方案，然后测量性能，判断是否真的需要更复杂的优化。不要过早优化。

---

## 三、负载均衡 (Balancing the Workload)

### 3.1 核心理念

理想状态下，所有处理器在整个程序执行过程中持续计算——同时进行计算，且**同时完成**各自的工作。

### 3.2 负载不均的危害

即便是少量负载不均，也会显著限制最大加速比。

示例：P4 做的工作量是其他线程的 2 倍

- P4 耗时 2 倍于其他线程
- 意味着并行程序运行时间的约 50% 是串行执行的
- 在 Amdahl 定律中，串行部分 S ≈ 0.2，但导致的影响远超 20%

```
时间线:
P1 |████████| 完成
P2 |████████| 完成
P3 |████████| 完成
P4 |████████████████| 完成 (2x工作量)
     ← 这段区间只有P4在跑，等效为串行 →
```

---

## 四、静态工作分配 (Static Assignment)

### 4.1 定义

工作分配在线程运行前即已确定，不依赖于运行时的动态行为。

> 注意：即使分配依赖于运行时参数（如输入数据大小、线程数量），只要分配方案在"已知工作量和工人数的时刻"就能确定，我们仍称之为静态分配。

### 4.2 示例：Programming Assignment 1 (Program 1)

为每个线程分配相等数量的网格单元。学生可以探索不同的静态分配策略。

```
T0 | T1 | T2 | T3 | T0 | T1 | T2 | T3 | T0 | T1 | T2 | T3 | ...
```

### 4.3 优点

- **简单**：实现代价极低（仅需少量索引计算即可完成分配）
- **几乎零运行时开销**：分配决策在编译/启动时已完成

### 4.4 静态分配的适用场景

**场景一：工作量可预测，且所有任务代价相同**

```
有 12 个任务，每个代价相同
静态分配：4 个处理器各分 3 个任务

P1 |████████████| (3个任务)
P2 |████████████| (3个任务)
P3 |████████████| (3个任务)
P4 |████████████| (3个任务)
同时完成
```

**场景二：工作量可预测，但各任务代价不同**

当执行的统计特性可预测时（例如，平均执行时间已知），可以将随机分布的任务均匀分配给各处理器，使**统计平均意义上的负载**达到均衡。

```
各任务代价不同（但长/短任务是随机交错的）
按数量均分 → 平均负载大致均衡
P1 |████  ████    ████|
P2 |  ████████  ██    |
P3 |██  ██  ████████  |
P4 |  ██████    ██████|
```

### 4.5 为什么 Programming Assignment 1 能用静态分配？

因为该问题的工作量是**可预测的**——每个网格单元的计算代价已知且相等。

---

## 五、半静态分配 (Semi-static Assignment)

### 5.1 核心理念

近期的过去是近期未来的良好预测器——应用程序周期性对执行情况进行**性能采样**，并据此重新调整分配方案。在两次调整之间，分配是"静态"的。

### 5.2 典型应用

- **自适应网格 (Adaptive Mesh)**：当物体移动或流过物体的流体发生变化时，网格随之变化，但变化缓慢。按颜色标定各处理器负责的区域。
- **粒子模拟 (Particle Simulation)**：当粒子在模拟过程中移动时，周期性地将粒子重新分配给各处理器（粒子移动缓慢 → 不需要频繁重新分配）。

---

## 六、动态工作分配 (Dynamic Assignment)

### 6.1 适用场景

任务的执行时间**未知或不可预测**，或任务总数未知。

### 6.2 示例：质数测试

**串行版本（独立循环迭代，每轮执行时间不确定）：**

```cpp
int N = 1024;
int* x = new int[N];
bool* is_prime = new bool[N];
// assume elements of x initialized here
for (int i = 0; i < N; i++) {
    // 未知执行时间：依赖于 x[i] 的值
    is_prime[i] = test_primality(x[i]);
}
```

**并行版本（SPMD 多线程 + 共享计数器）：**

```cpp
int N = 1024;
int* x = new int[N];
bool* is_prime = new bool[N];
// assume elements of x initialized here

LOCK counter_lock;
int counter = 0;  // 共享变量

while (1) {
    int i;
    lock(counter_lock);
    i = counter++;
    unlock(counter_lock);
    if (i >= N)
        break;
    is_prime[i] = test_primality(x[i]);
}
```

或使用原子操作：`atomic_incr(counter);`

### 6.3 使用工作队列的动态分配

```
          共享工作队列
        ┌──────────────┐
        │  sub-problem │ ← 子问题（a.k.a. "task", "work"）
        │  sub-problem │
T1 ────→│  sub-problem │←──── T2
T3 ────→│  sub-problem │←──── T4
        │  sub-problem │
        └──────────────┘

工作线程:
- 从共享队列取出数据/任务
- 将新产生的任务推入队列
（目前假设每个任务相互独立）
```

---

## 七、任务粒度 (Task Granularity)

### 7.1 细粒度 vs 粗粒度

**细粒度分配 (Fine Granularity)：1 个 task = 1 个元素**

```cpp
const int N = 1024;
float* x = new float[N];
bool* is_prime = new bool[N];

LOCK counter_lock;
int counter = 0;
while (1) {
    int i;
    lock(counter_lock);
    i = counter++;
    unlock(counter_lock);
    if (i >= N)
        break;
    is_prime[i] = test_primality(x[i]);
}
```

- **优点**：负载均衡好（大量小任务 → 任务数 >> 处理器数）
- **缺点**：同步开销高（临界区频繁竞争）
  - 临界区内的执行时间：串行部分！（Amdahl 定律）
  - 临界区是**串行程序中不存在的额外开销**

**粗粒度分配 (Coarse Granularity)：1 个 task = 10 个元素**

```cpp
const int N = 1024;
const int GRANULARITY = 10;
float* x = new float[N];
bool* is_prime = new bool[N];

LOCK counter_lock;
int counter = 0;
while (1) {
    int i;
    lock(counter_lock);
    i = counter;
    counter += GRANULARITY;
    unlock(counter_lock);
    if (i >= N)
        break;
    int end = min(i + GRANULARITY, N);
    for (int j = i; j < end; j++)
        is_prime[j] = test_primality(x[j]);
}
```

- **优点**：同步开销降低（临界区进入次数减少 10 倍）
- **缺点**：可能引入负载不均（大任务块）

### 7.2 选择任务大小的权衡

| 因素 | 倾向 | 原因 |
|------|------|------|
| 良好负载均衡 | **小任务/多任务** | 任务数 >> 处理器数 → 动态分配可均匀负载 |
| 低管理开销 | **大任务/少任务** | 减少同步/调度/分配逻辑开销 |
| **理想粒度** | **取决于工作负载和机器的特性** | 必须理解你的 workload 和你的 machine |

---

## 八、智能任务调度 (Smarter Task Scheduling)

### 8.1 问题：任务执行时间不同

```
按左→右顺序调度 16 个任务：

P1 |████  █  █  █  ████████████████| (最后一个长任务)
P2 | ████  █  █  █                  | (早早完成，空闲)
P3 |  ██ ██████ ██                  | (早早完成，空闲)
P4 |   █████ █  ██                  | (早早完成，空闲)
        ↑ 大量空闲时间
```

P2、P3、P4 早早完成任务后空闲等待，因为最后一个任务是 P1 的 "长杆"。

### 8.2 方案一：拆分任务

将任务拆分为更多更小的子任务 → "长杆"相对于总执行时间变短。

- 可能增加同步开销
- 有些长任务根本不可拆分（本质上是串行的）

### 8.3 方案二：智能调度——长任务优先

```
先调度长任务：
P1 |████████████████| 完成 (长任务，只做1个)
P2 |████  ██  ██  ██| 完成 (短任务，做4个)
P3 | ████  ██  ██  █| 完成 (短任务，做4个)
P4 |  ██ ████  ██  █| 完成 (短任务，做4个)
    同时完成！
```

- 执行长任务的线程做**更少但更大**的任务
- 执行短任务的线程做**更多但更小**的任务
- 最终所有线程做的工作量大致相同
- 需要对工作负载有一定了解（可预测性）

---

## 九、分布式工作队列 (Distributed Queues)

### 9.1 单队列的问题

所有工作线程都在争用**同一个**共享队列 → 同步瓶颈

### 9.2 分布式队列 + Work Stealing

```
        每个线程有自己专属的工作队列
        ┌──────┐   ┌──────┐
T1 ─────│Queue1│───│Queue2│───── T2
        └──────┘   └──────┘
           ↑          ↑
           │  steal!  │
           │←─────────┤
        ┌──────┐   ┌──────┐
T3 ─────│Queue3│───│Queue4│───── T4
        └──────┘   └──────┘

规则：
1. 线程从自己的队列中取出/放入任务（本地操作）
2. 当本地队列为空时 → 从其他线程的队列 STEAL（偷取）工作
```

**优势**：线程大部分时间操作本地队列，无需全局同步；仅在自己的队列为空时才访问其他队列。

---

## 十、任务依赖 (Task Dependencies)

工作队列中的任务不一定是相互独立的。

```
T1 ──→ Task A ──→ Task B (B 依赖 A)
T2 ──→ Task C

任务依赖图：
  A ──→ B
  │
  C (独立)

任务管理系统：调度器管理任务间的依赖关系
- 只有当前驱任务全部完成后，任务才能被分配执行
```

**Cilk 风格的依赖 API：**

```cpp
foo_handle = enqueue_task(foo);              // 入队 foo（与之前所有任务独立）
bar_handle = enqueue_task(bar, foo_handle);  // 入队 bar，必须等 foo 完成后才能运行
```

---

## 十一、总结：静态 vs 动态分配

| | 静态分配 | 动态分配 |
|---|---|---|
| 分配时机 | 编译时/启动时 | 运行时 |
| 开销 | 极低（仅索引计算） | 有调度和同步开销 |
| 适用场景 | 工作量可预测 | 工作量不可预测 |
| 负载均衡 | 依赖预测准确性 | 可自动适应 |

> 实际上，静态 vs 动态**不是非此即彼的选择**，而是一个连续谱。应该尽可能利用对工作负载的先验知识，以减少负载不均衡和任务管理开销。极限情况下，如果系统完全了解所有信息 → 全面使用静态分配。

---

## 十二、Fork-Join 并行编程模式

### 12.1 常见并行编程模式

| 模式 | 描述 | 代码示例 |
|------|------|----------|
| **数据并行 (Data Parallelism)** | 对多个数据元素执行相同的操作序列 | `map(foo, A, B)` |
| **显式线程管理** | 为每个执行单元（或所需的并发量）创建线程 | `std::thread` |
| **Fork-Join** | 自然表达分治算法中的独立工作 | `cilk_spawn` + `cilk_sync` |

**数据并行示例：**

```cpp
// OpenMP
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    B[i] = foo(A[i]);
}

// ISPC foreach
foreach (i = 0 ... N) {
    B[i] = foo(A[i]);
}

// ISPC bulk task launch
launch[numTasks] myFooTask(A, B);

// 高阶函数 map
map(foo, A, B);

// CUDA (后续课程)
foo<<<numBlocks, threadsPerBlock>>>(A, B);
```

**显式线程管理示例：**

```cpp
float* A;
float* B;
void myFunction(float* A, float* B) { ... }

std::thread thread[NUM_HW_EXEC_CONTEXTS];
for (int i = 0; i < NUM_HW_EXEC_CONTEXTS; i++) {
    thread[i] = std::thread(myFunction, A, B);
}
for (int i = 0; i < num_cores; i++) {
    thread[i].join();
}
```

### 12.2 Fork-Join 与分治算法

快速排序天然适合 fork-join：

```cpp
void quick_sort(int* begin, int* end) {
    if (begin >= end - 1)
        return;
    else {
        int* middle = partition(begin, end);
        quick_sort(begin, middle);       // 独立工作！
        quick_sort(middle + 1, last);    // 独立工作！
    }
}
```

```
调用树：
                quick_sort[0..n]
               /                \
    quick_sort[0..mid]    quick_sort[mid+1..n]
       /        \              /         \
    qs[..]    qs[..]       qs[..]     qs[..]
    (独立)    (独立)        (独立)      (独立)
```

---

## 十三、Cilk Plus 编程模型

### 13.1 Cilk 简介

- C++ 语言扩展
- 最初由 MIT 开发，现在被纳入开源标准（GCC、Intel ICC 支持）
- 本课所有 fork-join 代码示例使用 Cilk Plus

### 13.2 核心原语

**`cilk_spawn foo(args);`**

调用 `foo`，但不同于普通函数调用，**调用者可以与 `foo` 的执行异步并行继续执行**。

```
        fork: 创建新的逻辑控制线程
          ↓
    cilk_spawn foo(args);
          ↓
    [foo 和调用者的后续代码可以并行执行]
```

**`cilk_sync;`**

等待当前函数产生的所有 `cilk_spawn` 调用完成（"与产生的所有调用同步"）。

```
    [所有之前 spawn 的调用必须完成]
          ↓
    cilk_sync;  ← join 点
          ↓
    [继续执行后续代码]
```

**重要规则**：每个包含 `cilk_spawn` 的函数末尾有一个**隐式 `cilk_sync`**。
含义：当一个 Cilk 函数返回时，该函数产生的所有工作都已完成。

### 13.3 对比：普通函数调用 vs cilk_spawn

**普通函数调用（如 C 语言）：**

```
void my_func() {
    // part A
    foo();
    bar();
    // part B
}

执行顺序：
  part A → foo() → [返回] → bar() → [返回] → part B

控制流顺序转移：
  my_func → 进入 foo → 执行 foo → 返回 my_func → 继续执行
  线程始终在同一个调用栈中运行
```

**cilk_spawn：**

```
cilk_spawn foo();
bar();
cilk_sync;

执行可能：
  foo() 和 bar() 可以并行执行
```

### 13.4 Cilk 代码示例

**示例一：foo() 和 bar() 可并行**

```cpp
// foo() 和 bar() 可以并行执行
cilk_spawn foo();
bar();
cilk_sync;
```

**示例二：两个 spawn**

```cpp
// foo() 和 bar() 可以并行执行
cilk_spawn foo();
cilk_spawn bar();
cilk_sync;
```

```
注意：与示例一的并行度相同，但运行时开销可能更高
（两个 spawn 比一个 spawn 开销大）
```

**示例三：多个 spawn**

```cpp
// foo, bar, fizz, buzz 都可以并行执行
cilk_spawn foo();
cilk_spawn bar();
cilk_spawn fizz();
buzz();
cilk_sync;
```

```
并行执行可能：
  Thread 0: buzz()     (调用者自己执行)
  Thread 1: foo()
  Thread 2: bar()
  Thread 3: fizz()
```

### 13.5 抽象 vs 实现

- `cilk_spawn` 抽象**不指定** spawned 调用何时、如何被调度执行
- 它仅保证 spawned 调用**可能**与调用者并发运行
- 一个 Cilk 实现如果把 `cilk_spawn foo()` 实现为普通函数调用 `foo()`，是**正确的**吗？
  - 答案：是的，正确但不高效
- `cilk_sync` 则是一个**调度约束**：所有 spawned 调用必须在 `cilk_sync` 返回之前完成

### 13.6 并行快速排序 (Cilk Plus 版本)

```cpp
void quick_sort(int* begin, int* end) {
    if (begin >= end - PARALLEL_CUTOFF)
        std::sort(begin, end);      // 问题足够小时串行排序
    else {
        int* middle = partition(begin, end);
        cilk_spawn quick_sort(begin, middle);  // 左半部分可与其他工作并行
        quick_sort(middle + 1, last);            // 右半部分在调用者线程执行
    }
}
```

```
递归树（当问题小于 PARALLEL_CUTOFF 时回退到串行排序）：

      quick_sort()  ──── 分割(partition)
     /          \
  spawn          quick_sort()
  qs()           /          \
               spawn       quick_sort()
               qs()

叶子节点：std::sort() — 串行执行
```

**要点**：当问题规模足够小时，spawn 的开销超过并行化的收益 → 直接退回到串行排序。

---

## 十四、编写 Fork-Join 程序的原则

### 14.1 核心思想

通过 `cilk_spawn` 将**独立的工作（潜在并行性）暴露给系统**。

### 14.2 经验法则

- 创建的独立工作至少与机器并行执行能力相当（spawn 的工作量 >= 处理器数量）
- **"并行松弛度" (Parallel Slack)**：独立工作量 / 机器并行执行能力
  - 实践中 ~8 是较好的比值
  - 目的是保持良好的负载均衡
- 但独立工作也不能太多 → 粒度太细 → 管理细粒度工作的开销太大

```
并行松弛度示意图：

机器能力: |████████████████| (满负载)
独立工作: |██|██|██|██|██|██|██|██|██|██|...  (大量小任务)

松弛度过小 → 负载不均衡
松弛度过大 → 管理开销过高
目标值 ≈ 8 → 既均衡又高效
```

---

## 十五、一个简单的（但有缺陷的）调度方案

### 15.1 朴素实现

```
为每个 cilk_spawn 创建一个 pthread：
  cilk_spawn foo() → pthread_create(&t, foo)
  cilk_sync       → pthread_join(t)
```

### 15.2 性能问题

| 问题 | 描述 |
|------|------|
| **重量级 spawn 操作** | `pthread_create` 是系统调用，开销大 |
| **过多并发线程** | 远超物理核心数 |
| **上下文切换开销** | 大量线程竞争少量核心 → 频繁切换 |
| **缓存局部性差** | 工作集比必要的大，缓存命中率低 |

---

## 十六、Cilk Plus 运行时的 Worker 线程池

### 16.1 设计

Cilk Plus 运行时维护一个 **worker 线程池**：

- 在应用启动时创建（实际上是懒初始化——第一次 `cilk_spawn` 时创建）
- 线程数**恰好等于**机器上的执行上下文数（硬件线程数）
- ISPC 也采用相同的实现策略

```
示例：四核 + Hyper-Threading 笔记本 → 8 个 worker 线程

Thread 0 | Thread 1 | Thread 2 | Thread 3
Thread 4 | Thread 5 | Thread 6 | Thread 7

每个线程执行:
while (work_exists()) {
    work = get_new_work();
    work.run();
}
```

### 16.2 思考题

对于以下代码：

```cpp
cilk_spawn foo();
bar();
cilk_sync;
```

从 `foo()` 被 spawn 的时刻开始考虑。

```
spawned child: foo()           ← 产生的新工作
continuation:  bar()           ← 调用函数的剩余部分
```

**问题**：`foo()` 和 `bar()` 应该由哪个线程执行？

---

## 十七、两种执行策略：Run Child First vs Run Continuation First

### 17.1 串行实现（对比基准）

**Run Child First（先执行子任务）：**

```
Thread 0: 先执行 foo()，然后 bar()
→ 相当于普通函数调用
→ continuation 隐式存在于线程调用栈中
```

```
Thread 0 栈:
┌──────────┐
│  bar()   │ ← 等待 foo() 返回后执行
├──────────┤
│  foo()   │ ← 正在执行
├──────────┤
│ cilk_sync│
└──────────┘
Thread 1: 空闲
→ Thread 1 本可以在 foo() 执行期间做 bar()！
```

### 17.2 每线程工作队列

```
cilk_spawn foo() 时：
  线程将 continuation 放入自己的工作队列
  然后开始执行 foo()

Thread 0:
  调用栈: foo()
  工作队列: [bar()] ← continuation

Thread 1:
  工作队列: [] ← 空
```

### 17.3 Work Stealing 机制

```
如果 Thread 1 空闲（自己队列中没有工作）
→ 查看 Thread 0 的队列
→ 发现 bar()，将其偷取到自己的队列中
→ 开始执行 bar()

Thread 0:
  调用栈: foo()
  工作队列: [] ← 工作被偷走了

Thread 1:
  调用栈: bar()  ← 正在执行偷来的工作
  工作队列: [bar()] → []
```

### 17.4 两种策略对比

以以下代码为例分析：

```cpp
for (int i = 0; i < N; i++) {
    cilk_spawn foo(i);
}
cilk_sync;
```

---

**策略一：Run Continuation First（"Child Stealing"）**

```
先执行 continuation，将 child 放入队列供偷取

Thread 0 工作队列（breadth-first）:
[foo(N-1), foo(N-2), ..., foo(0)]
```

- 调用者线程在生成**所有迭代**的 spawned work 后才开始执行任何一个
- 广度优先遍历调用图
- **O(N) 空间**存储 spawned work（最大空间）
- **如果无 steal 发生**：执行顺序与删除 `cilk_spawn` 后的程序完全不同
- 产生很多大块工作 → 对偷取者不友好

---

**策略二：Run Child First（"Continuation Stealing"）**

```
先执行 child，将 continuation 放入队列供偷取

Thread 0 工作队列（depth-first）:
执行 foo(0) → 入队 cont: i=1
执行 foo(1) → 入队 cont: i=2
执行 foo(2) → 入队 cont: i=3
...
```

- 调用者线程每次只创建一个可供偷取项（continuation，代表所有剩余迭代）
- **如果无 steal 发生**：不断从队列弹出 continuation，入队新的 continuation（更新 i 值）
- 执行顺序**与删除 spawn 的串行程序相同**
- 深度优先遍历调用图
- **空间优势**：可证明 T 个线程系统的队列存储**不超过**单线程执行栈空间的 T 倍

```
若 continuation 被偷取：
  → 偷取线程 spawn 并执行下一个迭代

Thread 0:           Thread 1:
  执行 foo(0)         执行 foo(1) (从cont: i=1偷取后展开)
  队列: [cont: i=2]   队列: []
```

---

### 17.5 Cilk 的选择：Run Child First (Continuation Stealing)

Cilk Plus 选择 **Run Child First** 策略。原因：

1. **空间效率**：O(T * serial_stack_space)，而非 O(N)
2. **局部性好**：同一线程执行的工作在调用树上相邻 → 数据局部性更好
3. **偷取的粒度大**：steal 发生在队列头部，偷到的是最"粗"的 continuation

---

## 十八、Work Stealing 的 Dequeue 实现

### 18.1 Dequeue（双端队列）结构

每个 worker 线程持有一个 **dequeue（双端队列）**：

```
        tail (bottom)                   head (top)
本地线程: push/pop ←→ [ task | task | task | task ] ←→ steal: 远程线程
```

- **本地线程**：从 **tail（底部）** push 和 pop 工作
- **远程线程**：从 **head（顶部）** steal 工作

### 18.2 为什么从头部偷取？

以快速排序为例（200 个元素）：

```
Thread 0 工作队列:
  head                                    tail
  [cont:151-200] [cont:76-100] [cont:26-50]
                    ↑ steal

Thread 0 正在: 0-25

Thread 1 steal cont:151-200 → 开始处理 101-200 范围
Thread 2 steal cont:76-100  → 开始处理 76-100 范围
```

偷取队列头部的工作意味着：

1. **偷取最大的工作块**（减少偷取次数）
2. **最大化每个线程执行工作的局部性**（结合 run-child-first 策略）
3. **偷取线程和本地线程不争用同一端**（本地线程操作 tail，远程线程操作 head）

```
进一步分解后：

Thread 0: 0-12         ← 本地操作
Thread 1: 101-113      ← 偷来的大块
Thread 2: 51-63        ← 偷来的大块

各线程队列中的剩余工作（细粒度 continuation）：
Thread 0: [cont:13-25, cont:114-125, ...]
Thread 1: [cont:64-75, cont:126-150, ...]
Thread 2: [cont:76-100, cont:151-200, ...]
```

### 18.3 偷取受害者选择

- **随机选择**：空闲线程随机选择一个线程尝试偷取
- **锁无关实现**：由于本地线程和远程线程操作 dequeue 的不同端，可以实现高效的 **lock-free dequeue**

---

## 十九、对比：递归二分 vs 循环 spawn

```cpp
// 方式一：循环 spawn（不提前生成并行工作）
for (int i = 0; i < N; i++) {
    cilk_spawn foo(i);
}
cilk_sync;

// 方式二：递归二分 spawn（迅速生成大量并行工作）
void recursive_for(int start, int end) {
    while (start <= end - GRANULARITY) {
        int mid = (end - start) / 2;
        cilk_spawn recursive_for(start, mid);
        start = mid;
    }
    for (int i = start; i < end; i++)
        foo(i);
}
recursive_for(0, N);
```

**递归二分的好处**：快速生成并行工作，更快地填满多核机器。

```
循环 spawn 的调用树（链式）:
  foo(0) → foo(1) → foo(2) → ... → foo(N-1)
  (线性链，仅当被 steal 时才产生并行)

递归二分 spawn 的调用树（树状）:
            (0, N)
           /      \
      (0, N/2)   (N/2, N)
       /    \       /    \
  (0,N/4) (N/4,N/2) ...  ...
  (快速产生大量独立子任务，更充分使用多核)
```

---

## 二十、cilk_sync 的实现机制

### 20.1 示例场景

```cpp
for (int i = 0; i < 10; i++) {
    cilk_spawn foo(i);
}
cilk_sync;
bar();
```

### 20.2 情况一：无 Steal 发生

```
Thread 0: 顺序执行 foo(0)..foo(9)，然后执行 bar()

工作队列:
  [cont: i=10 (id=A)] ← 最终为空，所有工作已完成

cilk_sync → no-op（因为没有任何工作被其他线程偷取）
```

如果工作**没有被其他线程偷取**，`cilk_sync` 点**不需要做任何事情**。没有 steal → 没有额外同步开销。

### 20.3 情况二：有 Steal 发生

当有 steal 发生时，需要一个**块描述符 (block descriptor)** 来追踪状态。

```
初始状态（spawn 开始时）:
  block A 描述符: { spawns: 1, done: 0 }
  Thread 0: 执行 foo(0)
  Thread 0 工作队列: [cont: i=0 (id=A)]
```

**步骤 1：Thread 1 偷取 continuation**

```
Thread 1 steal → 获得 cont: i=0 (id=A)
Thread 1: 展开 continuation，spawn foo(1)
  → 更新描述符: { spawns: 2, done: 0 }

Thread 0: 还在执行 foo(0)
Thread 1: 执行 foo(1)，入队 cont: i=1 (id=A)
```

**步骤 2：Thread 2 也来偷取**

```
Thread 2 steal → 获得 cont: i=1
  → spawn foo(2)，更新描述符: { spawns: 3, done: 0 }

Thread 0: foo(0)   (属于 block A)
Thread 1: foo(1)   (属于 block A)
Thread 2: foo(2)   (属于 block A)
```

**步骤 3：Thread 0 完成 foo(0)**

```
描述符: { spawns: 3, done: 1 }
Thread 0 变空闲 → 可能会 steal 更多工作
```

**步骤 4：Thread 0 再次偷取**

```
Thread 0 steal cont: i=2 → spawn foo(3)
描述符: { spawns: 4, done: 1 }
Thread 0: foo(3)
```

**步骤 5：计算接近尾声**

```
只剩 foo(9) 未完成
描述符: { spawns: 10, done: 9 }

各线程:
  Thread 0: 空闲 (已完成 foo(0), foo(3), ...)
  Thread 1: 空闲 (已完成 foo(1), ...)
  Thread 2: 执行 foo(9) ← 最后一个
```

**步骤 6：最后一个 spawn 完成**

```
描述符: { spawns: 10, done: 10 }
→ 全部完成！描述符释放

Thread 2: 恢复执行 continuation → 执行 bar()
```

### 20.4 关键设计要点

```
描述符的生命周期:
  无 steal → 描述符从未创建（零开销）
  有 steal → 描述符追踪 spawn 数量和完成数量
          → 最后一个完成的线程负责清理描述符
```

- **描述符**跟踪：当前 block 产生的 spawn 总数、已完成的 spawn 数量
- **仅在 steal 发生时**才产生 bookkeeping 开销
- **大块工作被 steal** → steal 不频繁 → 开销很低
- **大部分时间**，线程都在操作自己的本地 dequeue（push/pop）

---

## 二十一、Cilk 的 Greedy Join 调度策略

### 21.1 策略定义

- 所有线程只要有工作可做就**尽力 steal**（不会停在 sync 点等待）
- 只有**整个系统中不存在可偷取的工作**时，线程才进入空闲状态
- 发起 `cilk_spawn` 的线程**不一定**是 `cilk_sync` 之后执行后续代码的线程

### 21.2 策略优势

```
Greedy Join 示意图：

Thread 0 达到 cilk_sync:
  → 检查自己 spawn 的工作是否全部完成
  → 如果未完成 → 不等待 idle，而是去 steal 其他人的工作
  → 直到所有工作完成 → 继续执行

Thread 1 完成所有工作:
  → 也来 steal 或等待

所有线程都在积极寻找工作 → 最大化资源利用率
```

---

## 二十二、Cilk 总结

| 维度 | 要点 |
|------|------|
| **编程模型** | Fork-Join 并行：自然表达分治算法 |
| **核心原语** | `cilk_spawn` + `cilk_sync` |
| **运行时** | 固定数量的 worker 线程池（= 硬件执行上下文数） |
| **调度策略** | 基于 work stealing 的局部性感知调度器 |
| **执行策略** | **Run child first (continuation stealing)** |
| **队列结构** | 每线程一个 dequeue：本地 push/pop tail，远程 steal head |
| **同步策略** | Greedy join：线程不等待，始终在 steal 寻找工作 |
| **零 steal = 零开销** | 无 steal 时 `cilk_sync` 是 no-op，描述符仅在 steal 时创建 |
| **其他系统** | OpenMP 等也提供 fork/join 原语 |

---

## 二十三、关键概念索引

| 概念 | 英文 | 所在章节 |
|------|------|----------|
| 负载均衡 | Workload Balancing | 三 |
| 静态分配 | Static Assignment | 四 |
| 半静态分配 | Semi-static Assignment | 五 |
| 动态分配 | Dynamic Assignment | 六 |
| 任务粒度 | Task Granularity | 七 |
| 长任务优先调度 | Longest-Task-First Scheduling | 八 |
| 分布式工作队列 | Distributed Work Queues | 九 |
| 任务依赖 | Task Dependencies | 十 |
| Fork-Join 模式 | Fork-Join Pattern | 十二 |
| Cilk Plus | Cilk Plus | 十三 |
| 并行松弛度 | Parallel Slack | 十四 |
| Worker 线程池 | Worker Thread Pool | 十六 |
| Work Stealing | Work Stealing | 十七 |
| Continuation Stealing | Continuation Stealing (Run Child First) | 十七 |
| Child Stealing | Child Stealing (Run Continuation First) | 十七 |
| 双端队列 | Dequeue | 十八 |
| Greedy Join 调度 | Greedy Join Scheduling | 二十一 |
| 块描述符 | Block Descriptor | 二十 |
| 零 steal 零开销 | Zero-cost When No Steal | 二十 |
| 递归二分 spawn | Recursive Bisection Spawn | 十九 |
| Amdahl 定律 | Amdahl's Law | 三 |
