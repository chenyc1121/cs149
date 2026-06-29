# Lecture 16: 细粒度锁与同步

> Stanford CS149, Fall 2025 — 锁的实现、细粒度同步与无锁编程简介

---

## 1. 课程概述

本讲主要涵盖以下内容：
- **锁的实现**：test-and-set、test-and-test-and-set、ticket lock 等不同锁实现方案及其在缓存一致性系统中的性能特征
- **细粒度锁**：如何使用细粒度锁来提高数据结构的并行度，以排序链表为例
- **无锁数据结构**：lock-free 设计的基本思想、典型例子（队列、栈）以及常见陷阱（ABA 问题、悬空指针）

---

## 2. 预备术语：死锁、活锁与饥饿

### 2.1 死锁（Deadlock）

**定义**：系统中有未完成的操作，但没有任何操作能够继续向前推进的状态。

死锁通常出现在每个操作都持有了另一个操作所需的共享资源的情况下。在死锁状态下，没有任何线程能够继续推进，除非某个线程主动放弃资源（“后退”）。

**死锁的四个必要条件**：
1. **互斥（Mutual Exclusion）**：同一时刻只有一个处理器可以持有给定资源
2. **持有并等待（Hold and Wait）**：处理器在等待其他资源时，必须继续持有已获得的资源
3. **不可抢占（No Preemption）**：处理器在完成所需操作之前不会主动放弃资源
4. **循环等待（Circular Wait）**：等待的处理器之间存在相互依赖关系（资源依赖图中存在环路）

**计算机系统中的死锁示例**：
```c
const int numEl = 1024;
float msgBuf1[numEl];
float msgBuf2[numEl];
int threadId = getThreadId();
// ... do work ...
MsgSend(msgBuf1, numEl * sizeof(int), threadId+1, ...);
MsgRecv(msgBuf2, numEl * sizeof(int), threadId-1, ...);
```
每个线程向 ID 比它大的下一个线程发送消息（阻塞发送），然后从 ID 比它小的上一个线程接收消息。工作队列满时会产生循环等待，导致死锁。

**图例**：线程 A 生产工作放入 B 的工作队列，线程 B 生产工作放入 A 的工作队列，两个队列都满了，互相等待对方消费。

### 2.2 活锁（Livelock）

**定义**：系统正在执行大量操作，但没有线程能取得**有意义的进展**。

- 死锁和活锁关乎**程序正确性**
- 死锁：系统停滞，无法推进
- 活锁：系统在活跃运行，但没有真正完成工作

**计算机系统中的示例**：
- 操作不断中止并重试（例如事务不断冲突回滚）
- 两个人在走廊里相遇，同时向左让，又同时向右让，反复循环但都无法通过

### 2.3 饥饿（Starvation）

**定义**：系统整体在向前推进，但某些进程完全无法取得进展。

- 饥饿通常不是永久状态（一旦条件解除，等待的线程最终可以继续）
- 饥饿更多是**公平性（fairness）**问题，而非正确性问题

---

## 3. 缓存一致性回顾：MSI 状态转换

在讨论锁的实现之前，先回顾 MSI 缓存一致性协议。每个缓存行处于以下三种状态之一：

| 状态 | 含义 | 说明 |
|------|------|------|
| **M (Modified)** | 已修改 | 该缓存行独有且已被修改，与内存不一致 |
| **S (Shared)** | 共享 | 该缓存行可能被多个缓存共享，与内存一致 |
| **I (Invalid)** | 无效 | 该缓存行无效，不可使用 |

**状态转换关键操作**：
- `PrRd`：处理器本地读
- `PrWr`：处理器本地写
- `BusRd`：总线读请求（其他处理器发起的）
- `BusRdX`：总线独占读请求（其他处理器要写，需要独占访问）

**一致性流量分析示例**：给定 P0 和 P1 对地址 X 和 Y 的一系列 load/store 操作，分析每个缓存在每个操作时的状态变化和行为（发出总线请求、刷新脏数据、状态转移等）。

理解 MSI 协议对于理解锁操作的互联开销至关重要——因为每次 test-and-set 都可能触发总线事务和缓存行失效。

---

## 4. 锁的实现方案

### 4.1 基于 Test-and-Set 的锁

**原子 test-and-set 指令**：
```
ts R0, mem[addr]    // 将 mem[addr] 的值加载到 R0
                    // 如果 mem[addr] 为 0，则将其设为 1
```

**锁的实现**：
```asm
lock:
    ts   R0, mem[addr]    // 加载锁变量到 R0
    bnz  R0, lock         // 如果不为 0，说明锁被持有，继续自旋
    // 成功获取锁，进入临界区

unlock:
    st   mem[addr], #0    // 将锁变量设为 0，释放锁
```

**x86 的 cmpxchg 指令**：使用 `lock prefix` 保证原子性。
```asm
lock cmpxchg dst, src
```
逻辑：如果 `dst == EAX`，则设置 `ZF = 1`，`dst = src`；否则设置 `ZF = 0`，`EAX = dst`。

**缓存一致性视角下的 T&S 锁**（多处理器争用场景）：
- 处理器 1 持有锁期间，处理器 2、3 等不断执行 T&S 尝试获取锁
- 每次 T&S 尝试都会发出 `BusRdX`（独占读），导致：
  - 大量总线事务
  - 频繁的缓存行失效
  - 持有锁的处理器甚至难以获取总线来释放锁
- 锁释放后，所有等待的处理器争相获取，只有一个成功，其余继续自旋

**T&S 锁性能特征**（来自 Culler, Singh, and Gupta 的基准测试）：
- 临界区时间被移除后，图表仅显示获取/释放锁的时间
- 随着处理器数量增加，互联争用显著增加锁获取时间
- 争用还会减慢临界区的执行速度（因为锁持有者也必须为总线访问而竞争）

### 4.2 理想锁的性能特征

一个理想的锁实现应具备以下特征：
- **低延迟（Low Latency）**：如果锁空闲且没有其他处理器竞争，应能快速获取锁
- **低互联流量（Low Interconnect Traffic）**：多个处理器同时竞争锁时，应以高效有序的方式获取，尽量减少总线流量
- **可扩展性（Scalability）**：延迟和流量应随处理器数量合理增长
- **低存储成本（Low Storage Cost）**：不要为锁使用过多内存
- **公平性（Fairness）**：避免饥饿或严重的不公平，理想情况下按请求顺序获取锁

**T&S 锁评分**：低延迟（低争用时）、高流量、扩展性差、低存储成本（一个 int）、无公平性保障。

### 4.3 Test-and-Test-and-Set 锁

对 T&S 的改进：在尝试原子操作之前，先通过普通读操作检查锁是否空闲。

```c
void Lock(int* lock) {
    while (1) {
        // 先普通读取，等待锁变为空闲
        while (*lock != 0);   // 假设 *lock 不会被分配为寄存器
        
        // 锁看起来空闲了，尝试原子获取
        if (test_and_set(*lock) == 0)
            return;
    }
}

void Unlock(int* lock) {
    *lock = 0;
}
```

**工作原理**：
1. 当锁被持有时，等待线程在本地缓存中反复读取锁变量（`while (*lock != 0)`）
2. 由于缓存行处于 **Shared (S)** 状态，该读取操作不会产生总线流量
3. 当锁被释放时（写入 0），所有等待者的缓存行被失效（Invalidate），触发一次 `BusRd`
4. 等待者重新读入缓存行（S 状态），然后执行 T&S 尝试获取

**T&T&S 的缓存一致性流量**：
- 每次锁释放时，每个等待处理器产生一次失效（O(P) 次失效）
- 如果所有处理器都缓存了锁变量，总流量为 O(P^2)
- 相比之下，T&S 锁每次 T&S 尝试都会产生失效，流量远大于 O(P^2)

**T&T&S 特征总结**：
- **优点**：大幅减少互联流量，可扩展性更好
- **缺点**：在无争用情况下延迟略高（需要先 test 再 test-and-set）、仍无公平性保障、存储成本不变

### 4.4 Ticket Lock（排队锁）

解决 T&S 系列锁的主要问题：锁释放后所有等待者同时争抢。

```c
struct lock {
    int next_ticket;   // 下一个可领取的号码
    int now_serving;   // 当前正在服务的号码
};

void Lock(lock* l) {
    int my_ticket = atomic_increment(&l->next_ticket);  // 取号
    while (my_ticket != l->now_serving);                 // 等待叫号
}

void unlock(lock* l) {
    l->now_serving++;   // 叫下一个号
}
```

**关键特性**：
- 获取锁时不需要原子操作（只需要读取 `now_serving`）
- 每个锁释放只产生一次失效（O(P) 互联流量，比 T&T&S 的 O(P^2) 更好）
- 天然提供公平性保证（FIFO 顺序）
- 获取锁需要一次 `atomic_increment` 原子操作来取号

---

## 5. 原子操作

### 5.1 CUDA 提供的原子操作

```c
int   atomicAdd(int* address, int val);
float atomicAdd(float* address, float val);
int   atomicSub(int* address, int val);
int   atomicExch(int* address, int val);
float atomicExch(float* address, float val);
int   atomicMin(int* address, int val);
int   atomicMax(int* address, int val);
unsigned int atomicInc(unsigned int* address, unsigned int val);
unsigned int atomicDec(unsigned int* address, unsigned int val);
int   atomicCAS(int* address, int compare, int val);
int   atomicAnd(int* address, int val);   // 位与
int   atomicOr(int* address, int val);    // 位或
int   atomicXor(int* address, int val);   // 位异或
```

### 5.2 用 atomicCAS 实现原子操作

**atomicCAS（Compare and Swap）定义**：
```c
int atomicCAS(int* addr, int compare, int val) {
    int old = *addr;
    *addr = (old == compare) ? val : old;
    return old;
}
```

**用 CAS 实现 atomic_min**：
```c
void atomic_min(int* addr, int x) {
    int old = *addr;
    int new = min(old, x);
    while (atomicCAS(addr, old, new) != old) {
        // CAS 失败，说明其他线程修改了 *addr
        // 重新读取当前值并计算新的 min
        old = *addr;
        new = min(old, x);
    }
}
```

> **思考题**：如何用同一模式实现 `atomic_increment` 和 `lock`？

**用 CAS 构建互斥锁**：
```c
typedef int lock;

// 简单版本：无自旋等待优化
void lock(Lock* l) {
    while (atomicCAS(l, 0, 1) == 1);
}

// 优化版本：先读后 CAS（类似 T&T&S 的思路）
void lock(Lock* l) {
    while (1) {
        while (*l == 1);            // 等待锁变为空闲
        if (atomicCAS(l, 0, 1) == 0) // 尝试获取
            return;
    }
}

void unlock(Lock* l) {
    *l = 0;
}
```

> 优化版在高争用下效率更高，因为等待线程可以在本地缓存中（S 状态）读取锁变量，减少总线流量。

### 5.3 Load-Linked / Store-Conditional (LL/SC)

- **不是**单一原子指令，而是一对对应的指令：
  - `load_linked(x)`：从地址 x 加载值
  - `store_conditional(x, value)`：如果自对应 load_linked 以来 x 未被任何处理器写入过，则将 value 存储到 x
- ARM 对应指令：`LDREX` 和 `STREX`
- 在缓存一致性处理器上的实现：利用缓存一致性协议检测是否有其他处理器写入过该地址

### 5.4 C++11 的 atomic\<T\>

```cpp
atomic<int> i;
i++;                              // 原子递增 i
int a = i;
// ... do stuff ...
i.compare_exchange_strong(a, 10); // 如果 i 的值等于 a，则将 i 设为 10
bool b = i.is_lock_free();        // 检查原子操作的实现是否是无锁的
```

**特性**：
- 提供完整对象的原子读、写、读-改-写操作
- 对于基本类型 T，原子性可能由处理器原子指令高效实现；对于复杂类型，可能由互斥锁实现
- 提供**内存顺序语义**（memory ordering semantics），默认使用顺序一致性（sequential consistency）
- 可通过 `std::memory_order` 进一步控制

---

## 6. 使用锁 — 细粒度锁

### 6.1 问题场景：排序链表的多线程并发操作

**数据结构定义**：
```c
struct Node {
    int value;
    Node* next;
};

struct List {
    Node* head;
};
```

**`insert` 操作（无锁版本）**：
```c
void insert(List* list, int value) {
    Node* n = new Node;
    n->value = value;
    // 假设头插情况已处理（保持代码简洁）
    Node* prev = list->head;
    Node* cur = list->head->next;
    while (cur) {
        if (cur->value > value)
            break;
        prev = cur;
        cur = cur->next;
    }
    n->next = cur;
    prev->next = n;
}
```

**`delete` 操作（无锁版本）**：
```c
void delete(List* list, int value) {
    // 假设删除头结点的情况已处理
    Node* prev = list->head;
    Node* cur = list->head->next;
    while (cur) {
        if (cur->value == value) {
            prev->next = cur->next;
            delete cur;
            return;
        }
        prev = cur;
        cur = cur->next;
    }
}
```

**多线程并发执行的问题**：
- **同时插入**：两个线程计算出相同的 `prev` 和 `cur`，后写入的插入覆盖了先写入的，导致一个插入丢失
- **同时插入和删除**：一个线程在插入时，另一个线程删除了 `cur` 指向的节点，导致插入新节点连接到已被删除的节点上，链表结构损坏

### 6.2 方案一：整体锁（Coarse-Grained Locking）

给整个链表加一把锁：

```c
struct List {
    Node* head;
    Lock  lock;     // 每个链表一把锁
};

void insert(List* list, int value) {
    Node* n = new Node;
    n->value = value;
    lock(list->lock);
    // ... 遍历和插入逻辑 ...
    unlock(list->lock);
}

void delete(List* list, int value) {
    lock(list->lock);
    // ... 遍历和删除逻辑 ...
    unlock(list->lock);
}
```

**评价**：
- **优点**：实现简单，正确性容易保证
- **缺点**：所有操作被串行化，限制了并行程序的性能

### 6.3 方案二：细粒度锁 — 手接手锁定（Hand-over-Hand Locking）

**核心思想**：不是锁整个链表，而是只锁定当前正在访问的节点。遍历时，先锁定下一个节点，再释放当前节点的锁，像“手接手”传递一样。

**数据结构修改**：每个节点包含自己的锁。

```c
struct Node {
    int value;
    Node* next;
    Lock* lock;    // 每个节点一把锁
};

struct List {
    Node* head;
    Lock* lock;    // 链表自身的锁（保护 head 指针）
};
```

**`insert` 操作（细粒度锁版本）**：
```c
void insert(List* list, int value) {
    Node* n = new Node;
    n->value = value;
    // 假设头插情况已处理
    
    Node* prev, *cur;
    
    // 先锁链表，获取 prev，再锁 prev
    lock(list->lock);
    prev = list->head;
    lock(prev->lock);
    unlock(list->lock);          // 释放链表锁，现在只持有 prev 的锁
    
    cur = prev->next;
    if (cur) lock(cur->lock);    // 锁定当前节点
    
    while (cur) {
        if (cur->value > value)
            break;
        
        // 手接手：保留旧 prev，向前移动
        Node* old_prev = prev;
        prev = cur;
        cur = cur->next;
        unlock(old_prev->lock);  // 释放旧 prev 的锁
        if (cur) lock(cur->lock); // 锁定新的 cur
    }
    
    n->next = cur;
    prev->next = n;
    unlock(prev->lock);
    if (cur) unlock(cur->lock);
}
```

**`delete` 操作（细粒度锁版本）**：
```c
void delete(List* list, int value) {
    // 假设删除头结点的情况已处理
    Node* prev, *cur;
    
    lock(list->lock);
    prev = list->head;
    lock(prev->lock);
    unlock(list->lock);
    
    cur = prev->next;
    if (cur) lock(cur->lock);
    
    while (cur) {
        if (cur->value == value) {
            prev->next = cur->next;
            unlock(prev->lock);
            unlock(cur->lock);
            delete cur;
            return;
        }
        Node* old_prev = prev;
        prev = cur;
        cur = cur->next;
        unlock(old_prev->lock);
        if (cur) lock(cur->lock);
    }
    unlock(prev->lock);
}
```

**遍历过程图解（以 delete(11) 为例）**：
- 线程 0 从链表头开始，锁定当前节点和下一个节点
- 每前进一步，释放旧节点的锁，锁定新节点
- 如果线程 1 同时执行 delete(10)，两个线程可以并行遍历链表的不同部分
- 只有当两个线程操作相邻节点时才可能产生竞争

**细粒度锁的权衡**：

| 方面 | 分析 |
|------|------|
| **目标** | 使数据结构上的操作可以并行执行，减少全局锁的争用 |
| **正确性** | 难以确保，需要仔细判断互斥的时机范围 |
| **死锁** | 手接手锁由于始终按同一方向（从表头向尾部）获取锁，所以避免死锁 |
| **开销** | 每次遍历步骤都需获取锁（额外指令 + 遍历引入内存写操作） |
| **存储** | 每个节点需要存储锁（额外空间成本） |
| **折中** | 是否可以锁定连续的 N 个节点作为一个组，来平衡并行度和开销？（类似任务粒度的选择） |

### 6.4 思考练习：二叉搜索树的细粒度锁

```c
struct Tree {
    Node* root;
};

struct Node {
    int value;
    Node* left;
    Node* right;
};

void insert(Tree* tree, int value);
void delete(Tree* tree, int value);
```

> 提示：考虑如何沿搜索路径进行手接手锁定，以及父子节点之间的锁依赖关系。

---

## 7. 无锁（Lock-Free）数据结构

### 7.1 阻塞 vs 无锁

**阻塞算法/数据结构**：
- 允许一个线程无限期地阻止其他线程完成对共享数据结构的操作
- 例如：线程 0 持有链表节点的锁，然后被 OS 换出、崩溃或发生缺页异常——此时其他线程都无法完成对该数据结构的操作
- **关键点**：无论锁的实现使用自旋还是抢占调度，使用锁的算法都是阻塞的

**无锁（Lock-Free）算法**：
- 保证**系统整体有线程能取得进展**（system-wide progress）
- 不可能因为抢占某一个线程而导致整个系统停滞
- 该定义不阻止对某个特定线程的饥饿

### 7.2 单读者/单写者的有界队列

```c
struct Queue {
    int data[N];
    int head;   // 队列头
    int tail;   // 下一个空闲位置
};

void init(Queue* q) {
    q->head = q->tail = 0;
}

// 队列满则返回 false
bool push(Queue* q, int value) {
    // tail 是 head 的前一个位置说明满了
    if (q->tail == MOD_N(q->head - 1))
        return false;
    q->data[q->tail] = value;
    q->tail = MOD_N(q->tail + 1);
    return true;
}

// 队列空则返回 false
bool pop(Queue* q, int* value) {
    if (q->head != q->tail) {
        *value = q->data[q->head];
        q->head = MOD_N(q->head + 1);
        return true;
    }
    return false;
}
```

**特性**：
- 只有两个线程（一个生产者、一个消费者）同时访问队列
- 线程之间永远不需要同步或等待：队列满时 push 失败，队列空时 pop 失败
- 需要顺序一致性的内存系统（或适当的内存屏障，或 C++11 的 `atomic<>`）

### 7.3 单读者/单写者的无界队列

```c
struct Node {
    Node* next;
    int   value;
};

struct Queue {
    Node* head;    // 指向队列头的前一个节点
    Node* tail;    // 指向最后一个元素（若队列非空）
    Node* reclaim; // 待回收节点链表的头部
};

void init(Queue* q) {
    q->head = q->tail = q->reclaim = new Node; // 哨兵节点
}

void push(Queue* q, int value) {
    Node* n = new Node;
    n->next = NULL;
    n->value = value;
    
    q->tail->next = n;          // 将新节点链接到尾部
    q->tail = q->tail->next;    // 更新尾指针
    
    // 生产者线程负责回收已被消费者消费的节点
    while (q->reclaim != q->head) {
        Node* tmp = q->reclaim;
        q->reclaim = q->reclaim->next;
        delete tmp;
    }
}

bool pop(Queue* q, int* value) {
    if (q->head != q->tail) {
        *value = q->head->next->value;
        q->head = q->head->next;  // 前进头指针
        return true;
    }
    return false;
}
```

**关键设计要点**：
- **tail** 指向最后添加的元素（非空时）
- **head** 指向队列头部**之前**的节点（哨兵节点）
- **节点分配和删除由同一个线程执行**（生产者线程），避免并发释放的问题
- 当 `reclaim` 落后于 `head` 时，生产者会在 push 时回收被 pop 过的节点

**队列操作图解**：
1. `push 3, push 10`：head/reclaim 指向哨兵节点，tail 指向 10
2. `pop (返回 3)`：head 移动到 3，reclaim 停留在哨兵节点
3. `pop (返回 10)`：head 移动到 10（与 tail 重合），表示队列空
4. `pop (返回 false)`：队列空，操作失败
5. `push 5 (触发回收)`：reclaim 追赶 head，回收哨兵节点和节点 3，然后添加节点 5

### 7.4 无锁栈（初次尝试）

```c
struct Node {
    Node* next;
    int   value;
};

struct Stack {
    Node* top;
};

void init(Stack* s) {
    s->top = NULL;
}

void push(Stack* s, Node* n) {
    while (1) {
        Node* old_top = s->top;       // 读取当前栈顶
        n->next = old_top;            // 新节点指向旧栈顶
        // CAS：如果 s->top 仍然是 old_top，则将其更新为 n
        if (compare_and_swap(&s->top, old_top, n) == old_top)
            return;
        // CAS 失败，说明其他线程修改了栈顶，重试
    }
}

Node* pop(Stack* s) {
    while (1) {
        Node* old_top = s->top;
        if (old_top == NULL)
            return NULL;
        Node* new_top = old_top->next;
        // CAS：如果 s->top 仍然是 old_top，则将其更新为 new_top
        if (compare_and_swap(&s->top, old_top, new_top) == old_top)
            return old_top;
        // CAS 失败，重试
    }
}
```

**核心思想**：只要没有其他线程修改了栈，当前线程的修改就可以安全进行。与细粒度锁的不同之处在于：线程根本不持有数据结构的锁。

### 7.5 ABA 问题

这是无锁数据结构中最经典的陷阱之一。

**问题场景**（假设栈初始为 A -> B -> C）：

| 时间 | 线程 0 | 线程 1 | 栈状态 |
|------|--------|--------|--------|
| T1 | 开始 `pop()`：old_top=A, new_top=B | | A -> B -> C |
| T2 | | 开始 `pop()`：old_top=A | A -> B -> C |
| T3 | | 完成 `pop()`，返回 A | B -> C |
| T4 | | 修改节点 A：例如 value=42 | B -> C |
| T5 | | 开始 `push(A)`，完成 `push(A)` | A -> B -> C |
| T6 | | 开始 `push(D)`，完成 `push(D)` | D -> A -> B -> C |
| T7 | CAS(s->top, A, B) **成功**！ | | B -> C（丢失了 D！） |
| T8 | 完成 `pop()`，返回 A | | |

**问题根因**：线程 0 的 CAS 只检查 `s->top` 是否还是 `A`，但它无法感知到在这期间 `A` 被弹出后又重新推入。CAS 看到 `s->top == A`（地址值匹配），但实际上栈的结构已经完全不同——节点 D 被意外丢失。

### 7.6 ABA 问题的解决方案：使用计数器

```c
struct Stack {
    Node* top;
    int   pop_count;   // 每 pop 一次就递增
};

void push(Stack* s, Node* n) {
    while (1) {
        Node* old_top = s->top;
        n->next = old_top;
        if (compare_and_swap(&s->top, old_top, n) == old_top)
            return;
    }
}

Node* pop(Stack* s) {
    while (1) {
        int pop_count = s->pop_count;    // 读取当前的 pop 计数
        Node* top = s->top;
        if (top == NULL)
            return NULL;
        Node* new_top = top->next;
        // 双重 CAS：同时检查 top 和 pop_count
        if (double_compare_and_swap(&s->top,       top,       new_top,
                                    &s->pop_count, pop_count, pop_count+1))
            return top;
    }
}
```

**关键**：需要硬件支持**双字比较并交换（DCAS 或 Double-Word CAS）**。即使节点 A 被弹出又推入，`pop_count` 已经发生变化，DCAS 会检测到不一致从而重试。

**x86 支持**：
- `cmpxchg8b`：比较并交换 8 字节（可用于两个 32 位值）
- `cmpxchg16b`：比较并交换 16 字节（可用于两个 64 位值）
- 将 `top` 和 `pop_count` 放在连续内存中，即可用一次双字 CAS 完成

也可以通过对节点分配/重用的策略来解决 ABA 问题（例如使用垃圾回收或 hazard pointer）。

### 7.7 另一个问题：引用已释放的内存

```c
int pop(Stack* s) {
    while (1) {
        Stack old;
        old.pop_count = s->pop_count;
        old.top = s->top;
        if (old.top == NULL)
            return NULL;

        Stack new_stack;
        new_stack.top = old.top->next;        // ← 此时 old.top 可能已被
        new_stack.pop_count = old.pop_count+1; //   其他线程弹出并释放！

        if (doubleword_compare_and_swap(s, old, new_stack)) {
            int value = old.top->value;        // ← 访问可能已释放的内存！
            delete old.top;
            return value;
        }
    }
}
```

**问题**：线程读取了 `old.top`，但在 CAS 成功之前，另一个线程可能已经 pop 并 delete 了该节点。当本线程尝试访问 `old.top->next` 或 `old.top->value` 时，访问的是已释放的内存。

### 7.8 [进阶] Hazard Pointer（危险指针）解决方案

**核心思想**：在释放一个节点之前，确保所有其他线程不再持有对该节点的引用。

```c
// 每个线程的私有变量
Node* hazard;          // 正在访问的节点（不能被删除）
Node* retireList;      // 待删除节点列表
int   retireListSize;  // 待删除数量

int pop(Stack* s) {
    while (1) {
        Stack old;
        old.pop_count = s->pop_count;
        old.top = hazard = s->top;    // 声明“我正在访问这个节点”
        // 其他线程在回收时看到 hazard 指向此节点，就不会删除它
        if (old.top == NULL) {
            return NULL;
        }
        
        Stack new_stack;
        new_stack.top = old.top->next;
        new_stack.pop_count = old.pop_count + 1;
        
        if (doubleword_compare_and_swap(s, old, new_stack)) {
            int value = old.top->value;
            retire(old.top);           // 延迟删除
            return value;
        }
        hazard = NULL;                 // CAS 失败，清除 hazard
    }
}

void retire(Node* ptr) {
    push(retireList, ptr);
    retireListSize++;
    if (retireListSize > THRESHOLD) {
        // 扫描待删除列表，删除不被任何线程引用的节点
        for (each node n in retireList) {
            if (n not pointed to by any thread's hazard pointer) {
                remove n from list;
                delete n;
            }
        }
    }
}
```

---

## 8. 无锁链表

### 8.1 无锁链表插入（仅插入场景）

```c
struct Node {
    int value;
    Node* next;
};

struct List {
    Node* head;
};

// 在指定节点后插入新节点
void insert_after(List* list, Node* after, int value) {
    Node* n = new Node;
    n->value = value;
    // 假设空链表情况已处理
    
    Node* prev = list->head;
    while (prev->next) {
        if (prev == after) {
            while (1) {
                Node* old_next = prev->next;
                n->next = old_next;
                // CAS：确保 prev->next 没被其他线程修改
                if (compare_and_swap(&prev->next, old_next, n) == old_next)
                    return;
                // 重试
            }
        }
        prev = prev->next;
    }
}
```

**与细粒度锁的对比**：
- 无获取锁的开销
- 无每节点存储锁的空间开销
- 但仅支持插入操作，删除更为复杂

### 8.2 无锁链表删除的复杂性

支持无锁删除会显著增加数据结构的复杂度。

**典型问题**：节点 B 被删除的同时，另一个线程在 B 之后插入节点 E。此时：
- CAS 在 `A->next` 上成功（让 A 指向 C）
- CAS 在 `B->next` 上也成功（让 B 指向 E）
- 结果：B 指向 E，但 B 不在链表中，E 被丢失

**推荐阅读**：
- Harris 2001. "A Pragmatic Implementation of Non-Blocking Linked-Lists"
- Fomitchev 2004. "Lock-free linked lists and skip lists"

---

## 9. 无锁 vs 加锁：性能对比

来自 Hunt 2011 的研究 "Characterizing the Performance and Energy Efficiency of Lock-Free Data Structures"：

- 比较了队列、链表、双端队列（Dequeue）上无锁算法与使用 pthread mutex 锁的性能
- 无锁算法在低到中争用下通常表现良好，但在高争用下：
  - **CAS 失败频繁，需要反复重试**（自旋），可能导致性能不如精心设计的锁方案
  - ls = "lock free"（无锁）, fg = "fine grained lock"（细粒度锁）
- **结论**：无锁设计不消除争用，它只是用一种不同的方式处理争用

---

## 10. 实践中何时使用无锁数据结构？

**适合使用锁的场景**（CS149 课程中的典型场景）：
- 假设只有你的程序在使用机器（科学计算、图形学、机器学习、数据分析等）
- 性能是最重要的关注点
- 在这些场景中，**编写良好的加锁代码可以和（甚至比）无锁代码一样快**，且实现通常简单得多

**锁可能出现问题的场景**：
- 存在大量线程（数据库、Web 服务器）
- 线程在临界区内可能发生缺页异常、被抢占等
- 锁可能导致优先级反转（priority inversion）、护航效应（convoying）、临界区内崩溃等问题

---

## 11. 总结

1. **细粒度锁**的目的是减少争用，最大化共享数据结构上的操作并行度
   - 代价：增加代码复杂度（更容易出错）、增加每次执行的开销（更多锁获取/释放指令）

2. **无锁数据结构**是非阻塞的解决方案，避免锁的某些开销和陷阱
   - 代价：实现复杂，确保正确性有额外开销
   - 在现代弱一致性硬件上仍需适当的内存屏障
   - **无锁不等于消除争用**：CAS 在高争用下可能失败，需要自旋重试

3. 无论是锁还是无锁方案，关键都在于**如何在正确性和性能之间做出权衡**

---

## 12. 下讲预告：事务内存（Transactional Memory）

- CAS 在无锁实现中的角色：判断其他线程是否在调用线程执行操作期间修改了数据结构
- **事务内存**：一种更通用的机制，允许系统推测（speculate）一个操作可以在其他线程尝试修改结构之前成功完成
- 提供“中止”（abort）机制，当其他线程确实做了修改时回滚操作

---

## 13. 推荐阅读

- **Michael and Scott 1996.** Simple, Fast and Practical Non-Blocking and Blocking Concurrent Queue Algorithms — 多读者/多写者无锁队列
- **Harris 2001.** A Pragmatic Implementation of Non-Blocking Linked-Lists
- **Fomitchev 2004.** Lock-free linked lists and skip lists
- **Hunt 2011.** Characterizing the Performance and Energy Efficiency of Lock-Free Data Structures
- Michael Sullivan's Relaxed Memory Calculus (RMC) compiler: https://github.com/msullivan/rmc-compiler
- Lock-free code pitfalls: http://www.drdobbs.com/cpp/lock-free-code-a-false-sense-of-security/210600279
- Common pitfalls in writing lock-free algorithms: http://developers.memsql.com/blog/common-pitfalls-in-writing-lock-free-algorithms/
