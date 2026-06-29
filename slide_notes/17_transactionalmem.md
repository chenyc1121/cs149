# Lecture 17-18: 事务内存 (Transactional Memory)

> Stanford CS149, Fall 2025 — 并行计算  
> Lecture 17: Transactional Memory  
> Lecture 18: Transactional Memory Part II + Course Wrap Up

---

## 目录

1. [引言：提升同步抽象的层次](#1-引言提升同步抽象的层次)
2. [锁机制的困境：在锁与困难之间](#2-锁机制的困境在锁与困难之间)
3. [事务内存的基本概念与语义](#3-事务内存的基本概念与语义)
4. [事务内存的优势](#4-事务内存的优势)
5. [atomic { } 与 lock/unlock 的区别](#5-atomic---与-lockunlock-的区别)
6. [事务内存的实现基础](#6-事务内存的实现基础)
7. [数据版本管理策略](#7-数据版本管理策略)
8. [冲突检测策略](#8-冲突检测策略)
9. [软件事务内存 (STM)](#9-软件事务内存-stm)
10. [硬件事务内存 (HTM)](#10-硬件事务内存-htm)
11. [Intel TSX：受限事务内存 (RTM)](#11-intel-tsx受限事务内存-rtm)
12. [HTM 事务一致性与连贯性示例](#12-htm-事务一致性与连贯性示例)
13. [TM 实现空间总结](#13-tm-实现空间总结)
14. [性能对比](#14-性能对比)
15. [课程总结](#15-课程总结)

---

## 1. 引言：提升同步抽象的层次

### 回顾：从底层原子操作到高层同步原语

本课程之前的内容：
- **机器级原子操作**：test-and-set、fetch-and-op、compare-and-swap (CAS)、load-linked/store-conditional (LL/SC)
- **用原子操作构建高层同步原语**：锁（locks）、屏障（barriers）、无锁数据结构（lock-free data structures）
- **挑战**：使用这些原语编写正确的程序非常困难，容易出现原子性违规、死锁等 bug

### 本讲目标：进一步提升同步的抽象层次

**核心理念：事务内存 (Transactional Memory, TM)**

本讲学习目标：
- 理解什么是事务（transaction）
- 理解 `atomic { }` 代码块与 `lock/unlock` 原语在语义上的区别
- 掌握事务内存实现的设计空间：
  - 数据版本管理策略（Data versioning policy）
  - 冲突检测策略（Conflict detection policy）
  - 检测粒度（Granularity of detection）
- 了解软件事务内存（STM）的基本原理
- 了解硬件事务内存（HTM）的基本原理，及其与缓存一致性协议的关系

### 声明式 vs. 命令式抽象

| 声明式 (Declarative) | 命令式 (Imperative) |
|---|---|
| 程序员定义**要做什么** | 程序员定义**如何做** |
| 示例：执行这 1000 个独立任务 | 示例：创建 N 个工作线程，从共享任务队列中取任务 |
| `atomic { ... }` — 声明需要原子性 | `lock(); ...; unlock();` — 手动管理锁 |

---

## 2. 锁机制的困境：在锁与困难之间

### 锁强制做出的权衡

| 粗粒度锁 (Coarse-grain locking) | 细粒度锁 (Fine-grain locking) |
|---|---|
| 低并发度，但正确性更高 | 高并发度，但正确性更低 |
| 例如：为整个数据结构或所有共享内存使用一把锁 | 例如：hand-over-hand locking（逐节点加锁） |

**核心矛盾**：
- 粗粒度锁：简单安全，但性能差
- 细粒度锁：性能好，但容易出错（死锁、竞争条件）

### 示例：银行账户存款操作

使用锁保证原子性：

```c
void deposit(Acct account, int amount)
{
    lock(account.lock);
    int tmp = bank.get(account);
    tmp += amount;
    bank.put(account, tmp);
    unlock(account.lock);
}
```

`deposit` 是一个"读-改-写"操作，需要相对于其他账户操作是原子的。锁通过互斥访问来保证原子性。

### 示例：双向链表的 PushLeft

```c
void PushLeft(DQueue *q, int val) {
    QNode *qn = malloc(sizeof(QNode));
    qn->val = val;

    QNode *leftSentinel = q->left;
    QNode *oldLeftNode = leftSentinel->right;
    qn->left = leftSentinel;
    qn->right = oldLeftNode;
    leftSentinel->right = qn;
    oldLeftNode->left = qn;
}
```

### 示例：树的并发更新 — Hand-over-hand Locking

考虑两个线程同时修改节点 3 和节点 4：

```
     1
    / \
   2   3
        \
         4
```

**Hand-over-hand locking** 的问题：
- 需要获取节点 1 的锁，然后节点 2/3 的锁，依次传递
- 即使两个线程修改的是不同节点，锁的获取顺序也可能导致不必要的阻塞
- 例如：更新节点 3 时对节点 1 和 2 的锁定，可能延迟另一个线程对节点 4 的更新

### 示例：Java HashMap 的线程安全问题

**非线程安全的 HashMap：**

```java
public Object get(Object key) {
    int idx = hash(key);          // 计算哈希
    HashEntry e = buckets[idx];   // 找到桶
    while (e != null) {           // 在桶中查找
        if (equals(key, e.key))
            return e.value;
        e = e.next;
    }
    return null;
}
```

- 优点：不需要同步时无锁开销
- 缺点：需要同步时不安全

**Java 1.4 的同步方案 — 粗粒度锁：**

```java
public Object get(Object key) {
    synchronized (myHashMap) {  // 对整个 HashMap 加锁
        return myHashMap.get(key);
    }
}
```

- 优点：线程安全，编程简单
- 缺点：限制并发，可扩展性差

**细粒度锁方案（每桶一把锁）：**

- 优点：线程安全，并发度更高
- 缺点：即使不需要同步也必须承受锁开销

---

## 3. 事务内存的基本概念与语义

### 事务编程

```java
// 使用锁（命令式）
void deposit(Acct account, int amount) {
    lock(account.lock);
    int tmp = bank.get(account);
    tmp += amount;
    bank.put(account, tmp);
    unlock(account.lock);
}

// 使用事务（声明式）
void deposit(Acct account, int amount) {
    atomic {
        int tmp = bank.get(account);
        tmp += amount;
        bank.put(account, tmp);
    }
}
```

**`atomic { }` 的特点：**
- **声明式**：程序员声明"要做什么"（保证该代码的原子性），而不指定"怎么做"
- 无需显式使用或管理锁
- 系统负责实现同步以保证原子性
- 实现方式：**乐观并发控制** — 只在真正有争用（R-W 或 W-W 冲突）时进行串行化

### 事务内存 (TM) 语义

内存事务：**一组原子的、隔离的内存访问序列**（灵感来源于数据库事务）

| 特性 | 定义 |
|---|---|
| **原子性 (Atomicity)** | 全部或全不：事务提交时，所有内存写入一次性生效；事务中止时，所有写入都不会生效（仿佛事务从未发生） |
| **隔离性 (Isolation)** | 其他处理器不能在事务提交之前观察到事务的写入 |
| **可串行化 (Serializability)** | 事务看起来按单一串行顺序提交，但具体顺序不由事务语义保证 |

**直观理解**：我们对单地址的一致性维护（在缓存一致性系统中），现在要扩展到对**一组**读写操作进行同样的维护。

例如：
- 事务 A：读 X, Y, Z；写 A, X
- 其他处理器要么观察到所有这些操作，要么一个都观察不到
- 这些操作在效果上**同时发生**

### 事务操作示例：树更新

**事务 A**：读节点 1, 2, 3，写节点 3

```
     1
    / \
   2   3              事务 A: READ {1, 2, 3}, WRITE {3}
        \
         4            事务 B: READ {1, 2, 4}, WRITE {4}
```

**无冲突情况**：
- 事务 A：READ{1, 2, 3}, WRITE{3}
- 事务 B：READ{1, 2, 4}, WRITE{4}
- **没有 R-W 冲突，也没有 W-W 冲突**
- 两个事务可以并行执行，无需串行化

**有冲突情况**：
- 事务 A：READ{1, 2, 3}, WRITE{3}
- 事务 B：READ{1, 2, 3}, WRITE{3}
- **存在 W-W 冲突**：两个事务都写节点 3
- 两个事务必须串行化执行

---

## 4. 事务内存的优势

### 1. 易于使用的同步构造

- 程序员声明需要原子性，系统负责实现
- 事务和粗粒度锁一样易用
- 性能却能和细粒度锁一样好

### 2. 提供自动读-读并发和细粒度并发

- **性能可移植性**：为 4 核 CPU 设计的锁方案可能在 64 核上不是最优的
- **生产力论点**：系统对事务的支持可以达到专家级细粒度锁编程 90% 的收益，但只需要 10% 的开发时间

### 3. 失败原子性与恢复

- 线程失败时不会丢失锁（因为没有显式锁）
- 失败恢复 = 事务中止 + 重新开始

```java
// 使用锁：需要手动编写异常处理
void transfer(A, B, amount) {
    synchronized(bank) {
        try {
            withdraw(A, amount);
            deposit(B, amount);
        } catch(exception1) { /* undo code 1 */ }
        catch(exception2) { /* undo code 2 */ }
        // ...
    }
}

// 使用事务：系统自动处理
void transfer(A, B, amount) {
    atomic {
        withdraw(A, amount);
        deposit(B, amount);
    }
}
```

使用事务时：
- 系统负责处理异常（除程序员显式管理的异常外）
- 事务中止，所有内存更新被撤销
- 事务要么提交，要么不提交：其他线程不会看到部分更新
- 失败的线程不会持有锁

### 4. 可组合性 (Composability)

**锁的组合问题：**

```java
// 两个线程互相转账 — 死锁！
// Thread 0:
void transfer(A, B, 100) {         // Thread 1:
    synchronized(A) {               void transfer(B, A, 200) {
        synchronized(B) {               synchronized(B) {
            withdraw(A, 100);                synchronized(A) {
            deposit(B, 100);                     withdraw(A, 200);
        }                                        deposit(B, 200);
    }                                        }
}                                        }
```

锁的嵌套容易导致死锁，需要系统级的获取策略（破坏模块化）。

**事务的组合方案：**

```java
void transfer(A, B, amount) {
    atomic {
        withdraw(A, amount);
        deposit(B, amount);
    }
}

// Thread 0: transfer(A, B, 100)
// Thread 1: transfer(B, A, 200)
```

事务可以优雅地组合：
- 程序员声明全局意图（`transfer` 的原子执行），无需了解全局实现策略
- `transfer` 中的事务包含 `withdraw` 和 `deposit` 中定义的任何事务
- 最外层事务定义原子性边界
- 系统根据冲突情况自动管理：冲突则串行化，不冲突则并行

---

## 5. atomic { } 与 lock/unlock 的区别

**关键区别：**

| `atomic { }` | `lock() + unlock()` |
|---|---|
| 高层声明：声明需要原子性 | 底层阻塞原语 |
| 不指定原子性的实现方式 | 本身不提供原子性或隔离性 |
| 实现可以是锁，也可以是其他机制 | 可用于实现 atomic 代码块 |

**注意事项：**

1. 锁可以用于实现 atomic 代码块，但锁也可以用于原子性之外的用途
2. **不能将所有的锁使用都替换为 atomic 区域**
3. `atomic { }` 消除了许多数据竞争，但编程错误仍可能导致原子性违规

### 不能用 atomic 替代的情况

```java
// Thread 1
synchronized(lock1) {
    // ...
    flagA = true;
    while (flagB == 0);  // 自旋等待
    // ...
}

// Thread 2
synchronized(lock2) {
    // ...
    flagB = true;
    while (flagA == 0);  // 自旋等待
    // ...
}
```

这种**使用锁进行条件同步**（flag-based signaling）的情况不能用 atomic 替代，因为事务不提供两个 atomic 块之间的排序语义。

### 原子性违规示例

```java
// Thread 1 — 程序员错误地将一个逻辑原子代码序列
// 拆分为两个 atomic 块
atomic {
    // ...
    ptr = A;
    // ...
}
// 另一个线程可能在这里将 ptr 设为 NULL！
atomic {
    B = ptr->field;  // 可能空指针解引用！
}

// Thread 2
atomic {
    // ...
    ptr = NULL;
}
```

即使使用了 `atomic { }`，程序员仍可能因错误地将本应原子执行的代码分成两个事务而导致原子性违规。

---

## 6. 事务内存的实现基础

### 两个核心实现问题

TM 系统必须在保证**原子性**和**隔离性**的同时，尽可能维持**高并发**。

1. **数据版本管理 (Data Versioning)**：
   系统如何为并发事务管理未提交的（新）版本和已提交的（旧）版本数据？

2. **冲突检测 (Conflict Detection)**：
   系统如何/何时确定两个并发事务之间存在冲突？

### 冲突类型

| 冲突类型 | 定义 |
|---|---|
| **R-W 冲突 (Read-Write)** | 事务 A 读取地址 X，而事务 B 正在写（但尚未提交）地址 X |
| **W-W 冲突 (Write-Write)** | 事务 A 和事务 B 都处于 pending 状态，且都写地址 X |

系统必须跟踪每个事务的：
- **读集 (Read-set)**：事务期间读取的地址集合
- **写集 (Write-set)**：事务期间写入的地址集合

---

## 7. 数据版本管理策略

### 策略一：急切版本管理 — 基于 Undo Log

**Eager versioning (undo-log based)**

```
开始事务:    Memory: X=10    Undo Log: (空)
写入 X←15:   Memory: X=15    Undo Log: X:10
提交事务:    Memory: X=15    Undo Log: X:10 (清除)
中止事务:    Memory: X=10    Undo Log: X:10 (回滚)
```

- **立即更新内存**，同时维护 undo log 以备中止时使用
- **优点**：提交更快（数据已经在内存中），不需要在提交时大量写数据
- **缺点**：中止较慢（需回滚），存在容错问题（事务中途崩溃会留下脏数据）
- **哲学**：立即写入内存，寄希望于事务不会中止；当中止发生时再处理

### 策略二：惰性版本管理 — 基于 Write Buffer

**Lazy versioning (write-buffer based)**

```
开始事务:    Memory: X=10    Write Buffer: (空)
写入 X←15:   Memory: X=10    Write Buffer: X:15
提交事务:    Memory: X=15    Write Buffer: X:15 (刷新)
中止事务:    Memory: X=10    Write Buffer: X:15 (清除)
```

- 在 write buffer 中缓冲写入数据，直到事务提交
- 提交时才更新实际内存位置
- **优点**：中止更快（只需清空日志），无容错问题
- **缺点**：提交较慢（需要冲刷缓冲区数据到内存）
- **哲学**：仅在必须时才写入内存

### 对比总结

| | Eager (Undo-log) | Lazy (Write-buffer) |
|---|---|---|
| 写入时机 | 立即写入内存 | 提交时写入 |
| 提交速度 | 快 | 慢 |
| 中止速度 | 慢（需回滚） | 快（清空 buffer） |
| 容错性 | 差（崩溃留下脏数据） | 好 |
| 额外开销 | 每次 store 需记录 undo log | 每次 store 写入 buffer |

---

## 8. 冲突检测策略

### 策略一：悲观检测 (Pessimistic / Eager Detection)

**检测时机**：每次 load 或 store 时立即检查冲突

**理念**："我怀疑可能发生冲突，所以在每次内存操作后立即检查。如果迟早要回滚，不如现在做，避免浪费更多工作。"

- **争用管理器 (Contention Manager)** 在检测到冲突时决定是**暂停 (stall)** 还是**中止 (abort)** 事务
- 有多种策略处理常见情况

#### 悲观检测的四种情况

```
情况 1 (成功):
  T0: rd A → wr B → check → wr C → check → commit
  T1:                      check → commit

情况 2 (早期检测 + 暂停):
  T0: wr A → check → commit
  T1: rd A → check → [stall] ... → commit

情况 3 (中止):
  T0: rd A → check → commit
  T1: wr A → check → [restart] → rd A → ...

情况 4 (无进展 — 活锁风险):
  T0: wr A → check → [restart] → wr A → check → [restart] → ...
  T1: wr A → check → [restart] → wr A → check → [restart] → ...
```

注意：以上假设"激进"的争用管理器，写者优先（writer wins），其他事务中止。

#### 悲观检测的优缺点

| 优点 | 缺点 |
|---|---|
| 尽早检测冲突（撤销更少的工作） | 无前向进展保证 |
| 部分中止可以转为暂停 | 在某些情况下导致更多中止 |
| — | 每次 load/store 都需要细粒度通信 |
| — | 冲突检测在关键路径上 |

### 策略二：乐观检测 (Optimistic / Lazy Detection)

**检测时机**：事务尝试提交时才检测冲突

**理念**："让我们寄希望于最好的情况，只在事务尝试提交时解决冲突。"

- 检测到冲突时，给正在提交的事务以**优先权**
- 其他事务可以稍后中止

#### 乐观检测的四种情况

```
情况 1 (成功):
  T0: rd A → wr B → wr C → commit [check]
  T1:                            commit [check]

情况 2 (中止):
  T0: wr A → commit [check]
  T1: rd A → commit [check] → [abort]

情况 3 (成功 — 通过串行化):
  T0: rd A → wr A → commit [check]
  T1:                        commit [check] (T0的写被读到)

情况 4 (前向进展):
  T0: rd A → wr A → commit [check]
  T1: rd A → wr A → commit [check] → [restart] → ... → commit [check]
```

#### 乐观检测的优缺点

| 优点 | 缺点 |
|---|---|
| 前向进展保证 | 检测冲突较晚（浪费更多工作） |
| 批量通信和批量冲突检测 | 仍可能存在公平性问题 |
| 不在每次 load/store 上进行检测 | — |

### 冲突检测策略对比

| | 悲观 (Pessimistic/Eager) | 乐观 (Optimistic/Lazy) |
|---|---|---|
| 检测时间 | 每次 load/store 后 | 提交时 |
| 前向进展保证 | 无（可能活锁） | 有 |
| 浪费工作 | 少 | 多 |
| 通信开销 | 细粒度（每次操作） | 批量（提交时） |
| 是否在关键路径上 | 是 | 仅提交时 |

### 检测粒度 (Granularity of Detection)

| 粒度 | 优点 | 缺点 |
|---|---|---|
| **对象级 (Object)** | 低开销映射操作，暴露优化机会 | 可能出现假冲突 (false conflicts) |
| **字段/元素级 (Element/Word)** | 减少假冲突，提高并发 | 时间/空间开销更大 |
| **缓存行级 (Cache Line)** | 与硬件 TM 天然匹配，减少事务记录存储开销 | 程序员和编译器难以分析 |
| **混合** | 例如：数组用元素级，非数组用对象级 | — |

假冲突示例：
```
事务 1: a.x = ... ; a.y = ...
事务 2: ... = ... a.z ...
```
如果以对象为粒度，事务 1 和事务 2 虽然访问不同字段，但仍被检测为冲突。

---

## 9. 软件事务内存 (STM)

### STM 基本原理

源代码通过编译器转换为带有事务屏障（STM function calls）的代码：

```c
// 程序员编写的源码
atomic {
    a.x = t1;
    a.y = t2;
    if (a.z == 0) {
        a.x = 0;
        a.z = t3;
    }
}
```

```c
// 编译器插桩后的代码
tmTxnBegin();
tmWr(&a.x, t1);
tmWr(&a.y, t2);
if (tmRd(&a.z) != 0) {
    tmWr(&a.x, 0);
    tmWr(&a.z, t3);
}
tmTxnCommit();
```

- N 个 STM 函数调用（软件屏障）用于事务记账
- 需要处理：版本管理、读写集跟踪、提交等
- 使用锁、时间戳、数据复制等技术
- **需要函数克隆或动态翻译**：同一个函数可能在事务内部和外部都被调用

### STM 运行时数据结构

**事务描述符 (Transaction Descriptor)** — 每线程一个：
- 用于冲突检测、提交、中止等
- 包含读集 (read set)、写集 (write set)、undo log 或 write buffer

**事务记录 (Transaction Record)** — 每个数据项一个：
- 指针大小的记录，保护共享数据
- 跟踪数据的事务状态
- **共享状态 (Shared)**：多个读者可访问，使用版本号或共享读锁
- **独占状态 (Exclusive)**：一个写者独占，使用指向所有者事务的写锁
- 这与硬件缓存一致性协议的原理类似

### 数据到事务记录的映射

**在对象中嵌入事务记录 (Java/C# 风格)：**

```
class Foo {
    int x;   ——> TxR (事务记录)
    int y;   ——> TxR
}
```

**基于地址哈希到全局表 (C/C++ 风格)：**

```
struct Foo {
    int x;   \
    int y;   /  通过 hash 映射到全局 TxR 表: TxR1, TxR2, ..., TxRn
}

// 或按字段/数组元素哈希:
f(obj.hash, field.index)  ——> TxR1, TxR2, ..., TxRn
```

两种方式的权衡：嵌入方式减少间接访问开销，哈希方式更灵活且不修改原有数据结构。

### Intel McRT STM 算法示例

**基础配置：**
- **急切版本管理 (Eager versioning)** + **乐观读 (Optimistic reads)** + **悲观写 (Pessimistic writes)**
- 基于**时间戳 (timestamp)** 进行版本跟踪

**时间戳机制：**
- **全局时间戳 (Global timestamp)**：当写事务提交时递增
- **局部时间戳 (Local timestamp)**：事务上一次验证时全局时间戳的值

**事务记录 (32-bit)：**
- **最低位 (LSb)**：0 = 被写者锁定，1 = 未锁定
- **高位 (MS bits)**：
  - 未锁定时：上次提交的时间戳（版本号）
  - 锁定时：指向所有者事务的指针

#### STM 操作流程

**STM 读（乐观）：**
1. 直接读取内存位置（急切版本管理）
2. 验证读取的数据：检查是否未锁定且数据版本 <= 局部时间戳
3. 如果不是，验证读集中所有数据的一致性
4. 插入读集
5. 返回值

**STM 写（悲观）：**
1. 验证数据：检查是否未锁定且数据版本 <= 局部时间戳
2. 获取锁
3. 插入写集
4. 创建 undo log 条目
5. 直接在内存中写入（急切版本管理）

**读集验证 (Read-set Validation)：**
1. 获取全局时间戳
2. 对读集中的每个项：
   - 如果被其他事务锁定，或数据版本 > 局部时间戳，则中止
3. 将局部时间戳设为步骤 1 中获取的全局时间戳值

**STM 提交 (Commit)：**
1. 原子地将全局时间戳加 2（LSb 用于写锁）
2. 如果递增前的（旧）全局时间戳 > 局部时间戳，验证读集（检查最近提交的事务）
3. 对写集中的每个项：
   - 释放锁并将版本号设为新的全局时间戳

#### STM 实例：对象复制

```c
// 事务 X1: 将对象 foo 复制到对象 bar
atomic {
    t = foo.x;
    bar.x = t;
    t = foo.y;
    bar.y = t;
}

// 事务 X2: 读取 bar
atomic {
    t1 = bar.x;
    t2 = bar.y;
}
```

```
foo: x=9, y=7 (timestamp=3)
bar: x=0, y=0 (timestamp=5)
```

**执行过程：**
1. X1 读 foo.x(=9) — 记录读 <foo, 3>
2. X1 写 bar.x(=9) — 记录写 <bar, 5>，undo log: <bar.x, 0>
3. X1 读 foo.y(=7) — 记录读 <foo, 3>
4. X1 写 bar.y(=7) — undo log: <bar.y, 0>
5. X1 提交 — 时间戳更新为 7

6. X2 读 bar.x / bar.y — 要么看到 [0,0]（X1 之前），要么看到 [9,7]（X1 之后），不会看到中间状态

### STM 优化

**整块屏障 vs. 分解屏障：**

```c
// 整块屏障 (Monolithic barriers) — 隐藏冗余
tmTxnBegin()
tmWr(&a.x, t1)    // 每次调用都是完整的屏障函数
tmWr(&a.y, t2)
if (tmRd(&a.z) != 0) {
    tmWr(&a.x, 0);    // 冗余：a 已被打开
    tmWr(&a.z, t3)
}
tmTxnCommit()
```

```c
// 分解屏障 (Decomposed barriers) — 暴露冗余
txnOpenForWrite(a)          // 打开对象 a 用于写入（一次）
txnLogObjectInt(&a.x, a)    // 记录 a.x 的旧值
a.x = t1
txnLogObjectInt(&a.y, a)    // a 已打开，无需再次打开
a.y = t2
if (a.z != 0) {
    txnLogObjectInt(&a.x, a)  // 只记录，无需再次打开
    a.x = 0
    txnLogObjectInt(&a.z, a)
    a.z = t3
}
```

```c
// 编译器优化后
txnOpenForWrite(a)          // 只打开一次
txnLogObjectInt(&a.x, a)
a.x = t1
txnLogObjectInt(&a.y, a)
a.y = t2
if (a.z != 0) {
    a.x = 0                // 直接赋值，无需额外记录（已在 undo log 中）
    txnLogObjectInt(&a.z, a)
    a.z = t3
}
```

**优化效果：**
- 生成更少、更廉价的 STM 操作
- 1 线程情况下，优化后的 STM 开销 < 40%（相对于无并发控制）
- < 30%（相对于基于锁的同步）

### STM 面临的挑战

1. **软件屏障开销**：每次读/写都需函数调用
2. **函数克隆**：同一函数需事务内和事务外两个版本
3. **鲁棒的争用管理**：避免活锁和饥饿
4. **内存模型**：强原子性 vs. 弱原子性

**为什么 STM 慢？**

单线程 STM 性能：比串行执行慢 1.8x – 5.6x

时间分布：
- 大部分时间花在**读屏障 (STMread)** 和**提交 (STMcommit)** 上
- 大多数应用读的数据比写的数据多

---

## 10. 硬件事务内存 (HTM)

### 硬件支持的类型

| 类型 | 版本管理 | 冲突检测 |
|---|---|---|
| **硬件加速 STM (HASTM)** | SW | SW — 为关键瓶颈提供 HW 加速 |
| **硬件 TM (HTM)** | HW | HW — 无 SW 屏障 |
| **混合 TM (Hybrid TM)** | HW/SW 切换 | HW/SW 切换 |

### HTM 核心实现原理

**数据版本管理在缓存中实现：**
- 将缓存作为 write buffer 或 undo log
- 为缓存行添加新的元数据位，跟踪事务的读集和写集

**冲突检测通过缓存一致性协议实现：**
- 一致性查找可以检测事务间的冲突
- 适用于 snooping 和 directory 两种一致性协议

**注意**：事务开始前还需对寄存器做检查点（checkpoint），以便中止时恢复执行上下文。

### HTM 缓存行设计

```
+---------+------+------+------+---------------------------+
|   Tag   | MESI |  R   |  W   |  Line Data (e.g., 64B)   |
+---------+------+------+------+---------------------------+
```

- **R 位 (Read bit)**：表示该缓存行被事务读取（在 load 时设置）
- **W 位 (Write bit)**：表示该缓存行被事务写入（在 store 时设置）
- **MESI 状态位**：常规缓存一致性协议状态位
- R/W 位在事务提交或中止时**批量清除 (gang-clear)**
- 对于急切版本管理，需要第二次缓存写入来存储 undo log

### HTM 冲突检测方式

利用一致性请求检查 R/W 位：

| 观察到的事件 | 冲突类型 |
|---|---|
| 对 W-word 的共享请求 | **Read-Write 冲突** |
| 对 R-word 的独占请求（意图写入） | **Write-Read 冲突** |
| 对 W-word 的独占请求（意图写入） | **Write-Write 冲突** |

### HTM 示例实现：Lazy-Optimistic (惰性-乐观)

#### CPU 变更

- 寄存器检查点能力（保存/恢复寄存器状态）
- TM 状态寄存器（状态、中止处理程序指针等）

#### 缓存变更

- R 位：标记读集成员
- W 位：标记写集成员

#### 事务执行流程

**1. 事务开始 (Xbegin)：**
```
Xbegin
```
- 初始化 CPU 和缓存状态
- 对寄存器做检查点
- 所有 R 和 W 位初始为 0

**2. Load 操作：**
```
Load A    (A=33)
Load B    (B=5)
```
- 按需处理缓存缺失
- 将数据标记为读集（设置 R 位=1）

**3. Store 操作：**
```
Store C ← 5
```
- 按需处理缓存缺失
- 将数据标记为写集（设置 W 位=1）
- 注意：这不是将缓存行加载到独占状态（惰性版本管理）

**4. 快速两阶段提交 (Fast Two-Phase Commit)：**
```
Xcommit
```
- **验证阶段 (Validate)**：请求写集缓存行的 RdX 访问权限（如需要），确保没有其他核心冲突
- **提交阶段 (Commit)**：批量清除 R 和 W 位，将写集数据转为有效（dirty）数据
- 提交后：C 变为 dirty 状态（M 状态），A 和 B 保持共享状态

**5. 快速冲突检测与中止：**
```
(远程核心提交了对 A 和 D 的写入)
→ 发送 upgradeX A 和 upgradeX D 一致性请求
→ 本地检查：对 A 的独占请求与本地对 A 的读冲突
→ 触发本地事务中止
```
- **检测**：检查独占请求是否针对读集或写集中的地址
  - 对 R-word 的独占请求 = Write-Read 冲突
  - 对 W-word 的独占请求 = Write-Write 冲突
- **中止**：使写集缓存行无效，批量清除 R 和 W 位，恢复寄存器检查点

### HTM 性能

- 比 STM 快 2x – 7x
- 单线程性能在串行执行的 10% 以内
- 随处理器数量高效扩展

---

## 11. Intel TSX：受限事务内存 (RTM)

### Intel Haswell 架构的 RTM 支持

**新指令：**

| 指令 | 功能 |
|---|---|
| `XBEGIN` | 开始事务，接受一个中止时的回退地址（fallback address），例如回退到使用 spin-lock 的代码路径 |
| `XEND` | 提交事务 |
| `XABORT` | 显式中止事务 |

### 实现特点

- 在 **L1 缓存**中跟踪读集和写集
- 处理器确保所有内存操作原子地提交
- **但处理器可能因多种原因自动中止事务**，例如：
  - 读集或写集中的缓存行被逐出 → 事务中止
  - 缓存容量不足
  - 检测到冲突
  - 中断/异常

### 最佳努力 HTM (Best-Effort HTM)

- 实现**不保证前向进展**（这就是为什么需要 fallback 地址）
- 如果事务反复中止，回退到 fallback 路径（如使用 spin-lock）
- 《Intel 优化指南》第 12 章给出了提高事务成功率的指导方针

**典型使用模式：**

```c
// 尝试使用事务
if (_xbegin() == _XBEGIN_STARTED) {
    // 事务代码路径
    // 读/写共享数据
    _xend();
} else {
    // fallback 路径：使用传统的锁
    lock(&fallback_lock);
    // 执行相同的操作
    unlock(&fallback_lock);
}
```

### Lock Elision（锁消除）

RTM 的一个重要应用场景：如果使用锁保护的临界区之间没有真正的数据冲突，事务可以完全消除锁的开销。如果发生冲突，事务中止并回退到正常的锁路径。

---

## 12. HTM 事务一致性与连贯性示例

这个示例展示了将 **TM 作为一致性机制**使用——所有事务，随时随地进行。

**假设：**
- 惰性 + 乐观 (Lazy + Optimistic)
- 每个执行步骤所有处理器有一个"提交"
- 被中止的事务会重新执行

### 事务序列

```
P1: Begin T1 | Read A | Write A,1 | Write C,2 | Read D | Commit T1
P2: Begin T2 | Read A | Write E,3 | Commit T2
P3: Begin T3 | Write C,4 | Read A | Write E,5 | Commit T3
     Begin T4 | Read E | Write B,6 | Write C,7 | Read F | Commit T4
```

### 执行过程分析

```
步骤 1:  B T1 | B T2 | B T4
步骤 2:  R A (A:0) | R A (A:0) | R E (E:0)
步骤 3:  W A,1 (RS:{A:0}, WS:{A:1}) | W E (RS:{A:0}, WS:{E:3}) | W B,6 (RS:{E:0}, WS:{B:6})
步骤 4:  W C,2 (RS:{A:0}, WS:{A:1,C:2}) | C T2 | B T4  ← T2 提交成功
步骤 5:  R D (RS:{A:0,D:0}, WS:{A:1,C:2}) | B T3 | R E (→ 看到 E:3)  ← T1 继续 | T3 开始
步骤 6:  C T1 (RS:{A:0,D:0}, WS:{A:1,C:2}) | W C,5 (WS:{C:5}) | W B,6 (RS:{E:3}, WS:{B:6})  ← T1 提交成功
步骤 7:  — | R A (RS:{A:1}, WS:{C:5}) | W C,7 (RS:{E:3}, WS:{B:6,C:7})  ← T3 读到 A=1
步骤 8:  — | W E,6 (RS:{A:1}, WS:{C:5,E:6}) | R F (RS:{E:3,F:0}, WS:{B:6,C:7})
步骤 9:  — | — | C T4 (RS:{E:3,F:0}, WS:{B:6,C:7})  ← T4 提交
步骤 10: — | C T3 (RS:{A:1}, WS:{C:5,E:6})  ← T3 提交
```

**关键观察：**
- T2 读取 A:0（T1 尚未提交），所以看到初始值
- T3 开始后读取 A:1（T1 已提交），所以看到 T1 写入的值
- T4 读取 E:3（T2 已提交），与 T3 之间存在 W-W 冲突（都写 E），但 T4 的 E 读是 3（T2 的值），写 B,6 和 C,7 后在提交时成功
- 最终结果取决于串行化顺序：T2 → T1 → T4 → T3

---

## 13. TM 实现空间总结

### 实现组合示例

| 系统 | 版本管理 | 冲突检测 | 类型 |
|---|---|---|---|
| Sun TL2 | Lazy | Optimistic (rd/wr) | STM |
| MS OSTM | Lazy | Optimistic (rd) / Pessimistic (wr) | STM |
| Intel STM | Eager | Optimistic (rd) / Pessimistic (wr) | STM |
| Intel STM | Eager | Pessimistic (rd/wr) | STM |
| Stanford TCC | Lazy | Optimistic | HTM |
| MIT LTM | Lazy | Pessimistic | HTM |
| Intel VTM | Lazy | Pessimistic | HTM |
| Wisconsin LogTM | Eager | Pessimistic | HTM |

**最优设计仍是开放问题**——对 HW、SW 和 Hybrid 可能各不相同。

### 总体总结

**STM 系统：**
- 编译器添加版本管理和冲突检测代码
- STM 屏障 = 插桩代码（如 StmRead, StmWrite）
- 基础数据结构：
  - 每线程的事务描述符（状态、读集、写集等）
  - 每个数据项的事务记录（锁定/版本号）

**HTM 系统：**
- 版本化数据保存在缓存中
- 冲突检测机制建立在缓存一致性协议之上

---

## 14. 性能对比

### 锁 vs. 事务（平衡树 & HashMap）

| 处理器数 | 粗粒度锁 | 细粒度锁 | TCC (HTM) |
|---|---|---|---|
| 1 | ~基准 | ~基准 | ~基准 |
| 2-16 | 性能不扩展 | 较好扩展 | **最佳扩展** |

TCC 是硬件实现的 TM 系统，在平衡树和 HashMap 两种数据结构上均表现最优。

### STM vs. HTM (3-tier Server Vacation 基准测试)

```
处理器数  | 1    2    4    8    16
理想加速  | 1x   2x   4x   8x   16x
STM      | ~1x ~1.5x ~2x ~2x ~2x
HTM      | ~1x ~2x  ~4x ~7x ~14x
```

HTM 性能：
- 比 STM 快 2x – 7x
- 单线程在串行执行的 10% 以内
- 随处理器数量高效扩展

---

## 15. 课程总结

### CS149 课程的核心问题

1. **识别并行性**（或识别依赖关系）
2. **高效调度工作**：
   - 实现良好的负载均衡
   - 克服通信约束：带宽限制、延迟处理、同步
3. **利用数据/计算局部性** = 高效管理状态

这些议题在多个尺度和场景中被讨论：
- 异构移动 SoC
- 单芯片多核 CPU
- 多核 GPU
- CPU+GPU
- 机器集群
- AI 加速器硬件

### 关键抽象

- **数据并行思维** (Data parallel thinking)
- **功能并行** (Functional parallelism)
- **事务 (Transactions)** — 本讲重点
- **任务 (Tasks)**
- **SPMD** (Single Program Multiple Data)

### 展望

- 在可预见的未来，获得更高性能计算硬件的主要途径是**增加并行性**和**硬件专业化**的组合
- 现代软件的效率与机器的峰值能力仍有巨大差距
- 理解并行机器的工作原理对于获得高性能至关重要

---

> **本笔记涵盖 Lecture 17 和 Lecture 18 的全部内容，包括事务内存的基本概念、语义、STM 与 HTM 实现原理、Intel TSX、冲突检测策略、版本管理策略、性能对比以及课程总结。**
