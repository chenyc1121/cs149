# 作业2：从零构建任务执行库

**截止日期：10月16日周四 11:59pm**

**总分100分**

## 概述

每个人都喜欢快速完成任务，在这个作业中，我们要求你做的正是这件事！你将实现一个C++库，在多核CPU上尽可能高效地执行应用程序提供的任务。

在作业的第一部分，你将实现一个任务执行库的版本，支持批量（数据并行）启动同一任务的多个实例。此功能类似于你在作业1中使用的[ISPC任务启动行为](http://ispc.github.io/ispc.html#task-parallelism-launch-and-sync-statements)。

在第二部分，你将扩展任务运行时系统以执行更复杂的**任务图**，其中任务的执行可能依赖于其他任务产生的结果。这些依赖关系约束了任务调度系统可以安全并行运行哪些任务。

本作业将要求你：

* 使用线程池管理任务执行
* 使用互斥锁和条件变量等同步原语协调工作线程的执行
* 实现反映任务图依赖关系的任务调度器
* 理解工作负载特性以做出高效的任务调度决策

我们建议复习[C++同步教程](tutorial/README_zh.md)和[测试用例说明](tests/README_zh.md)。

## 环境配置

> **⚠️ 本地运行说明：** 原作业要求在Amazon AWS `c7g.4xlarge`实例上运行（ARM架构，16个执行上下文）。由于你没有此VM，我们已适配为**本地运行**。你的机器核心数可能不同，使用 `-n` 参数设置为你机器的线程数即可。性能数据会因硬件不同而变化，但代码逻辑和优化思路完全适用。

### 获取代码

```bash
# 如果你已有代码（本地目录）：
cd asst2/asst2-master

# 或者从GitHub下载：
wget https://github.com/stanford-cs149/asst2/archive/refs/heads/master.zip
unzip master.zip
```

> **重要：** 不要修改提供的 `Makefile`。修改可能会破坏评分脚本。

## Part A：同步批量任务启动

在作业1中，你使用了ISPC的任务启动原语来启动N个ISPC任务实例（`launch[N] myISPCFunction()`）。在本作业的第一部分，你将在任务执行库中实现类似的功能。

首先，熟悉 `itasksys.h` 中 `ITaskSystem` 的定义。这个[抽象类](https://www.tutorialspoint.com/cplusplus/cpp_interfaces.htm)定义了任务执行系统的接口。接口包含一个 `run()` 方法：

```cpp
virtual void run(IRunnable* runnable, int num_total_tasks) = 0;
```

`run()` 执行指定任务的 `num_total_tasks` 个实例。我们将每次 `run()` 调用称为**批量任务启动**。

`tasksys.cpp` 中的起始代码包含了一个正确但串行的 `TaskSystemSerial::run()` 实现。`run()` 的一个重要细节是它必须**相对于调用线程同步执行**：当 `run()` 返回时，应用程序保证任务系统已完成批量任务启动中**所有任务**的执行。

### 运行测试

使用 `runtasks` 脚本运行测试。例如，运行 `mandelbrot_chunked` 测试：

```bash
./runtasks -n 16 mandelbrot_chunked
```

> **本地运行：** 将 `-n` 参数设置为你机器的线程数（例如4核8线程则用 `-n 8`，或通过 `nproc` 查看）。

`-n` 选项指定任务系统实现可以使用的最大线程数。完整的可用测试列表通过命令行帮助查看（`-h` 选项）。

`-i` 选项指定性能测量期间运行测试的次数。

我们还提供了性能评分用的测试框架：

```bash
python3 ../tests/run_test_harness.py
```

关于参考二进制文件：
- Linux ARM: `runtasks_ref_linux_arm`
- macOS ARM (M1): `runtasks_ref_osx_arm`
- macOS x86: `runtasks_ref_osx_x86`

> **本地运行：** 选择与你机器架构匹配的参考二进制文件。如果没有匹配的，`run_test_harness.py` 可能无法正常工作，但 `runtasks` 脚本本身仍然可用。

### 你需要做的事情

你需要实现三个版本的任务系统，复杂性逐步增加：

* `TaskSystemParallelSpawn`
* `TaskSystemParallelThreadPoolSpinning`
* `TaskSystemParallelThreadPoolSleeping`

**在 `part_a/` 子目录中实现你的Part A代码。**

#### 步骤1：迁移到并行任务系统

实现 `TaskSystemParallelSpawn` 类：
- 创建额外的工作线程来执行批量任务启动
- 在 `run()` 开始时创建工作线程，在返回前join它们
- 考虑静态分配还是动态分配任务给线程
- 注意保护共享变量

#### 步骤2：使用线程池避免频繁创建线程

实现 `TaskSystemParallelThreadPoolSpinning` 类：
- 预先创建所有工作线程（线程池模式）
- 工作线程持续循环检查是否有新工作（"自旋"）
- 确保 `run()` 的正确同步行为

#### 步骤3：在无事可做时让线程休眠

实现 `TaskSystemParallelThreadPoolSleeping` 类：
- 使用条件变量让线程在没有工作时休眠
- 考虑可能的竞态条件
- 线程休眠时不占用CPU资源

## Part B：支持任务图执行

在Part B中，你将扩展Part A的任务系统实现，支持可能依赖前序任务的异步任务启动。

`ITaskSystem` 接口有一个额外方法：

```cpp
virtual TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps) = 0;
```

关键区别：
- **异步启动：** `runAsyncWithDeps()` 应立即返回，即使任务尚未完成
- **显式依赖：** 第三个参数指定当前批量任务启动依赖的前序任务
- **同步：** 调用 `sync()` 等待所有前序批量任务启动完成

### 你需要做的事情

扩展你的线程池+休眠实现，正确实现 `runAsyncWithDeps()` 和 `sync()`。**不需要为其他 `TaskSystem` 类实现Part B。**

**在 `part_b/` 子目录中实现你的Part B代码。**

提示：
- 可以将 `runAsyncWithDeps()` 视为将工作推入"工作队列"
- 需要适当的簿记来跟踪依赖关系
- 考虑使用两个数据结构：一个用于等待依赖的任务，一个用于就绪可运行的任务

## 评分

**Part A（50分）**
- `TaskSystemParallelSpawn::run()`：正确性5分 + 性能5分（共10分）
- `TaskSystemParallelThreadPoolSpinning::run()` 和 `TaskSystemParallelThreadPoolSleeping::run()`：正确性各10分 + 性能各10分（共40分）

**Part B（40分）**
- 正确性30分
- 性能10分

**实验报告（10分）**

## 提交

> **⚠️ 本地运行说明：** 虽然你在本地运行，但提交时仍需确保代码结构正确，遵循原始提交格式。

提交以下文件：
- `part_a/tasksys.cpp`
- `part_a/tasksys.h`
- `part_b/tasksys.cpp`
- `part_b/tasksys.h`
- 实验报告PDF

## 参考链接

- [C++同步教程](tutorial/README_zh.md)
- [测试用例说明](tests/README_zh.md)
- [C++线程文档](https://en.cppreference.com/w/cpp/thread/thread)
- [std::mutex文档](https://en.cppreference.com/w/cpp/thread/mutex)
- [条件变量教程](https://www.modernescpp.com/index.php/condition-variables)
