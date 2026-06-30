# 作业1：四核CPU性能分析

**截止日期：10月6日周一 11:59pm**

**总分100分 + 6分额外加分**

## 概述

本作业旨在帮助你理解现代多核CPU中存在的两种主要并行执行形式：

1. 单个处理核心内的SIMD执行
2. 使用多核心的并行执行（你还将看到Intel超线程的效果）

你还将获得测量和推理并行程序性能的经验（这是一项具有挑战性但重要的技能，将在整个课程中使用）。本作业只涉及少量编程，但需要大量分析！

## 环境配置

> **⚠️ 本地运行说明：** 原作业要求在Stanford myth机器（`myth[51-66].stanford.edu`）上运行，这些机器配备4核4.2 GHz Intel Core i7处理器。由于你没有这些机器的访问权限，本指南已适配为**本地运行**。你可以在自己的Linux机器上完成所有程序。性能数据会因你的CPU而异，但分析思路和编程任务完全相同。

### 步骤1：安装ISPC

ISPC是编译本作业中许多程序所必需的编译器。

从ISPC[下载页面](https://ispc.github.io/downloads.html)获取适合你系统的ISPC编译器二进制文件：

**Linux (x86_64):**

```bash
wget https://github.com/ispc/ispc/releases/download/v1.28.1/ispc-v1.28.1-linux.tar.gz
tar -xvf ispc-v1.28.1-linux.tar.gz
export PATH=$PATH:${HOME}/Downloads/ispc-v1.28.1-linux/bin
```

**macOS (Apple Silicon):**

```bash
wget https://github.com/ispc/ispc/releases/download/v1.28.1/ispc-v1.28.1-MacOS.tar.gz
tar -xvf ispc-v1.28.1-MacOS.tar.gz
export PATH=$PATH:${HOME}/Downloads/ispc-v1.28.1-MacOS/bin
```

可以将 `export` 行添加到 `~/.bashrc` 文件中以永久生效。

### 步骤2：克隆代码

```bash
git clone https://github.com/stanford-cs149/asst1.git
```

## 程序1：使用线程并行生成分形图（20分）

进入 `prog1_mandelbrot_threads/` 目录，构建并运行代码。（输入 `make` 构建，`./mandelbrot` 运行。）该程序生成图像文件 `mandelbrot-serial.ppm`，这是一个著名的复数集合——Mandelbrot集合的可视化。

> **本地查看PPM图像：** 在Linux上可以使用 `display` 命令（需安装ImageMagick：`sudo apt install imagemagick`）或 `eog`。也可以将PPM转换为PNG：`convert mandelbrot-serial.ppm mandelbrot-serial.png`。你的工作是将图像计算并行化，使用[std::thread](https://en.cppreference.com/w/cpp/thread/thread)。起始代码在 `mandelbrotThread.cpp` 的 `mandelbrotThread()` 函数中。目前启动的线程不做任何计算就立即返回。你需要在 `workerThreadStart` 函数中添加代码来完成此任务。

**需要做的事情：**

1. 修改起始代码，使用两个线程并行化Mandelbrot生成。具体来说，线程0计算图像的上半部分，线程1计算下半部分。这种问题分解方式称为**空间分解**。
2. 扩展代码以使用2、3、4、5、6、7和8个线程，相应地划分图像生成工作（线程应获得图像的块）。注意，你的处理器核心数可能与原作业描述的4核不同。在你的实验报告中，制作一个**相对于参考串行实现的加速比**图表，作为线程数的函数（**针对视图1**）。加速比是否与线程数成线性关系？在你的报告中假设为什么会这样（或为什么不这样）？
3. 为了确认（或反驳）你的假设，在 `workerThreadStart()` 的开始和结束处插入计时代码，测量每个线程完成工作所需的时间。
4. 修改工作到线程的映射，在Mandelbrot集合的**两个视图上达到尽可能高的加速比**。你不可以在解决方案中使用线程间的同步。我们希望你提出一个适用于所有线程数的单一工作分解策略——不允许为每种配置硬编码特定解决方案！（提示：有一个非常简单的静态分配可以实现此目标，且不需要线程间的通信/同步。）
5. 现在用超过物理核心数的线程数运行改进后的代码。性能是否明显优于使用等于物理核心数的线程数？为什么或为什么不？

## 程序2：使用SIMD Intrinsics向量化代码（20分）

查看作业1代码库中 `prog2_vecintrin/main.cpp` 里的 `clampedExpSerial` 函数。`clampedExp()` 函数对输入数组的所有元素计算 `values[i]` 的 `exponents[i]` 次幂，并将结果限制在9.999999。在程序2中，你的工作是将这段代码向量化，使其可以在具有SIMD向量指令的机器上运行。

我们要求你使用CS149的"伪向量intrinsics"（在 `CS149intrin.h` 中定义）来实现你的版本。`CS149intrin.h` 库提供了一组在向量值和/或向量掩码上操作的向量指令。（这些函数并不翻译为真实的CPU向量指令，而是在我们的库中为你模拟这些操作，并提供便于调试的反馈。）

**需要做的事情：**

1. 在 `clampedExpVector` 中实现 `clampedExpSerial` 的向量化版本。你的实现应适用于任何输入数组大小（`N`）和向量宽度（`VECTOR_WIDTH`）的组合。
2. 运行 `./myexp -s 10000` 并将向量宽度从2、4、8扫描到16。记录得到的向量利用率。你可以通过更改 `CS149intrin.h` 中的 `#define VECTOR_WIDTH` 值来实现。
3. **额外加分（1分）：** 在 `arraySumVector` 中实现 `arraySumSerial` 的向量化版本。你的实现可以假设 `VECTOR_WIDTH` 是输入数组大小 `N` 的因子。串行实现的运行时间为 `O(N)`，你的实现应力求达到 `(N / VECTOR_WIDTH + VECTOR_WIDTH)` 甚至 `(N / VECTOR_WIDTH + log2(VECTOR_WIDTH))` 的运行时间。

## 程序3：使用ISPC并行生成分形图（20分）

现在你已经熟悉了SIMD执行，我们将回到并行Mandelbrot分形生成（与程序1类似）。程序3通过同时利用CPU的多核和每个核内的SIMD执行单元，实现比程序1更大的加速比。

在程序1中，你通过为系统中的每个处理核心创建一个线程来并行化图像生成。程序3则使用ISPC语言结构来描述**独立的计算**。这些计算可以并行执行而不会违反程序正确性。对于Mandelbrot图像，计算每个像素的值是一个独立的计算。有了这个信息，ISPC编译器和运行时系统负责生成尽可能高效利用CPU并行执行资源的程序。

你将修复程序3中的一个简单错误（该错误导致性能问题，而非正确性问题）。通过正确的修复，你应该观察到性能是原始串行Mandelbrot实现的32倍以上。

### 程序3，第1部分：ISPC基础（20分中的10分）

阅读ISPC代码时，你必须记住，虽然代码看起来很像C/C++代码，但ISPC的执行模型与标准C/C++不同。与C不同，ISPC程序的多个程序实例总是在CPU的SIMD执行单元上并行执行。同时执行的程序实例数量由编译器决定（并针对底层机器专门选择）。

**请再读一遍上一段。真的。**

**需要做的事情：**

1. 编译并运行程序 `mandelbrot ispc`。**ISPC编译器当前配置为生成8-wide AVX2向量指令**（在本地x86机器上）。根据你对这些CPU的了解，你期望的最大加速比是多少？为什么你观察到的数字可能低于这个理想值？（提示：考虑你正在执行的计算的特性？描述图像中对SIMD执行带来挑战的部分。比较渲染Mandelbrot集合不同视图的性能可能有助于确认你的假设。）

> **本地运行注意：** 如果你在ARM机器（如Apple Silicon Mac）上运行，需要将Makefile中的 `avx2-i32x8` 改为 `neon-i32x8`，将 `x86-64` 改为 `aarch64`。

### 程序3，第2部分：ISPC任务（20分中的10分）

ISPC的SPMD执行模型和 `foreach` 等机制便于创建利用SIMD处理的程序。该语言还提供了另一种在ISPC计算中利用多核的机制——启动**ISPC任务**。

**需要做的事情：**

1. 使用 `--tasks` 参数运行 `mandelbrot_ispc`。在视图1上观察到什么加速比？
2. 有一种简单的方法可以通过更改代码创建的任务数来提高 `mandelbrot_ispc --tasks` 的性能。仅通过更改 `mandelbrot_ispc_withtasks()` 函数中的代码，你应该能达到超过串行版本32倍的性能！
3. **额外加分（2分）：** 线程抽象（程序1中使用）和ISPC任务抽象之间有什么区别？

## 程序4：迭代`sqrt`（15分）

程序4是一个ISPC程序，计算2000万个0到3之间随机数的平方根。它使用快速迭代实现，利用牛顿法求解方程 ${\frac{1}{x^2}} - S = 0$。

**需要做的事情：**

1. 构建并运行 `sqrt`。报告单CPU核心（无任务）和使用所有核心（有任务）时的ISPC实现加速比。SIMD并行化带来的加速比是多少？多核并行化带来的加速比是多少？
2. 修改数组值的内容以提高ISPC实现的相对加速比。构造一个特定的输入，**最大化相对于串行版本的加速比**。
3. 为 `sqrt` 构造一个特定输入，**最小化ISPC（无任务）相对于串行版本的加速比**。
4. **额外加分（最多2分）：** 使用AVX2 intrinsics（x86）或Neon intrinsics（ARM）手动编写你自己的 `sqrt` 函数版本。

## 程序5：BLAS `saxpy`（10分）

程序5是BLAS（基本线性代数子程序）库中saxpy例程的实现。`saxpy` 计算简单操作 `result = scale*X+Y`，其中 `X`、`Y` 和 `result` 是 `N` 个元素的向量（程序5中 `N` = 2000万），`scale` 是标量。

**需要做的事情：**

1. 编译并运行 `saxpy`。程序将报告ISPC（无任务）和ISPC（有任务）实现的性能。你观察到使用ISPC with tasks的加速比是多少？
2. **额外加分（1分）：** 注意 `main.cpp` 中消耗的总内存带宽计算为 `TOTAL_BYTES = 4 * N * sizeof(float);`。即使 `saxpy` 从X加载一个元素，从Y加载一个元素，向`result`写入一个元素，乘以4仍然是正确的。为什么？
3. **额外加分：** 提高 `saxpy` 的性能。我们期望的是显著的加速比，而不仅仅是几个百分点。

## 程序6：让K-Means更快（15分）

程序6使用K-Means数据聚类算法对一百万个数据点进行聚类。起始代码中给出了K-means算法的正确实现，但当前状态还不够快。你的工作是找出**在哪里**需要改进以及**如何**改进。

**需要做的事情：**

1. 下载数据集。原作业通过AFS访问数据：

   ```bash
   # 原作业命令（需要Stanford AFS访问权限）：
   # ln -s /afs/ir.stanford.edu/class/cs149/data/data.dat ./data.dat
   ```

   > **⚠️ 本地运行：** 你没有AFS访问权限。你需要从课程GitHub仓库查找数据文件的下载链接，或联系课程教师获取数据访问方式。也可以尝试使用 `scp` 从myth机器下载（如果有账号），或使用自己生成的测试数据来验证代码逻辑的正确性。
   >
2. 运行 `pip install -r requirements.txt` 安装必要的绘图包。然后运行 `python3 plot.py` 生成可视化图像。
3. 使用 `common/CycleTimer.h` 中的计时功能来确定代码中哪里存在性能瓶颈。
4. 基于上一步的发现，改进实现。我们期望达到约2.1倍或更高的加速比。

**约束条件：**

- 只能修改 `kmeansThread.cpp` 中的代码
- 不能修改 `stoppingConditionMet` 函数
- 只能并行化以下**一个**函数：`dist`、`computeAssignments`、`computeCentroids`、`computeCost`

## 关于ARM Mac

如果你使用的是Apple ARM架构的笔记本电脑（M1/M2/M3等），请查看[ARM版本说明](README_aarch64_zh.md)。在ARM机器上运行各种程序会产生不同于x86机器的性能特征，这对理解不同架构的并行执行非常有价值。

> **注意：** 原作业说明中关于ARM的部分是可选的，不计入学分。但作为本地运行的学习者，理解不同架构的特性非常有意义。

## 推荐阅读

想知道ISPC及其创建过程吗？ISPC的两位创建者之一Matt Pharr写了一篇**很棒的博客文章**讲述其开发历史，名为[The story of ispc](https://pharr.org/matt/blog/2018/04/30/ispc-all)。它涉及了并行系统设计的许多问题——特别是限制编程语言范围与通用编程语言的价值。

## 提交说明

> **⚠️ 本地运行说明：** 由于你是本地运行而非在Stanford myth机器上，请注意：
>
> - 在报告中明确说明你使用的机器配置（CPU型号、核心数、是否支持超线程、SIMD宽度等）
> - 性能数据将因硬件不同而与原作业参考值有所差异，这完全正常
> - 分析思路和方法比具体数字更重要

## 资源和备注

- 丰富的ISPC文档和示例可在 [http://ispc.github.io/](http://ispc.github.io/) 找到
- 放大Mandelbrot图像的不同位置非常有趣
- Intel提供了大量关于AVX2向量指令的支持材料：[http://software.intel.com/en-us/avx/](http://software.intel.com/en-us/avx/)
- [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/) 非常有用

