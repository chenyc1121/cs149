# 作业1：四核CPU性能分析（ARM/Mac版本）

**截止日期：10月3日周五 11:59pm**

**总分100分 + 6分额外加分**

> **注意：在ARM上运行是可选的，不计入学分。** 本指南面向在Apple Silicon (M系列) Mac或其他ARM架构机器上本地运行作业的学生。

## 概述

本作业旨在帮助你理解现代多核CPU中存在的两种主要并行执行形式：

1. 单个处理核心内的SIMD执行
2. 使用多核心的并行执行

## 环境配置

> **⚠️ 本地运行说明：** 你需要在ARM架构的机器上运行代码，如M系列Mac。所有操作均在本地完成，无需远程服务器。

### 步骤1：安装ISPC

从ISPC [下载页面](https://ispc.github.io/downloads.html)获取ISPC编译器：

**macOS (Apple Silicon):**
```bash
wget https://github.com/ispc/ispc/releases/download/v1.28.1/ispc-v1.28.1-MacOS.tar.gz
tar -xvf ispc-v1.28.1-MacOS.tar.gz
export PATH=$PATH:${HOME}/Downloads/ispc-v1.28.1-MacOS/bin
```

**Linux (AArch64):**
```bash
wget https://github.com/ispc/ispc/releases/download/v1.28.1/ispc-v1.28.1-linux.aarch64.tar.gz
tar -xvf ispc-v1.28.1-linux.aarch64.tar.gz
export PATH=$PATH:${HOME}/Downloads/ispc-v1.28.1-linux.aarch64/bin
```

可以将 `export` 行添加到 `~/.bashrc` 或 `~/.zshrc` 文件中以永久生效。

### 步骤2：克隆代码

```bash
git clone https://github.com/stanford-cs149/asst1.git
```

## 程序1：使用线程并行生成分形图（20分）

构建并运行 `prog1_mandelbrot_threads/` 目录中的代码。该程序生成Mandelbrot集合的可视化图像。

> **本地查看PPM图像：** macOS上可以直接用Preview打开PPM文件。也可以安装ImageMagick后使用 `convert` 命令转换为PNG。

你的工作是使用 `std::thread` 并行化图像计算。起始代码在 `mandelbrotThread.cpp` 中提供。

**需要做的事情：**

1. 修改起始代码，使用两个线程并行化Mandelbrot生成（空间分解）。

2. 扩展代码以使用从2到你的Mac性能CPU核心数的线程数，相应地划分图像生成工作。在你的报告中假设加速比是否与线程数成线性关系？为什么（或不为什么）？

3. 在 `workerThreadStart()` 的开始和结束处插入计时代码，测量每个线程的工作时间。

4. 修改工作到线程的映射，在Mandelbrot集合上尽可能提高加速比。你不可以使用线程间的同步。在你的报告中描述你的并行化方法，并报告使用与性能CPU核心数相同线程数时获得的最终加速比。

5. 现在用2倍性能CPU核心数的线程运行改进后的代码。性能是否明显优于使用等于核心数的线程数？为什么或为什么不？

## 程序2：使用SIMD Intrinsics向量化代码（20分）

查看 `prog2_vecintrin/main.cpp` 中的 `clampedExpSerial` 函数。使用CS149的"伪向量intrinsics"（`CS149intrin.h`）实现向量化版本。

> **注意：** ARM版本的作业中，真实SIMD intrinsics使用Neon而非AVX2。但程序2使用的是CS149的伪向量intrinsics，与架构无关。

**需要做的事情：**

1. 在 `clampedExpVector` 中实现向量化版本。

2. 运行 `./myexp -s 10000` 并扫描向量宽度从2到16，记录向量利用率。

3. **额外加分（1分）：** 在 `arraySumVector` 中实现 `arraySumSerial` 的向量化版本。

## 程序3：使用ISPC并行生成分形图（20分）

与程序1类似，但使用ISPC语言结构来实现更大的加速比。

**需要做的事情：**

1. **重要：** 编译前需要修改Makefile，将 `avx2-i32x8` 改为 `neon-i32x8`，将 `x86-64` 改为 `aarch64`。ISPC编译器当前配置为生成8-wide Neon向量指令。根据你对ARM CPU的了解，期望的最大加速比是多少？

2. 运行 `mandelbrot_ispc --tasks`。观察到的加速比是多少？

3. 通过更改任务数来提高性能。

4. **额外加分（2分）：** 线程抽象和ISPC任务抽象之间的区别是什么？

## 程序4：迭代`sqrt`（15分）

计算2000万个随机数的平方根。

> **注意：** 同样需要修改Makefile，将 `avx2-i32x8` 改为 `neon-i32x8`，将 `x86-64` 改为 `aarch64`。

**需要做的事情：**

1. 构建并运行sqrt，报告加速比。

2. 构造最大化加速比的输入。

3. 构造最小化加速比的输入。

4. **额外加分（最多2分）：** 使用[Arm Neon Intrinsics](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html)手动编写 `sqrt` 函数。

## 程序5：BLAS `saxpy`（10分）

> 此程序在ARM上的操作与x86版本相同。正常编译运行即可。

**需要做的事情：**

1. 编译运行，观察加速比。

2. **额外加分（1分）：** 解释内存带宽计算。

3. **额外加分：** 提高性能。

## 程序6：让K-Means更快（15分）

**需要做的事情：**

1. 下载数据集：
   ```bash
   # 如果有Stanford账号：
   scp [你的SUNetID]@myth[51-66].stanford.edu:/afs/ir.stanford.edu/class/cs149/data/data.dat ./data.dat
   ```
   > **⚠️ 本地运行：** 如果没有Stanford账号，需要寻找替代数据获取方式或使用测试数据验证代码逻辑。可以参考[主README中文版](README_zh.md)中的说明。

2. 安装依赖并运行可视化。

3. 使用计时功能定位性能瓶颈。

4. 改进实现，达到约2.1倍以上的加速比。

## 推荐阅读

ISPC的创建者Matt Pharr写的[The story of ispc](https://pharr.org/matt/blog/2018/04/30/ispc-all)是CS149学生必读的文章。

## 提交说明

> **⚠️ 本地运行说明：** 
> - 不要提交在本地ARM机器上运行的代码到Gradescope（原作业只在Myth机器上运行代码）
> - 在你的实验报告中说明使用的ARM机器型号和配置
> - 将ARM的发现粘贴在Myth实验报告之后

## 资源和备注

- ISPC文档：<http://ispc.github.io/>
- ARM Neon intrinsics参考：<https://developer.arm.com/architectures/instruction-sets/intrinsics/>
- [Arm Neon Intrinsics Guide](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html) 非常有用
