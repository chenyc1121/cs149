# 作业3：简单的CUDA渲染器

**截止日期：10月30日周四 11:59PM PST**

**总分100分**

![效果图](handout/teaser.jpg?raw=true)

## 概述

在本作业中，你将用CUDA编写一个绘制彩色圆圈的并行渲染器。虽然这个渲染器非常简单，但并行化渲染器需要你设计并实现能够高效并行构建和操作的数据结构。这是一个具有挑战性的作业，建议尽早开始。**说真的，请尽早开始。** 祝你好运！

## 环境配置

> **⚠️ 本地运行说明：** 原作业要求在AWS上使用GPU虚拟机（NVIDIA T4 GPU）运行。如果你有本地NVIDIA GPU，可以完全在本地运行。你需要确保安装了CUDA Toolkit。如果没有GPU，无法运行CUDA代码，但仍可以学习代码逻辑和参考实现（`refRenderer.cpp`）。

### 步骤1：检查CUDA环境

```bash
# 检查是否有NVIDIA GPU
nvidia-smi

# 检查CUDA编译器
nvcc --version
```

如果没有CUDA，需要安装：
- **Ubuntu/Debian：** `sudo apt install nvidia-cuda-toolkit`
- 或从[NVIDIA官网](https://developer.nvidia.com/cuda-downloads)下载

### 步骤2：克隆代码

```bash
git clone https://github.com/stanford-cs149/asst3
```

### 参考资源

- [CUDA C编程指南（PDF）](http://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)或[Web版](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Udacity CS344：CUDA并行编程导论](https://www.udacity.com/blog/2014/01/update-on-udacity-cs344-intro-to.html)（免费课程）
- [C++ Super-FAQ](https://isocpp.org/faq)

> **注意：** 你的本地GPU可能与T4的compute capability 7.5不同。在Makefile中调整 `-arch` 参数以匹配你的GPU。使用 `nvidia-smi --query-gpu=compute_cap --format=csv` 查看你的GPU计算能力。

## 第1部分：CUDA热身1 - SAXPY（5分）

在 `/saxpy` 目录中重新实现作业1的SAXPY函数的CUDA版本。

需要做的事情：
- 在 `saxpy.cu` 的 `saxpyCuda` 函数中完成SAXPY实现
- 分配设备全局内存，将数据复制到GPU，执行计算，复制回主机
- **添加计时代码**：测量仅内核执行时间（使用 `cudaDeviceSynchronize()`）
- 比较内核执行时间与含数据传输的总时间

**问题1：** 与CPU串行SAXPY相比，CUDA版本的性能如何？

**问题2：** 比较两组计时器的结果差异。观察到的带宽值与机器各组件的报告带宽大致一致吗？

## 第2部分：CUDA热身2 - 并行前缀和（10分）

实现 `find_repeats` 函数：给定整数列表 `A`，返回所有满足 `A[i] == A[i+1]` 的索引 `i` 的列表。

首先实现并行**独占前缀和**（exclusive prefix sum），然后基于它实现 `find_repeats`。

在 `scan/scan.cu` 中：
1. 实现 `exclusive_scan` 函数（使用课件中的upsweep/downsweep算法）
2. 实现 `find_repeats` 函数

运行 `./checker.py scan` 和 `./checker.py find_repeats` 检查正确性和性能。

**性能提示：** 不要为每个外层循环迭代启动N个CUDA线程——只为内层并行循环的每次迭代启动一个线程。

## 第3部分：简单圆形渲染器（85分）

`/render` 目录包含渲染器的两个版本：
- **`refRenderer.cpp`：** 顺序的单线程C++参考实现
- **`cudaRenderer.cu`：** **不正确的**并行CUDA实现（需要你修复）

### 渲染器概述

基本串行算法：
```
清空图像
对每个圆：更新位置和速度
对每个圆：
    计算屏幕边界框
    对边界框内所有像素：
        如果像素中心在圆内：
            计算颜色贡献
            混合到输出图像
```

渲染器绘制**半透明**圆，使用alpha混合。**顺序很重要！** 圆必须按应用程序提供的深度顺序绘制。

### CUDA渲染器要求

你的并行CUDA渲染器必须保持两个不变性：

1. **原子性：** 所有图像更新操作必须是原子的（读取RGBA → 混合 → 写回）
2. **顺序：** 对同一像素的更新必须按圆的输入顺序进行。不同像素之间的圆没有顺序要求。

### 你需要做的事情

**编写最快且正确的CUDA渲染器实现。**

建议步骤：
1. 首先重写CUDA起始代码，使其在并行运行时逻辑正确（建议使用不需要锁或同步的方法）
2. 然后确定解决方案的性能问题
3. 此时真正的思考才开始...

**测试命令：**
```bash
# 运行CUDA渲染器
./render -r cuda rand10k

# 运行CPU参考实现
./render -r cpuref rand10k

# 正确性检查
./render -r cuda --check rand10k

# 性能评分
./checker.py
```

可用场景：`rgb`, `rgby`, `rand10k`, `rand100k`, `rand1M`, `biglittle`, `littlebig`, `pattern`, `micro2M`, `bouncingballs`, `fireworks`, `hypnosis`, `snow`, `snowsingle`

### 评分指南

- 实验报告：18分
- 并行前缀和：10分
- 渲染器实现：72分（8个场景，每个9分 = 2分正确性 + 7分性能）
- 额外加分：最多10分

## 提示和技巧

- 两个并行维度：跨像素并行和跨圆并行（需尊重重叠圆的顺序要求）
- `circleBoxTest.cu_inl` 中的圆-框相交测试是你的好帮手
- `exclusiveScan.cu_inl` 中的共享内存前缀和可能很有用
- 考虑使用寄存器中的本地累加器，减少全局内存写入
- 首先考虑确保顺序，然后再处理原子性问题
- 对于大量圆的测试（`rand1M`、`micro2M`），注意临时结构的全局内存分配

### 捕获CUDA错误

使用提供的宏包装CUDA调用以捕获错误：
```cpp
#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif
```

## 提交

> **⚠️ 本地运行说明：** 提交前确保代码能在你的本地GPU上正确编译和运行。

1. 提交实验报告 `writeup.pdf`
2. 运行 `sh create_submission.sh` 生成zip提交

## 资源

- [CUDA C编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA T4规格](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)
- [Thrust库](http://thrust.github.io/)
