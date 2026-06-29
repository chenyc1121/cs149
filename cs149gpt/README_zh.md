# 作业4（另一版本）：NanoGPT149

**截止日期：12月4日周一 11:59pm PST**

**总分100分 + 12分额外加分**

## 概述

在本作业中，你将实现并优化一个基于Transformer的深度神经网络的关键组件，该网络能合成莎士比亚风格的文本。你将用C++实现注意力层，专注于提高算术强度、减少内存占用以及利用CPU上的多核和SIMD并行性。

作业将带你经历以下过程：
- 简单串行注意力实现
- 循环分块优化（blocked matrix multiply）
- 循环融合（fused attention）
- Flash Attention算法
- 可选的ISPC向量化优化

## 环境配置

> **⚠️ 本地运行说明：** 原作业需要SSH访问Stanford共享机器集群。你已拥有本地代码，可以完全在本地运行。需要确保以下依赖已安装。

### 依赖要求

```bash
# Python依赖
pip install torch numpy

# ISPC编译器（用于额外加分的向量化优化）
# 参见作业1中的ISPC安装说明
```

### 克隆代码（如果你还没有）

```bash
git clone https://github.com/stanford-cs149/cs149gpt.git
```

### 验证环境

```bash
# 测试推理是否正常
python3 gpt149.py part0 --inference -m shakes128
```

首次运行时会进行编译（"Compiling code into a PyTorch module..."），之后你会看到生成的莎士比亚风格文本。

> **编译挂起问题：** 如果编译挂起，尝试删除锁文件：
> ```bash
> rm ~/.cache/torch_extensions/py310_cpu/custom_module/lock
> ```

## 注意力机制概述

Transformer的关键组件是**注意力机制**（attention mechanism）。输入是三个矩阵 `Q`、`K`、`V`（查询、键、值），每个大小为 `N×d`。

注意力计算包括三个步骤：

1. **矩阵乘法：** $S = QK^T$（产生 `N×N` 矩阵）
2. **Softmax：** $P = \text{softmax}(\text{S的每一行})$
3. **矩阵乘法：** $O = PV$（产生 `N×d` 输出矩阵）

**重要：** 在实际实现中，这些矩阵是4D张量（增加了batch和head维度）。

## 热身：访问张量（3分）

在 `module.cpp` 中实现 `fourDimRead` 和 `fourDimWrite` 函数。

4D张量的访问公式推广自2D访问公式 `[i * num_cols + j]`。

**测试：**
```bash
python3 gpt149.py 4Daccess
```

**实验报告问题：** 简要描述4D张量/数组在内存中的布局。为什么选择这种约定？它如何利用硬件？

## 第1部分：简单（但效率不高）的注意力实现（10分）

在 `module.cpp` 中实现 `myNaiveAttention`：
- 对每个batch、每个head：
  - 计算 `Q × K^T`（注意循环顺序以避免显式转置）
  - 对每行计算softmax（exp → 求和 → 除法）
  - 计算 `softmax结果 × V` 并存入 `O`

**测试：**
```bash
python3 gpt149.py part1
python3 gpt149.py part1 -N <值>    # 测试不同的N值
python3 gpt149.py part1 --inference -m shakes128   # 用你的注意力层生成文本
```

正确实现的CPU时间应接近参考值（230ms左右，15ms容差内）。

## 第2部分：分块矩阵乘法和未融合Softmax（20分）

在 `module.cpp` 中实现 `myUnfusedAttentionBlocked`：

使用循环分块（loop blocking/tiling）优化矩阵乘法。将大矩阵分解为适合缓存的小子矩阵（tile），在逐出缓存前完成子矩阵的处理。

**需要对 `QK^t` 和 `PV` 两个矩阵乘法都使用分块。**

处理"剩余"tile：当tile大小不能整除N时，使用 `min(tile_size, N - tileIndex * tileSize)` 确定subtile大小。

**测试：**
```bash
python3 gpt149.py part2
python3 gpt149.py part2 -N <值>
```

**实验报告问题：**
- 分享你尝试的不同tile大小及对应的性能时间。最优tile大小是多少？解释原因。
- 对于 $Q$ (N×d) 和 $K^T$ (d×N) 的矩阵乘法，第2部分与第1部分的DRAM访问比率是多少？

## 第3部分：融合注意力（25分）

在 `module.cpp` 中实现 `myFusedAttention`：

**核心思想：** 不需要先计算完整的 `N×N` 矩阵再做softmax。而是：
- 计算 `Q × K^T` 的一行
- 立即对该行做softmax
- 立即将softmax后的行乘以 `V`
- 重用同一个 `N×1` 临时向量处理下一行

这样内存占用从 $O(N^2)$ 降低到 $O(N)$。

**使用OpenMP并行化：**
```cpp
#pragma omp parallel for collapse(3)
for ()
    for ()
        for ()
```

注意：每个OpenMP线程应有自己的 `N×1` 临时数组副本以避免竞态条件。

**测试：**
```bash
python3 gpt149.py part3
python3 gpt149.py part3 -N <值>
```

**实验报告问题：**
- 为什么第3部分的内存使用量比第1、2部分大幅减少？
- 注释掉 `#pragma omp` 语句后CPU时间变为多少？为什么融合注意力比第1部分更容易充分利用多线程？

## 第4部分：Flash Attention（35分）

在 `module.cpp` 中实现 `myFlashAttention`：

Flash Attention通过将softmax分解为块来进一步优化。对于每个块：
- 计算 `Q_block (Br×d) × K^T_block (d×Bc) → QK^T_block (Br×Bc)`
- 计算该块的局部softmax
- 乘以 `V_block (Bc×d)` 并累加到 `O_block (Br×d)`

**关键：** 使用 `running_max` 和 `running_sum` 来正确组合不同块的softmax结果。

参考伪代码（见原README中的FlashAttention伪代码图）。`Br` 和 `Bc` 可通过命令行参数设置。

**测试：**
```bash
python3 gpt149.py part4
python3 gpt149.py part4 -br 128 -bc 512
python3 gpt149.py part4 -N <值> -br <值> -bc <值>
```

**注意：** 第4部分仅检查正确性，不检查性能。

**实验报告问题：**
- 第4部分的内存使用量与前几部分相比如何？为什么？
- 第4部分的性能比前几部分慢。我们是否已完全优化了第4部分？还可以做哪些性能改进？

## 额外加分：使用ISPC Intrinsics向量化（每部分3分，共12分）

在 `module.ispc` 中编写ISPC函数，对矩阵乘法和行求和等操作进行向量化。

```bash
ispc -O3 --target=avx2-i32x8 --arch=x86-64 --pic module.ispc -h module_ispc.h -o module_ispc.o
```

> **ARM Mac用户：** 将target改为 `neon-i32x8`，arch改为 `aarch64`。

然后在 `module.cpp` 中取消以下两行的注释：
```cpp
#include "module_ispc.h"
using namespace ispc;
```

## 评分概览（100分 + 12分EC）

| 项目 | 分数 |
|------|------|
| `fourDimRead` | 1.5 |
| `fourDimWrite` | 1.5 |
| `myNaiveAttention` | 10 |
| `myUnfusedAttentionBlocked` | 20 |
| `myFusedAttention` | 25 |
| `myFlashAttention` | 35 |
| 实验报告问题 | 7 |
| 额外加分（向量化，每部分3分） | 最多12 |

## 实验报告问题汇总

1. **热身：** 4D张量在内存中的布局？为什么选择这种约定？
2. **第2部分：** 尝试的tile大小及性能？最优tile大小及原因？DRAM访问比率？
3. **第3部分：** 内存使用量为何大幅减少？注释OpenMP后的CPU时间？融合注意力为何更易多线程化？
4. **第4部分：** 内存使用量比较？还可以做哪些性能改进？

## 提交

- `module.cpp`
- `module.ispc`（如果完成了额外加分）
- `writeup.pdf`

> **⚠️ 本地运行说明：** 所有开发和测试均可在本地完成。在实验报告中注明使用的CPU型号、核心数、缓存大小等信息。
