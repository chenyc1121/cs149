# 作业4：编程机器学习加速器（本地学习版）

> **原作业**要求AWS Trainium2硬件（`trn2.3xlarge`实例），无法本地运行。
> **本文档将作业重新设计为本地可执行的练习**，使用纯NumPy/PyTorch教授相同的核心概念。

---

## 你将从中学到什么

Trainium2 作业教的核心概念**不依赖特定硬件**，这些技能在任何AI编译器/框架开发中都是通用的：

| 概念 | 在哪学 | 本地可运行？ |
|------|--------|-------------|
| 软件管理的片上存储 & DMA 传输 | 第1部分 → NumPy模拟 | ✅ |
| 循环分块（tiling）与 tile 尺寸约束 | 第1部分 → NumPy模拟 | ✅ |
| 卷积→矩阵乘法映射 | 第2部分 → `conv2d_numpy.py` | ✅ |
| 算子融合（conv + maxpool）减少内存 | 第2部分 → NumPy实现 | ✅ |
| 性能分析与瓶颈定位 | 第2部分 → Python profiler | ✅ |

---

## 环境配置（本地）

```bash
pip install numpy torch
```

不需要任何AWS账号、不需要GPU、不需要Trainium。

---

## 第1部分：理解加速器编程模式

> **学习目标：** 理解为什么加速器需要软件管理的内存层级、DMA传输和tiling。
> **方法：** 阅读提供的NKI代码 → 用NumPy重写相同逻辑 → 对比不同策略的性能差异。

代码位于 `part1/kernels.py`。虽然NKI代码无法运行，但逻辑清晰可读——NKI是Python DSL，内核逻辑和普通Python循环结构非常相似。

### 1.1 核心概念：分区维度（Partition Dimension）

Trainium 的硬件约束：一次最多处理 **128** 个元素并行（partition dimension ≤ 128）。这与SIMD宽度限制是同一类问题——你在作业1中已经见过了。

阅读 `kernels.py` 中的 `vector_add_naive`（第30-50行）：它只能处理 ≤128 的向量，因为整个向量被一次性加载到SBUF。

### 1.2 练习A：用NumPy模拟Tiled向量加法

阅读 `vector_add_tiled`（第58-89行）的逻辑，然后用NumPy实现等价的tiled向量加法：

```python
# 在 part1/ 目录下创建 local_exercises.py，实现以下函数：

import numpy as np

def vector_add_tiled_numpy(a: np.ndarray, b: np.ndarray, row_chunk: int) -> np.ndarray:
    """
    模拟 NKI 的 tiled vector add。
    - row_chunk 模拟了每次 DMA 传输加载的元素数（对应 NKI 中的 ROW_CHUNK）
    - 每次循环处理 row_chunk 个元素（模拟一次 SBUF tile 的加载→计算→存储）
    
    思考：row_chunk=1 vs row_chunk=128，哪个更快？为什么？
    """
    M = len(a)
    out = np.zeros_like(a)

    for m in range(0, M, row_chunk):
        # 模拟从 HBM 加载一个 tile 到 SBUF
        a_tile = a[m : m + row_chunk]
        b_tile = b[m : m + row_chunk]
        # 计算
        res = a_tile + b_tile
        # 模拟将结果从 SBUF 存回 HBM
        out[m : m + row_chunk] = res

    return out
```

**实验 & 记录：**
1. 对长度为 25600 的向量，分别测试 `row_chunk=1` 和 `row_chunk=128`，用 `timeit` 测量性能。
2. 观察到了什么？如何用"摊销DMA传输开销"来解释？
3. 试试 `row_chunk=256` —— 为什么在真实Trainium上会报错？（提示：partition dimension上限）

### 1.3 练习B：用NumPy模拟Streaming（2D Tiling）

阅读 `vector_add_stream`（第97-134行）。关键创新：将1D向量reshape为2D `(128, FREE_DIM)`，每次DMA传输加载一个2D tile而非1D chunk，大幅减少传输次数。

```python
def vector_add_stream_numpy(a: np.ndarray, b: np.ndarray, free_dim: int) -> np.ndarray:
    """
    模拟 NKI 的 streaming vector add。
    
    - 将向量 reshape 为 (128, M//128)，然后按 free_dim 列块处理
    - 每次循环加载 (128, free_dim) 的tile → 相当于一次 DMA 传输
    - 总 DMA 传输次数 = M / (128 * free_dim)
    
    思考：free_dim=2 vs free_dim=2000，DMA传输次数分别是多少？为什么free_dim=1000
    可能比free_dim=2000更快？（提示：pipelining 和 SBUF 容量压力）
    """
    PARTITION_DIM = 128
    M = len(a)
    assert M % PARTITION_DIM == 0

    a_2d = a.reshape(PARTITION_DIM, M // PARTITION_DIM)
    b_2d = b.reshape(PARTITION_DIM, M // PARTITION_DIM)
    out_2d = np.zeros_like(a_2d)

    for m in range(0, M // PARTITION_DIM, free_dim):
        a_tile = a_2d[:, m : m + free_dim]
        b_tile = b_2d[:, m : m + free_dim]
        out_2d[:, m : m + free_dim] = a_tile + b_tile

    return out_2d.reshape(M)
```

**实验 & 记录：**
1. 计算不同 `free_dim` 下的DMA传输次数（即循环迭代次数）。
2. 测量不同 `free_dim` 下的NumPy性能，记录在实验报告中。
3. 在真实硬件上，为什么更大的tile不一定更好？（SBUF容量 = 28 MiB，tile太大会占满SBUF导致无法pipeline）

### 1.4 练习C：用NumPy实现分块矩阵转置

阅读 `matrix_transpose`（第141-152行），然后用纯NumPy实现：

```python
def matrix_transpose_tiled_numpy(a: np.ndarray, tile_dim: int = 128) -> np.ndarray:
    """
    分块矩阵转置。
    
    - 将 M×N 矩阵分解为 tile_dim × tile_dim 的子矩阵
    - 每个子矩阵独立转置，放到输出的对应位置
    - 模拟了在加速器上处理超大矩阵（无法一次放入SBUF）的策略

    思考：这个实现与 np.transpose(a) 相比，哪种更"缓存友好"？为什么？
    """
    M, N = a.shape
    assert M % tile_dim == 0 and N % tile_dim == 0
    out = np.zeros((N, M), dtype=a.dtype)

    for i in range(0, M, tile_dim):
        for j in range(0, N, tile_dim):
            tile = a[i : i + tile_dim, j : j + tile_dim]
            out[j : j + tile_dim, i : i + tile_dim] = tile.T

    return out
```

**实验 & 记录：**
1. 在 4096×4096 矩阵上对比你的tiled实现和 `np.transpose()` 的性能。
2. 尝试不同的 `tile_dim`（16, 32, 64, 128, 256），哪个最快？为什么？
3. 用 `perf stat -e cache-misses,cache-references` 分析不同tile size的缓存未命中率。与你对问题2的回答一致吗？

---

## 第2部分：从卷积到融合内核（核心）

> **学习目标：** 理解卷积→矩阵乘法映射、算子融合如何减少内存占用。
> **方法：** 阅读算法描述 → 用NumPy逐层实现 → 融合 → 性能分析。

**关键文件：** `part2/conv2d_numpy.py` 包含了纯NumPy/PyTorch的参考实现，**可以直接运行**。

### 2.0 验证环境

```bash
cd part2/
python3 -c "
from conv2d_numpy import conv2d_cpu_torch, conv_numpy, maxpool_numpy
import numpy as np

# 生成随机输入
X = np.random.rand(4, 128, 32, 16).astype(np.float32)
W = np.random.rand(128, 128, 3, 3).astype(np.float32)
bias = np.zeros(128, dtype=np.float32)

# 测试PyTorch参考实现
out_torch = conv2d_cpu_torch(X, W, bias, pool_size=1)
print(f'PyTorch output shape: {out_torch.shape}')

# 测试纯NumPy实现
out_numpy = conv_numpy(X, W, bias)
print(f'NumPy output shape: {out_numpy.shape}')
print(f'Results match: {np.allclose(out_torch.numpy(), out_numpy, atol=1e-5)}')
"
```

如果以上代码正确输出，说明你的本地环境已就绪。

### 2.1 理解卷积→矩阵乘法映射算法

这是本作业最核心的算法思想。阅读以下伪代码（在 `part2/conv2d.py` 的注释和原README中也有描述）：

```
给定:
  输入 X:    shape (InChannels, Height, Width)
  滤波器 W:  shape (OutChannels, InChannels, FilterH, FilterW)

算法:
  将 X 展平为 (InChannels, Height * Width)
  
  for i in range(FilterH):
      for j in range(FilterW):
          # 将输入按 (i,j) 偏移，对齐当前滤波器位置
          input_shifted = shift(X_flat, offset=(i, j))
          
          # 取滤波器在位置 (i,j) 的切片: (InChannels, OutChannels)
          weight_slice = W[:, :, i, j]  # 实际是 (OutChannels, InChannels)
          
          # 矩阵乘法: (OutChannels, InChannels) × (InChannels, H*W)
          #       → (OutChannels, H*W)
          output += matmul(weight_slice, input_shifted)
```

**关键观察：** 
- 一次卷积被分解为 `FilterH × FilterW` 次独立的矩阵乘法
- 每次矩阵乘法的shape是 `(OutChannels, InChannels) × (InChannels, H*W)`
- 这些矩阵乘法可以独立并行
- 在Trainium上，`InChannels` 和 `OutChannels` 必须被128整除（tile约束）

### 2.2 练习D：实现卷积→矩阵乘法的NumPy版本

在 `part2/` 下创建 `local_conv2d.py`：

```python
import numpy as np

def conv2d_via_matmul_numpy(X, W, bias):
    """
    用"卷积→矩阵乘法"映射实现 Conv2D。
    
    参数:
        X:    (Batch, InChannels, H, W)
        W:    (OutChannels, InChannels, FilterH, FilterW)
        bias: (OutChannels,)
    
    返回:
        out:  (Batch, OutChannels, H_out, W_out)
    
    这是训练/推理框架底层会用到的实现策略。
    """
    B, C_in, H, W_img = X.shape
    C_out, _, Fh, Fw = W.shape
    H_out = H - Fh + 1
    W_out = W_img - Fw + 1
    
    # 将输入展平为 (Batch, InChannels, H*W)
    X_flat = X.reshape(B, C_in, H * W_img)
    
    out = np.zeros((B, C_out, H_out, W_out), dtype=X.dtype)
    
    for i in range(Fh):
        for j in range(Fw):
            # 取滤波器在 (i,j) 的切片: shape (OutChannels, InChannels)
            W_slice = W[:, :, i, j]
            
            # 对每个batch和每个输出位置，构造移位的输入
            for b in range(B):
                for h in range(H_out):
                    for w in range(W_out):
                        # 输入中对应 (h+i, w+j) 位置的列
                        col_idx = (h + i) * W_img + (w + j)
                        x_vec = X_flat[b, :, col_idx]  # shape (InChannels,)
                        
                        # 矩阵-向量乘法: (OutChannels, InChannels) @ (InChannels,)
                        # → (OutChannels,)
                        out[b, :, h, w] += W_slice @ x_vec
    
    # 加 bias: (OutChannels,) → broadcast 到 (Batch, OutChannels, H_out, W_out)
    out += bias.reshape(1, -1, 1, 1)
    
    return out
```

**验证正确性：**
```python
from conv2d_numpy import conv_numpy

X = np.random.rand(2, 256, 16, 16).astype(np.float32)
W = np.random.rand(128, 256, 3, 3).astype(np.float32)
bias = np.random.rand(128).astype(np.float32)

out_mine = conv2d_via_matmul_numpy(X, W, bias)
out_ref = conv_numpy(X, W, bias)

print(f'Results match: {np.allclose(out_mine, out_ref, atol=1e-4)}')
```

### 2.3 练习E：性能对比——直接卷积 vs 矩阵乘法映射

用 `timeit` 对比你的 `conv2d_via_matmul_numpy` 和 `conv_numpy`（直接四重循环）：

```python
import timeit

# 对比小图像和大图像
for name, X, W, bias in [
    ("small",  np.random.rand(4, 128, 32, 16).astype(np.float32),
               np.random.rand(128, 128, 3, 3).astype(np.float32),
               np.random.rand(128).astype(np.float32)),
    ("large",  np.random.rand(4, 256, 224, 224).astype(np.float32),
               np.random.rand(256, 256, 3, 3).astype(np.float32),
               np.random.rand(256).astype(np.float32)),
]:
    t1 = timeit.timeit(lambda: conv_numpy(X, W, bias), number=5)
    t2 = timeit.timeit(lambda: conv2d_via_matmul_numpy(X, W, bias), number=5)
    print(f"{name}: direct={t1:.3f}s, matmul_mapped={t2:.3f}s")
```

**思考题：** 哪种方法在小图像上更快？在大图像上呢？为什么矩阵乘法映射在加速器上更优（即使NumPy上不一定更快）？

### 2.4 练习F：算子融合——减少中间内存（核心练习）

这是作业第2部分最核心的概念。对比融合和非融合版本的内存占用：

```python
def conv2d_then_maxpool_unfused(X, W, bias, pool_size=2):
    """
    非融合版本：先做完整卷积，将结果写入内存，再做maxpool。
    
    中间结果大小: (Batch, OutChannels, H_out, W_out)
    对于 224×224 输入，3×3 filter，256通道:
      中间结果 = 4 × 256 × 222 × 222 × 4 bytes ≈ 200 MB
    """
    conv_out = conv2d_via_matmul_numpy(X, W, bias)  # 写入内存
    return maxpool_numpy(conv_out, pool_size)         # 从内存读取


def conv2d_maxpool_fused(X, W, bias, pool_size=2):
    """
    融合版本：在计算卷积输出像素的同时，立即做maxpool。
    
    不需要存储完整的 conv_out 中间结果！
    只需要存储 pool_size×pool_size 的小窗口就足够了。
    
    节省的内存 ≈ 整个卷积输出的大小
    """
    B, C_in, H, W_img = X.shape
    C_out, _, Fh, Fw = W.shape
    H_out = H - Fh + 1
    W_out = W_img - Fw + 1
    
    H_pool = H_out // pool_size
    W_pool = W_out // pool_size
    
    out = np.zeros((B, C_out, H_pool, W_pool), dtype=X.dtype)
    
    for b in range(B):
        for c_out in range(C_out):
            for ph in range(H_pool):
                for pw in range(W_pool):
                    # 这个 pool 窗口对应的卷积输出区域的起始位置
                    h_start = ph * pool_size
                    w_start = pw * pool_size
                    
                    pool_vals = np.zeros((pool_size, pool_size), dtype=X.dtype)
                    
                    for pi in range(pool_size):
                        for pj in range(pool_size):
                            h_conv = h_start + pi
                            w_conv = w_start + pj
                            
                            # 计算这一个卷积像素
                            val = bias[c_out]
                            for c_in in range(C_in):
                                for fi in range(Fh):
                                    for fj in range(Fw):
                                        val += (X[b, c_in, h_conv + fi, w_conv + fj]
                                                * W[c_out, c_in, fi, fj])
                            pool_vals[pi, pj] = val
                    
                    # 取 pool 窗口内的最大值
                    out[b, c_out, ph, pw] = np.max(pool_vals)
    
    return out
```

**实验 & 记录：**
1. 对小图像（Batch=1, 128通道, 32×16），用 `tracemalloc` 或手动计算对比融合和非融合版本的峰值内存占用。
2. 对大图像（Batch=1, 256通道, 224×224），计算：如果不融合，中间卷积结果占用多少内存？融合后节省了多少？
3. 融合版本的缺点是什么？（提示：代码复杂度、编译器优化的难度）

### 2.5 练习G：用PyTorch验证和扩展

`conv2d_numpy.py` 中的 `conv2d_cpu_torch` 可以直接运行：

```python
from conv2d_numpy import conv2d_cpu_torch
import torch
import numpy as np

# 测试所有配置组合
configs = [
    # (use_bias, pool_size, dtype)
    (False, 1, np.float32),
    (True, 1, np.float32),
    (True, 2, np.float32),
    (True, 2, np.float16),
]

for use_bias, pool_size, dtype in configs:
    X = np.random.rand(4, 128, 32, 16).astype(dtype)
    W = np.random.rand(128, 128, 3, 3).astype(dtype)
    bias = np.random.rand(128).astype(dtype) if use_bias else np.zeros(128, dtype=dtype)
    
    out = conv2d_cpu_torch(X, W, bias, pool_size=pool_size)
    print(f"bias={use_bias}, pool={pool_size}, dtype={dtype.__name__}: "
          f"output shape={out.shape}")
```

### 2.6 练习H：理解NKI内核代码中的Tiling逻辑（阅读+分析）

打开 `part2/conv2d.py`，阅读 `fused_conv2d_maxpool` 的骨架代码（第38-77行）。虽然无法运行，但可以分析以下问题并在报告中回答：

1. 代码中 `c_in_pmax = nl.tile_size.pmax`（=128），`n_tiles_c_in = in_channels // c_in_pmax`——这是在做什么分块？为什么 `in_channels` 维度需要分块？
2. 第58行 `assert nl.tile_size.gemm_moving_fmax >= out_width` 检查了什么约束？为什么输出宽度必须 ≤ 512？
3. 如果你来实现 `for b in nl.affine_range(batch_size):` 内部的逻辑，你会如何组织循环嵌套顺序？在报告中画出你的循环嵌套结构。
4. 如果 `pool_size=2`，maxpool 操作应该在循环嵌套的哪一层执行才能实现真正的融合（不写出完整中间结果）？

---

## 学习检查清单

完成以下项目即表示你掌握了本作业的核心内容：

- [ ] 能用NumPy写出tiled向量加法，并解释为什么更大的tile更好（以及什么时候不好）
- [ ] 能用NumPy写出分块矩阵转置，理解tile size与缓存命中率的关系
- [ ] 能手写卷积→矩阵乘法映射的NumPy实现，并验证正确性
- [ ] 能写出融合conv+maxpool的NumPy实现
- [ ] 能计算并对比融合vs非融合版本的内存占用
- [ ] 能阅读NKI内核代码（`conv2d.py`），识别其中的tiling策略和循环结构
- [ ] 能解释为什么加速器需要软件管理的内存层级（与自动缓存相比的优劣）

---

## 与课程其他作业的关联

| 本作业概念 | 在哪里再次出现 |
|-----------|---------------|
| 循环分块 (tiling) | `cs149gpt` Part 2/4: blocked & flash attention |
| 算子融合减少内存 | `cs149gpt` Part 3: fused attention (O(N²)→O(N)) |
| SIMD宽度/partition限制 | `asst1` Program 2 & 3: vector width constraints |
| 软件管理的内存 | `asst3` CUDA shared memory (也是软件显式管理的！) |
| 矩阵乘法映射 | 这是ML编译器（TVM, Triton, XLA）的基础操作 |

---

## 如果你有AWS访问权限

按照 `cloud_readme.md` 设置Trainium实例，然后可以运行原始的NKI内核代码和 `test_harness.py`。此时本文档的NumPy练习应该已经让你完全理解了算法，你只需要专注于Trainium特定的性能调优（DMA pipelining, tile size选择, MFU优化）。

---

## 参考资源

- [NKI教程](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/tutorials.html) — 阅读概念，不一定要跑代码
- [NKI API参考](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/index.html)
- [卷积动画可视化](https://github.com/vdumoulin/conv_arithmetic) — 直观理解卷积操作
- [Flash Attention论文](https://arxiv.org/abs/2205.14135) — 融合优化的经典案例
- [TVM教程](https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html) — 另一个学习算子调度优化的好资源
