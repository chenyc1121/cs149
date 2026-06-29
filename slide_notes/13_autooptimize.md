# Lecture 13: 自动优化技术

## 1. 课程概述

### 1.1 今日主题

本讲探讨提升性能优化生产力的机制和技术，包括：

- **思路一：提升抽象层次（Raise Level of Abstraction）**——让程序员在更高的抽象层面上表达计算，而不是直接编写底层优化代码。
- **思路二：智能搜索（Intelligent Search）**——在巨大的优化方案空间中自动搜索最优实现。
- **思路三（新兴）：利用现代大语言模型（LLM）**——借助LLM的推理和代码生成能力来完成自动优化。

### 1.2 为什么需要自动优化？

- CS149培养的程序员（懂并行计算、SIMD、缓存优化的专家）非常稀缺。
- 使用C++、ISPC、CUDA等语言进行性能优化的**生产力很低**。写一次高性能代码需要大量手工调优，且针对不同硬件需要重新优化。
- 手工优化的代码可读性极差，难以理解和维护。

---

## 2. 性能-生产力-通用性三角权衡

### 2.1 理想并行编程语言的铁三角

任何编程语言都需要在这三个维度之间做出权衡：

- **性能（Performance）**：程序运行速度
- **生产力（Productivity）**：编写代码的难易程度和速度
- **通用性（Generality）**：语言的适用范围

三者难以同时兼得。通用语言（如C++、CUDA）通常性能好但生产力低；高生产力语言（如Python）通常性能较差。

### 2.2 领域特定语言（DSL）的定位

DSL（Domain-Specific Language）试图在特定领域内同时获得高性能、高生产力和足够的通用性：

- **牺牲通用性**以换取在特定领域的**高性能**和**高生产力**。
- 针对特定应用领域设计，具有受限的表达能力。
- 通常是**高级语言**、**声明式**和**确定性的**。

---

## 3. 领域特定编程系统

### 3.1 核心理念

**提升程序表达的抽象层次**：

1. 目标一：快速编写针对目标机器的高性能程序。
2. 目标二：编写一次程序，在不同机器上都高效运行。

### 3.2 设计原则

- **引入特定于应用领域的高级编程原语**：
  - **高生产力**：直观易用，跨机器可移植，原语对应目标领域中频繁使用的行为模式。
  - **高性能**：系统利用领域知识提供高效的、优化的实现。
  - 给定一台机器，系统知道该使用什么算法、什么样的并行化策略。

- **代价**：丧失通用性和完备性。

---

## 4. Halide：面向图像处理的领域特定语言

Halide是由Jonathan Ragan-Kelley、Andrew Adams等人开发的一种用于图像处理的领域特定语言（SIGGRAPH 2012, PLDI 2013）。

### 4.1 工业应用

- Google手机上的相机处理流水线（HDR+、人像模式的某些方面等）。
- Instagram、Adobe等公司的图像处理应用。
- Google HDR+流水线包含超过2000个Halide函数。

---

## 5. 图像处理优化案例：3x3 Box Blur

### 5.1 原始3x3模糊代码

```c
int WIDTH = 1024;
int HEIGHT = 1024;
float input[(WIDTH+2) * (HEIGHT+2)];
float output[WIDTH * HEIGHT];
float weights[] = {1.f/9, 1.f/9, 1.f/9,
                   1.f/9, 1.f/9, 1.f/9,
                   1.f/9, 1.f/9, 1.f/9};
for (int j=0; j<HEIGHT; j++) {
  for (int i=0; i<WIDTH; i++) {
    float tmp = 0.f;
    for (int jj=0; jj<3; jj++)
      for (int ii=0; ii<3; ii++)
        tmp += input[(j+jj)*(WIDTH+2) + (i+ii)] * weights[jj*3 + ii];
    output[j*WIDTH + i] = tmp;
  }
}
```

**计算量**：每个像素需要 9 次乘加操作，整张图像的计算量为 **9 x WIDTH x HEIGHT**。对于 N x N 的滤波器，计算量为 N^2 x WIDTH x HEIGHT。

---

### 5.2 两遍（Two-Pass）模糊

利用2D可分离滤波器（如box filter）的特性，将2D模糊分解为两个1D滤波操作：

- 第一步：水平方向模糊
- 第二步：垂直方向模糊

```c
int WIDTH = 1024;
int HEIGHT = 1024;
float input[(WIDTH+2) * (HEIGHT+2)];
float tmp_buf[WIDTH * (HEIGHT+2)];
float output[WIDTH * HEIGHT];
float weights[] = {1.f/3, 1.f/3, 1.f/3};

// 1D horizontal blur
for (int j=0; j<(HEIGHT+2); j++)
  for (int i=0; i<WIDTH; i++) {
    float tmp = 0.f;
    for (int ii=0; ii<3; ii++)
      tmp += input[j*(WIDTH+2) + i+ii] * weights[ii];
    tmp_buf[j*WIDTH + i] = tmp;
  }

// 1D vertical blur
for (int j=0; j<HEIGHT; j++) {
  for (int i=0; i<WIDTH; i++) {
    float tmp = 0.f;
    for (int jj=0; jj<3; jj++)
      tmp += tmp_buf[(j+jj)*WIDTH + i] * weights[jj];
    output[j*WIDTH + i] = tmp;
  }
}
```

**优势**：
- 计算量降为 **6 x WIDTH x HEIGHT**（对于N x N滤波器为 2N x WIDTH x HEIGHT）。
- 比2D模糊减少了计算量。

**代价**：
- 需要额外的中间缓冲区（WIDTH x (HEIGHT+2) 个float）。
- **算术强度（Arithmetic Intensity）降低了一半**。
- 对 tmp_buf 的加载/存储是额外开销。

---

### 5.3 局部性分析

**输入数据（input）的局部性**：
- 每个输入元素被复用3次（在一行的连续3次i循环迭代中立即使用）。
- 完美的缓存行为：数据只需加载一次，不会加载不必要的缓存行。

**中间数据（tmp_buf）的局部性**：
- 每个 tmp_buf 元素被复用3次，但访问间隔是三行图像数据。
- 如果缓存容量能够容纳三行数据，则数据只需加载一次。
- 完美的缓存行利用。

**内在带宽需求**：
- 应用程序必须读取每个输入图像元素，必须写入每个输出图像元素。
- 对 tmp_buf 的读写是可避免的——它们不是计算本身固有的需求。

---

### 5.4 分块（Chunked）两遍模糊 — 版本1

通过分块减少中间缓冲区大小，使中间数据完全驻留在缓存中：

```c
int WIDTH = 1024;
int HEIGHT = 1024;
float input[(WIDTH+2) * (HEIGHT+2)];
float tmp_buf[WIDTH * 3];   // 只需要3行中间缓冲区
float output[WIDTH * HEIGHT];
float weights[] = {1.f/3, 1.f/3, 1.f/3};

for (int j=0; j<HEIGHT; j++) {
  // Step 1: 生成3行tmp_buf（仅生成产出当前行输出所需的）
  for (int j2=0; j2<3; j2++)
    for (int i=0; i<WIDTH; i++) {
      float tmp = 0.f;
      for (int ii=0; ii<3; ii++)
        tmp += input[(j+j2)*(WIDTH+2) + i+ii] * weights[ii];
      tmp_buf[j2*WIDTH + i] = tmp;
    }

  // Step 2: 组合生成一行输出
  for (int i=0; i<WIDTH; i++) {
    float tmp = 0.f;
    for (int jj=0; jj<3; jj++)
      tmp += tmp_buf[jj*WIDTH + i] * weights[jj];
    output[j*WIDTH + i] = tmp;
  }
}
```

**特点**：
- 每行输出总计算量：3 x 3 x WIDTH + 3 x WIDTH = 12 x WIDTH。
- 整张图计算量：12 x WIDTH x HEIGHT（增加了计算量，因为每行重新计算水平模糊的3行）。
- 但 tmp_buf 完全适合缓存，缓存命中率高。
- 中间缓冲区只需要 (W x 3) 个元素。

---

### 5.5 分块两遍模糊 — 版本2（泛化）

以更大的 CHUNK_SIZE 为单位进行分块：

```c
int WIDTH = 1024;
int HEIGHT = 1024;
float input[(WIDTH+2) * (HEIGHT+2)];
float tmp_buf[WIDTH * (CHUNK_SIZE+2)];  // 块大小+额外的2行用于垂直滤波边界
float output[WIDTH * HEIGHT];
float weights[] = {1.f/3, 1.f/3, 1.f/3};

for (int j=0; j<HEIGHT; j+=CHUNK_SIZE) {
  // Step 1: 生成足够的tmp_buf行以产出CHUNK_SIZE行输出
  for (int j2=0; j2<CHUNK_SIZE+2; j2++)
    for (int i=0; i<WIDTH; i++) {
      float tmp = 0.f;
      for (int ii=0; ii<3; ii++)
        tmp += input[(j+j2)*(WIDTH+2) + i+ii] * weights[ii];
      tmp_buf[j2*WIDTH + i] = tmp;
    }

  // Step 2: 产出CHUNK_SIZE行输出
  for (int j2=0; j2<CHUNK_SIZE; j2++)
    for (int i=0; i<WIDTH; i++) {
      float tmp = 0.f;
      for (int jj=0; jj<3; jj++)
        tmp += tmp_buf[(j2+jj)*WIDTH + i] * weights[jj];
      output[(j+j2)*WIDTH + i] = tmp;
    }
}
```

**分析（以 CHUNK_SIZE=16 为例）**：
- Step 1 计算量：18 x 3 x WIDTH
- Step 2 计算量：16 x 3 x WIDTH
- 总的整图计算量：(34/16) x 3 x WIDTH x HEIGHT = **6.4 x WIDTH x HEIGHT**
- 当 CHUNK_SIZE 增大时，计算量趋近于理想值 6 x WIDTH x HEIGHT。
- 整个 tmp_buf 适合缓存，捕获所有生产者-消费者局部性。

---

### 5.6 极致优化的C++代码

最终手工优化版本包含了：
- **SIMD向量指令**（使用SSE intrinsics）
- **分块迭代顺序**（256x32的tile，最大化缓存命中率）
- **多核并行执行**（按垂直方向分区图像）
- **两次pass融合为一次**（tmp数据从缓存读取）

**结果**：比原始两遍代码快约10倍。

**代价**：
- 仅针对SSE（不兼容AVX2）
- 仅CPU代码
- **代码几乎无法阅读和理解**！

---

## 6. Halide 语言设计

### 6.1 基本概念

Halide是一种嵌入在C++中的领域特定语言，用于描述图像处理操作序列。

```cpp
Var x, y;
Func blurx, blury, bright, out;
Halide::Buffer<uint8_t> in = load_image("myimage.jpg");
Halide::Buffer<uint8_t> lookup = load_image("s_curve.jpg");  // 255-pixel 1D image

// perform 3x3 box blur in two-passes
blurx(x,y) = 1/3.f * (in(x-1,y)    + in(x,y)      + in(x+1,y));
blury(x,y) = 1/3.f * (blurx(x,y-1) + blurx(x,y) + blurx(x,y+1));

// brighten blurred result by 25%, then clamp
bright(x,y) = min(blury(x,y) * 1.25f, 255);

// access lookup table to contrast enhance
out(x,y) = lookup(bright(x,y));

// execute pipeline to materialize values of out in range (0:1024,0:1024)
Halide::Buffer<uint8_t> result = out.realize(1024, 1024);
```

### 6.2 Halide的核心概念

- **Halide函数（Func）**：在N维整数域上定义的无限（但离散）值集合。例如：每个像素位置的颜色值。
- **Halide表达式（Expr）**：无副作用的表达式，描述如何在某一点通过其他函数的值计算当前函数的值。
- **Var**：表示坐标轴的变量名（如 x, y）。

### 6.3 图像处理流水线建模为DAG

Halide程序被表示为**有向无环图（DAG）**：

```
  in (myimage.jpg)        lookup (s_curve.jpg)
       |                        |
     blurx                      |
       |                        |
     blury -------- bright -----+
       |
      out
```

### 6.4 表示的三个关键特性

1. **直观的表达**：
   - 采用局部"逐点（point-wise）"视角表达算法。
   - Halide是**声明式**语言——不定义迭代顺序、不定义存储方式。
   - 只定义计算所需的值，迭代是隐含的（无显式循环）。

2. **表示分离（Decoupling of Algorithm and Schedule）**：
   - **算法（Algorithm）**：描述"做什么"——要计算什么值。
   - **调度（Schedule）**：描述"怎么做"——如何高效地执行计算（循环结构、并行化、向量化等）。

3. **生产力与性能兼顾**：
   - 程序员用自然的"逐点"方式表达算法。
   - 编译器负责生成高效的并行+向量化实现。

### 6.5 简单序列化实现

```cpp
Func blurx, out;
Var  x, y, xi, yi;
Halide::Buffer<uint8_t> in = load_image("myimage.jpg");

// the "algorithm description" (declaration of what to do)
blurx(x,y) = (in(x-1, y) + in(x,y) + in(x+1,y)) / 3.0f;
out(x,y)   = (blurx(x,y-1) + blurx(x,y) + blurx(x,y+1)) / 3.0f;

// execute pipeline on domain of size 1024x1024
Halide::Buffer<uint8_t> result = out.realize(1024, 1024);
```

等效的C风格循环嵌套：

```c
allocate in(1024+2, 1024+2);   // (width, height)… initialize from image
allocate blurx(1024, 1024+2);   // (width, height)
allocate out(1024, 1024);       // (width, height)

for y=0 to 1024:
   for x=0 to 1024+2:
      blurx(x,y) = … compute from in

for y=0 to 1024:
   for x=0 to 1024:
      out(x,y) = … compute from blurx
```

这种简单实现存在的问题：
- 完整分配中间缓冲区 blurx。
- 没有利用生产者-消费者局部性（先完成所有blurx的计算，再开始计算out）。
- 没有并行化和向量化。

### 6.6 选择"正确"表示的重要性

好的表示应该：
- **对生产力有好处**：体现自然的思考方式。
- **使系统能提供有用的服务**：
  - 验证/提供保障（正确性、资源边界、类型检查等）。
  - 性能（并行化、向量化、专用硬件利用）。

在Halide的设计中，**关键洞见**是：表达了图像处理计算后，问题转化为"为特定Halide程序生成高效实现"。

---

## 7. Halide 调度（Schedule）

### 7.1 调度原语

Halide提供了第二套表示用于"调度"——程序员用高级原语描述如何将算法映射到并行机器上，具体细节由编译器生成。

**调度示例（同时使用调度原语）**：

```cpp
Func blurx, out;
Var  x, y, xi, yi;
Halide::Buffer<uint8_t> in = load_image("myimage.jpg");

// the "algorithm description" (declaration of what to do)
blurx(x,y) = (in(x-1, y) + in(x,y) + in(x+1,y)) / 3.0f;
out(x,y)   = (blurx(x,y-1) + blurx(x,y) + blurx(x,y+1)) / 3.0f;

// "the schedule" (how to do it)
out.tile(x, y, xi, yi, 256, 32).vectorize(xi, 8).parallel(y);
blurx.compute_at(out, x).vectorize(x, 8);

// execute pipeline on domain of size 1024x1024
Halide::Buffer<uint8_t> result = out.realize(1024, 1024);
```

**调度含义解释**：
- `out.tile(x, y, xi, yi, 256, 32)`：以256x32的tile大小对out进行2D分块遍历。
- `.vectorize(xi, 8)`：将最内层xi循环向量化（8-wide SIMD）。
- `.parallel(y)`：将y循环并行化（多线程）。
- `blurx.compute_at(out, x)`：按需计算blurx——在out的tile级别计算blurx的对应tile。
- `.vectorize(x, 8)`：向量化blurx的最内层循环。

### 7.2 迭代N维域的原语

- 指定迭代顺序
- 指定如何并行化（多线程、SIMD向量化）

典型模式——2D分块迭代顺序：外层循环遍历tile，内层循环遍历tile内的元素。

---

## 8. Halide循环嵌套的组织

### 8.1 默认顺序（compute_root）

```
Halide 调度:
blurx.compute_root();

循环嵌套图:          等效C代码:
<root>               allocate blurx(W, H+2)
  |                  for y:
blurx_y_loop           for x:
  |                      blurx(x,y) = ...
blurx_x_loop         for y:
  |                    for x:
out_y_loop               out(x,y) = ... from blurx
  |
out_x_loop
```

- blurx和out的循环是分离的，blurx全部计算完成后才开始out。
- 需要为blurx分配完整大小的缓冲区。

### 8.2 细粒度融合（compute_at out, xi）

```
Halide 调度:
out.tile(x, y, xi, yi, 256, 32);
blurx.compute_at(out, xi);

循环嵌套图:               等效C代码:
<root>                    allocate blurx(1, 3)   // 仅3个元素！
  |                       for tiles_y:
out_y_loop                  for tiles_x:
  |                           for yi:
out_x_loop                      for xi:
  |                                blurx(0,0) = ...
out_yi_loop                        blurx(0,1) = ...
  |                                blurx(0,2) = ...
out_xi_loop                        out(xi,yi) = ... from blurx
  |
blurx_y_loop
```

- 在计算out的每个像素时，按需计算3个blurx元素。
- 仅需为blurx分配1x3的缓冲区。
- 最大化生产者-消费者局部性。

### 8.3 中等粒度融合（compute_at out, x）

```
Halide 调度:
out.tile(x, y, xi, yi, 256, 32);
blurx.compute_at(out, x);

循环嵌套图:               等效C代码:
<root>                    allocate blurx(256, 34)    // tile大小
  |                       for tiles_y:
out_y_loop                  for tiles_x:
  |                           for yi (0..32+2):
out_x_loop                      for xi (0..256):
  |                                blurx(xi,yi) = ...
blurx_yi_loop                  for yi (0..32):
  |                               for xi (0..256):
blurx_xi_loop                       out(xi,yi) = ...
  |
out_yi_loop
  |
out_xi_loop
```

- 在tile级别计算blurx的一个tile（256x34）。
- 平衡了中间缓冲区大小和计算局部性。
- 注意：blurx需要额外2行（34=32+2），用来支持垂直方向的边界。

### 8.4 完整优化调度的总结

```cpp
// 算法描述
blurx(x,y) = (in(x-1, y) + in(x,y) + in(x+1,y)) / 3.0f;
out(x,y)   = (blurx(x,y-1) + blurx(x,y) + blurx(x,y+1)) / 3.0f;

// 调度
out.tile(x, y, xi, yi, 256, 32).vectorize(xi, 8).parallel(y);
blurx.compute_at(out, x).vectorize(x, 8);
```

等效的并行循环嵌套：

```c
allocate in(1024+2, 1024+2);
allocate out(1024, 1024);

for y=0 to num_tiles_y:   // 此循环的迭代被并行化为多线程
   for x=0 to num_tiles_x:
      allocate blur_x(258, 34);  // tile的blurx缓冲区（含边界）
      for yi=0 to 32+2:
         for xi=0 to 256+2 BY 8:
            blurx(xi, yi) = ... // 使用8-wide SIMD指令
                                 // 编译器自动处理256+2不被8整除的边界条件
      for yi=0 to 32:
         for xi=0 to 256 BY 8:
            idx_x = x*256 + xi;
            idx_y = y*32 + yi;
            out(idx_x, idx_y) = ... // 使用8-wide SIMD指令
```

---

## 9. Halide 的设计哲学

### 9.1 分工明确

- **程序员的职责**：描述图像处理算法 + 提供高级调度决策（循环结构、展开、向量化、多核并行化）。
- **系统（Halide编译器）的职责**：机械地执行调度的细节（pthreads、AVX intrinsics等）。编译器本身并不是"聪明的"。

### 9.2 语言约束

为使编译器能提供所需服务，Halide对语言施加了约束：

- **应用领域范围**：在规则的N维域上的计算。
- **仅支持前馈流水线**（附带对归约和固定深度递归的特殊支持）。
- **所有依赖关系均可由编译器推断**。

---

## 10. Halide 的实际效果

### 10.1 学术实验结果

**应用一：相机RAW处理流水线**（将RAW传感器数据转换为RGB图像）
- 原始代码：463行手工调优的ARM NEON汇编
- Halide代码：代码量减少 2.75倍，性能提升 5%。

**应用二：双边滤波（Bilateral Filter）**
- 原始代码：122行C++
- Halide算法：34行 + 调度：6行 = 总计40行
- CPU实现：快 5.9倍
- GPU实现：比手写CUDA快 2倍

### 10.2 实际生产中的复杂度

实际生产级图像处理管线可能包含数百到数千个Halide函数：

| 应用 | Halide函数数量 |
|------|----------------|
| Two-pass blur | 2 |
| Unsharp mask | 9 |
| Harris Corner detection | 13 |
| Camera RAW processing | 30 |
| Non-local means denoising | 13 |
| Max-brightness filter | 9 |
| Multi-scale interpolation | 52 |
| Local-laplacian filter | 103 |
| Synthetic depth-of-field | 74 |
| Bilateral filter | 8 |
| Histogram equalization | 7 |
| VGG-16 deep network eval | 64 |
| **Google HDR+** | **2000+** |

---

## 11. 自动生成 Halide 调度

### 11.1 问题

- 尽管调度原语大大提升了生产力，但**极少有程序员具备编写好的Halide调度的能力**。
- Google有80+程序员编写Halide算法，但只有极少数被信任编写调度。
- 调度仍然需要对硬件和优化技术的深刻理解。

### 11.2 解决方案

扩展Halide编译器，使其能**自动分析Halide程序并生成高效调度**（Adams et al., SIGGRAPH 2019）。

### 11.3 将调度建模为选择序列

对于程序DAG中的每个节点N（从DAG末尾开始）：

1. 选择当前节点N在现有循环嵌套中的放置位置（决定 `N.compute_at()`）。
2. 为N选择tile大小（假设外层维被线程并行化，内层维被向量化）。

重复直到整个DAG被调度完毕。

### 11.4 使用搜索找到最佳调度

- 在巨大的调度空间中进行搜索（例如贪心搜索、束搜索（Beam Search））。
- 每个部分调度有一个估计的执行成本。
- 挑战：可能需要搜索成千上万种可能的调度。

### 11.5 AI辅助成本估计

- 给定程序 + 调度，使用机器学习估计执行成本。
- 使用简单的**MLP（多层感知器）**，每次预测仅需数十微秒（例如在166秒内测试了140万种调度）。
- 训练数据：大规模随机生成的Halide程序，编译并运行以获得真实执行成本。
- 实际上，MLP不直接预测成本，而是输出27个系数，这些系数被插入一个手工设计的成本模型中。

### 11.6 自动调度器效果

**自动调度器**针对CPU上图像处理应用的吞吐量，可与最佳人类手写调度相媲美。

- TL;DR：对于CPU上的图像处理应用，很难手工写出比Halide自动调度器更好的调度。

### 11.7 自动调度器节省专家时间

实验中比较了自动调度器 vs 两位Halide专家（Dillon, Andrew）在三个应用上的表现：

- **Max filter**：自动调度器在极短时间（约5分钟）内达到与专家30分钟相当的吞吐量。
- **Non-local means denoising**：自动调度器在很短时间（约5分钟）内接近最佳效果，而专家通常需要50-90分钟。
- **Lens blur**：自动调度器在1分钟内达到较好水平。

### 11.8 核心启示

- Halide的调度原语最初设计是为了**提升人类专家程序员的生产力**。
- 但调度的高层抽象也**清晰地枚举了所有可能调度的空间**，使自动搜索成为可能。
- 对比：考虑搜索一个C++程序的所有可能排列——这不切实际。

---

## 12. LLM 代码生成与自动优化

### 12.1 通过反思进行试错优化

一种新兴的优化方法是利用大语言模型（LLM）作为优化引擎：

```
Starting code (e.g., PyTorch) + LLM prompt
    ↓
LLM → CUDA Code
    ↓
Execute / Profile
    ↓
Correct: Y/N → Timing: 32 ms → SM util: 42% → DRAM util: 89% → L2 cache hit rate: 68%
    ↓
LLM (reflection + edit based on profiling stats)
    ↓
Improved CUDA Code
```

- **第一步**：用提示词引导LLM将PyTorch代码重写为高性能CUDA代码。
- **第二步**：编译运行并获取性能分析数据。
- **第三步**：将profile数据反馈给LLM，要求其反思瓶颈并修改代码。
- **迭代**：重复直到达到满意的性能。

### 12.2 KernelBench

一个包含数百个PyTorch kernel的基准测试集，LLM agent的目标是自动生成正确且高性能的CUDA kernel。

---

## 13. 为LLM优化设计的DNN DSL

### 13.1 DSL降低LLM生成难度

使用面向DNN的领域特定语言有助于自动化：

**优势**：
- LLM组装的是高层性能原语，而非编写低级CUDA代码。
- 降低正确性错误/幻觉的概率。

**挑战**：
- LLM在不那么流行的语言上编写代码更困难（训练数据较少，但随着时间推移会解决）。

相关的DNN DSL包括：**Triton**、**CUTLASS/CuTe**、**TileLang**。

---

## 14. LLM Agent 的自我提升策略

### 14.1 思路一：基于经验微调LLM

- 使用优化经验微调一个专用的LLM，专门用于特定类型的编程任务。
- 需要大量的任务数据和微调更大LLM的能力。

### 14.2 思路二：构建"示例解决方案"数据库

LLM Agent通过积累"练习问题"的解答来自我提升：

```
问题/起点代码 → 数据库检索相关示例 → LLM → 输出代码 → 执行/Profile → 结果
```

- 建立一个高质量内核编程解决方案的数据库（例如在Thunderkittens或Cute中）。
- 不仅存储解决方案，还存储**优化决策序列**。
- 遇到新问题时，检索最相关的实践问题及其解答作为参考。
- 实验表明，带数据库的agent相比无数据库的LLM起点**显著提高了问题解决成功率**。

### 14.3 思路三：通过优化Prompt来自我提升

与思路二相同，但根据经验更新给LLM的prompt（而非仅提供相关示例）：

```
问题/起点代码 → LLM → 执行/Profile → 输出代码 → 结果
                    ↑
            Prompt Optimizer
        (检查优化循环的轨迹，
         总结为重要事实和原则)
```

- 检查优化循环的轨迹。
- 将经验总结为重要的事实和原则。
- 更新提示词以指导未来的优化。

### 14.4 思路四：搜索 + LLM 混合方法

- 将穷举搜索技术（如Halide自动调优）与LLM agentic思想结合。
- 优化成本极高，但可能获得最佳结果。

---

## 15. 课程总结

### 15.1 核心观点

- 性能优化需要高度专业知识。
- 即使对专家来说也是繁琐和困难的。
- 而且需要为新机器、稍有不同的任务重复做类似工作。
- 公司在AI计算成本上每年花费数千万到数亿美元。
- **这显然是一个非常适合自动化的领域**。

### 15.2 未来展望

- 未来最优秀的CS149学生将能与**自动化的Agent并行工作**，加速他们的思考和工作效率。
- 一个有趣的争论点是：通往成功的真正价值在于**DSL的设计**还是**LLM Agent**？

### 15.3 三条技术路线

| 路线 | 核心理念 | 代表性工作 |
|------|---------|-----------|
| 提升抽象层次 | 分离算法描述与执行策略 | Halide（算法-调度分离） |
| 智能搜索 | 在优化方案空间中自动搜索 | Halide自动调度器（ML成本模型 + 束搜索） |
| LLM代码生成 | 利用LLM的推理能力进行迭代优化 | KernelBench + 反思式优化循环 |

### 15.4 关键设计思想

1. **表示分离（Separation of Concerns）**：将"做什么"（算法）和"怎么做"（调度）分离，使编译器能够自动探索实现空间。
2. **领域知识编码**：将领域特定的优化模式和方法编码到系统中。
3. **从经验中学习**：无论是机器学习模型从训练数据学习，还是LLM agent从每次优化尝试中获得反思。
4. **组合式方法**：将搜索、学习和代码生成结合起来，可能获得最佳结果。
