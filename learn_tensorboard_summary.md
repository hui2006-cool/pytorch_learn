# TensorBoard 可视化代码总结

## 一、代码概述

本代码演示了如何使用 PyTorch 的 TensorBoard 可视化功能，将一张图片和一些标量曲线记录到日志文件中，方便在 TensorBoard 页面中查看。

---

## 二、使用的库

| 库 | 导入方式 | 作用 |
|----|----------|------|
| **numpy** | `import numpy as np` | 将图像转换为 NumPy 数组，便于 TensorBoard 处理 |
| **torch.utils.tensorboard** | `from torch.utils.tensorboard import SummaryWriter` | 用于创建 TensorBoard 日志写入器 |
| **PIL (Pillow)** | `from PIL import Image` | 用于打开和读取图片文件 |

---

## 三、核心语法

### 3.1 创建日志写入器

```python
writer = SummaryWriter("logs")
```

- `SummaryWriter` 用来把数据写入 TensorBoard 日志目录。
- 这里的 `logs` 表示日志文件会保存到当前目录下的 `logs` 文件夹中。

### 3.2 读取图片

```python
img_path = r"C:\Users\48596\Desktop\寒假学习\deeplearn\ants&bees_data\val\ants\35558229_1fa4608a7a.jpg"
img_PIL = Image.open(img_path)
img_np = np.array(img_PIL)
```

- `Image.open()` 用于打开图片文件。
- `np.array()` 将图片转换为 NumPy 数组，TensorBoard 可以直接处理这种格式。

### 3.3 写入图片

```python
writer.add_image("test", img_np, 1, dataformats="HWC")
```

- `"test"`：图片标签名，用于在 TensorBoard 中显示。
- `img_np`：要写入的图片数据。
- `1`：表示第 1 个 step。
- `dataformats="HWC"`：表示图片数据格式为高、宽、通道（Height-Width-Channel）。

### 3.4 写入标量

```python
for i in range(100):
    writer.add_scalar("y=x", i, i)
    writer.add_scalar("y=2x", 2*i, i)
```

- `add_scalar()` 用来记录数值曲线。
- 这里分别记录了两条曲线：
  - `y=x`
  - `y=2x`
- 每次循环中，`i` 会作为 x 轴的值（step）和 y 轴的值。

### 3.5 关闭写入器

```python
writer.close()
```

- 关闭日志写入器，确保数据被正确保存。

---

## 四、变量说明

| 变量 | 类型 | 说明 |
|------|------|------|
| `writer` | `SummaryWriter` | TensorBoard 日志写入器 |
| `img_path` | `str` | 图片文件路径 |
| `img_PIL` | `PIL.Image` | 打开的图片对象 |
| `img_np` | `numpy.ndarray` | 转换后的图片数组 |
| `i` | `int` | 循环变量，用于生成标量数据 |

---

## 五、逻辑流程

```text
创建 SummaryWriter
    ↓
读取图片文件
    ↓
把图片转换为 NumPy 数组
    ↓
把图片写入 TensorBoard
    ↓
循环生成两条标量曲线
    ↓
关闭 writer
```

---

## 六、这个代码的作用

这个脚本主要用于学习 TensorBoard 的基本用法，展示了两类常见可视化方式：

1. 图像可视化：`add_image()`
2. 曲线可视化：`add_scalar()`

它适合初学者用来理解 TensorBoard 如何记录训练过程中的图像和数值变化。

---

## 七、注意事项

### 7.1 图片路径要正确

图片路径必须准确，否则会报错。

### 7.2 图片格式要符合要求

这里使用了 `dataformats="HWC"`，这表示图片是按“高-宽-通道”的方式组织的。

### 7.3 需要启动 TensorBoard 查看结果

运行脚本后，需要在终端中执行：

```bash
tensorboard --logdir=logs
```

然后在浏览器中打开给出的地址，即可查看可视化结果。

---

## 八、总结

这个程序是一个非常基础的 TensorBoard 入门示例，核心思想是：

- 用 `SummaryWriter` 记录日志
- 用 `add_image()` 记录图片
- 用 `add_scalar()` 记录数值曲线

通过这个例子，可以初步理解 TensorBoard 在深度学习训练可视化中的作用。
