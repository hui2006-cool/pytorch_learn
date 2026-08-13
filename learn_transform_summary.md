# 图像变换与 TensorBoard 可视化总结

## 一、代码概述

本代码演示了如何使用 torchvision 中的图像变换函数，对图片进行多种预处理操作，并将处理后的结果通过 TensorBoard 可视化展示。

---

## 二、使用的库

| 库 | 导入方式 | 作用 |
|----|----------|------|
| `torchvision.transforms` | `from torchvision import transforms` | 提供图像预处理操作，如 Resize、RandomCrop、ToTensor、Normalize |
| `PIL.Image` | `from PIL import Image` | 用于打开和读取图片 |
| `torch.utils.tensorboard` | `from torch.utils.tensorboard import SummaryWriter` | 用于把图片写入 TensorBoard 日志 |

---

## 三、核心知识点

### 3.1 Compose

```python
base_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

`Compose` 的作用是把多个变换组合起来，按顺序执行。

- 先把图片转成 Tensor
- 再做归一化

它的优点是代码更整洁，也更容易管理一整套预处理流程。

### 3.2 Resize

```python
transforms.Resize((256, 256))
```

`Resize` 用来把图片缩放到指定大小。

- 适合统一输入尺寸
- 让不同大小的图片可以进入神经网络

### 3.3 RandomCrop

```python
transforms.RandomCrop(224)
```

`RandomCrop` 会从原图中随机裁剪出一块指定大小的区域。

- 可以增加数据多样性
- 常用于数据增强
- 让模型更不容易“记住”固定位置的特征

### 3.4 ToTensor

```python
transforms.ToTensor()
```

`ToTensor` 会把 PIL 图像转换成 PyTorch 张量。

转换后，像素值范围通常会变成 $[0,1]$，适合神经网络输入。

### 3.5 Normalize

```python
transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
```

`Normalize` 用来对像素值做标准化处理。

公式为：

$$x' = \frac{x - mean}{std}$$

这里把每个通道都按均值 0.5、标准差 0.5 来处理，目的是让数据更接近均值为 0、方差为 1 的分布，训练时更稳定。

---

## 四、代码流程

```text
读取图片
    ↓
创建基础变换：ToTensor + Normalize
    ↓
创建组合变换：Resize + RandomCrop + ToTensor + Normalize
    ↓
生成不同版本图片
    ↓
写入 TensorBoard
```

---

## 五、代码中的几个关键变量

| 变量 | 作用 |
|------|------|
| `img_PIL` | PIL 图像对象 |
| `base_transforms` | 基础预处理流程 |
| `train_transforms` | 包含 Resize、RandomCrop、ToTensor、Normalize 的组合流程 |
| `img_tensor` | 原图转成张量后的结果 |
| `img_resized` | Resize 后的图片 |
| `img_cropped` | RandomCrop 后的图片 |
| `img_norm_crop` | 裁剪后再归一化的结果 |
| `writer` | TensorBoard 写入器 |

---

## 六、这段代码的作用

这段代码主要展示了三件事：

1. 图像预处理流程如何写
2. `Compose` 如何把多个变换串联起来
3. `RandomCrop` 如何做数据增强

它是深度学习中非常常见的一部分，尤其在训练图像分类模型时经常使用。

---

## 七、注意事项

### 7.1 `Resize` 和 `RandomCrop` 的顺序

在这里，先 `Resize` 再 `RandomCrop`，这是一种常见的组合方式。

- 先缩放，保证图片大小统一
- 再裁剪，获得更丰富的训练样本

### 7.2 `RandomCrop` 结果会变化

因为是随机裁剪，所以每次运行可能得到不同的裁剪区域。

### 7.3 TensorBoard 中要看对应标签

代码中写入了几个标签：

- `origin`
- `resize`
- `random_crop`

所以在 TensorBoard 的 Images 页面里要对应查看这些标签。

---

## 八、总结

这次学习的重点是：

- `Compose` 用来组合多个图像变换
- `Resize` 用来统一图片尺寸
- `RandomCrop` 用来做随机裁剪，增强数据
- `Normalize` 用来归一化像素值
- 这些步骤都是深度学习图像预处理的基础

掌握这些变换，后面再训练 CNN、ResNet 等模型时会非常有用。
