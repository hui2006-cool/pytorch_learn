# learn_dataloder.py 代码总结

## 1. 这个脚本的作用
这个脚本用于演示 PyTorch 中如何：
- 加载 CIFAR10 数据集
- 使用 DataLoader 批量读取图片
- 查看图片张量的形状
- 将图片写入 TensorBoard 进行可视化

## 2. 主要步骤说明

### 2.1 导入依赖
脚本先导入了以下模块：
- `torchvision`：用于加载 CIFAR10 数据集
- `torch.utils.data.DataLoader`：用于批量读取数据
- `torchvision.transforms`：用于对图像做变换
- `torch`：PyTorch 基础库
- `torch.utils.tensorboard.SummaryWriter`：用于向 TensorBoard 写入日志

### 2.2 加载 CIFAR10 数据集
代码使用了 CIFAR10 数据集，并设置了：
- `root='learn_pytorch/cifar-10-python'`：数据集存放路径
- `train=False`：加载测试集
- `download=True`：如果本地没有数据，则下载
- `transform=transforms.ToTensor()`：将图片转换为 PyTorch 张量

### 2.3 创建 DataLoader
通过 `DataLoader` 将数据集包装为可批量读取的对象：
- `batch_size=64`：每次读取 64 张图片
- `shuffle=True`：打乱顺序

这一步的作用是让训练或测试时更方便地按批次处理数据。

### 2.4 查看第一个样本
代码取出数据集中的第一个样本，并打印：
- 图片张量的形状
- 对应标签

这有助于理解数据在 PyTorch 中的形式。

### 2.5 写入 TensorBoard
脚本创建了一个 `SummaryWriter`，日志目录名为 `logs2`。
然后遍历 DataLoader 中的每个 batch：
- 取出图片和标签
- 将图片写入 TensorBoard

## 3. 关键概念

### 3.1 Dataset
`test_data` 是数据集对象，表示数据集本身。

### 3.2 DataLoader
`test_loader` 是数据加载器，用来按批次从数据集中取样本。

### 3.3 TensorBoard
`SummaryWriter` 用于把图片、标量等信息写入 TensorBoard，便于可视化。

## 4. 代码的核心思路
这个脚本的整体思路可以概括为：
1. 从 CIFAR10 数据集中读取图片
2. 用 DataLoader 以 batch 的形式取出数据
3. 打印图片和标签的信息
4. 把图片写入 TensorBoard




