import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

# 1. 加载 CIFAR10 数据集
# 图片大小是 32x32，3 个通道，所以每张图片有 32*32*3 = 196608 个像素
Dataset = torchvision.datasets.CIFAR10(
    root='learn_pytorch/cifar-10-python',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

# 2. 每次取 4 张图片，方便观察
loader = DataLoader(Dataset, batch_size=64, shuffle=True)


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=3 * 32 * 32, out_features=10)

    def forward(self, x):
        # 把每张图片展开成一维向量
        # 输入 x 的形状是 [batch_size, 3, 32, 32]
        # 展开后变成 [batch_size, 196608]
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


model = LinearModel()

for imgs, targets in loader:
    print('输入图片形状：', imgs.shape)
    output = model(imgs)
    print('输出形状：', output.shape)
    print('标签：', targets)
    break
