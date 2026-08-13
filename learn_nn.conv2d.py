import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 1. 加载 CIFAR10 数据集
# 这里使用测试集，并把图片转成 PyTorch 张量
Dataset = torchvision.datasets.CIFAR10(
    root='learn_pytorch/cifar-10-python',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

# 2. 使用 DataLoader 按批次读取数据
# batch_size=4，意思是每次取 4 张图片
loader = DataLoader(Dataset, batch_size=4, shuffle=True)


class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        # 输入通道 3 表示彩色图像，输出通道 6 表示提取 6 种特征
        # kernel_size=3 表示 3x3 卷积核，stride=1，padding=1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x


model = SimpleConv()
print(model)

writer = SummaryWriter('logs_conv')
step = 0

for data in loader:
    imgs, targets = data
    outputs = model(imgs)

    print('输入形状：', imgs.shape)
    print('输出形状：', outputs.shape)
    print('标签：', targets)

    # 写入 TensorBoard，可视化输入和卷积输出
    writer.add_images('input', imgs, step)
    writer.add_images('conv_output', outputs[:, :3], step)

    step += 1
    if step >= 3:
        break

writer.close()
