import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import MaxPool2d
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='learn_pytorch/cifar-10-python', train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

class MaxPool(nn.Module):
    def __init__(self, kernel_size=3):
        super(MaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.maxpool = MaxPool2d(kernel_size=self.kernel_size, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool(input)
        return output
writer = SummaryWriter("logs_maxpool")
step = 0
hui = MaxPool(kernel_size=3)
for data in dataloader:
    imgs, targets = data
    output = hui(imgs)
    print(output.shape)
    print(output)
    writer.add_images("input", imgs, step)
    writer.add_images("maxpool_output", output, step)
    step = step + 1
writer.close()
