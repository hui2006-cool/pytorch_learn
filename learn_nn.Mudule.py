import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='learn_pytorch/cifar-10-python', train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

class hui(nn.Module):
    def __init__(self):
        super(hui, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
  

    def forward(self, x):
        x= self.conv1(x)
        return x
    
hui = hui()
print(hui)

writer = SummaryWriter("logs3")
step = 0
for data in dataloader:
    imgs, targets = data
    output = hui(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input", imgs, step)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    
    step = step + 1
writer.close()
