import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
input = torch.tensor([[1.0, -0.5], [-1.0, 3.0]])

input = torch.reshape(input, (-1, 1, 2, 2))
print (input.shape)
dataset = torchvision.datasets.CIFAR10(
    root='learn_pytorch/cifar-10-python',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
class NonLinearActivation(nn.Module):
    def __init__(self):
        super(NonLinearActivation, self).__init__()
        self.relu = nn.ReLU()
    def forward(self, input):
        output = self.relu(input)
        return output
nonlinear_activation = NonLinearActivation()
output = nonlinear_activation(input)
print(output)
class NonLinearActivation2(nn.Module):
    def __init__(self):
        super(NonLinearActivation2, self).__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        output = self.sigmoid(input)
        return output
    
nonlinear_activation2 = NonLinearActivation2()
output2 = nonlinear_activation2(input)
print(output2)
writer = SummaryWriter("logs_nonlinear_activation")
step = 0
for data in dataloader:
    imgs,targets = data
    output = nonlinear_activation(imgs)
    output2 = nonlinear_activation2(imgs)
    writer.add_images("input", imgs, step)
    writer.add_images("relu_output", output, step)
    writer.add_images("sigmoid_output", output2, step)
    step += 1
writer.close()