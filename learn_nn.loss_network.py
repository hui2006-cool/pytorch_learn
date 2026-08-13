import torch
import torch.nn as nn
import torchvision

dataset = torchvision.datasets.CIFAR10(root='learn_pytorch/cifar-10-python', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
        
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 4 * 4, out_features=64),
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        return self.network(x)
    
loss = nn.CrossEntropyLoss()
model = SequentialModel()
for data in dataloader:
    imgs, targets = data
    outputs = model(imgs)
    print('输入形状：', imgs.shape)
    print('输出形状：', outputs.shape)
    print('标签：', targets)
    loss_value = loss(outputs, targets)
    print('损失值：', loss_value)
    loss_value.backward()
    print('梯度：', model.network[0].weight.grad)
