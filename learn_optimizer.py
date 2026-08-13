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
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = model(imgs)
    
        loss_value = loss(outputs, targets)
        optimizer.zero_grad()  # 清空梯度
        loss_value.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数
        running_loss += loss_value
    print(f"Epoch {epoch + 1}, Loss: {running_loss.item()}")