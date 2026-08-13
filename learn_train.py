import torch 
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from train_model import *
from torch.utils.tensorboard import SummaryWriter
#准备数据集
train_data = torchvision.datasets.CIFAR10(root='learn_pytorch\cifar-10-python',
                                           train=True, 
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
test_data = torchvision.datasets.CIFAR10(root='learn_pytorch\cifar-10-python',
                                           train=False, 
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)

#length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集长度：{}".format(train_data_size))
print("测试数据集长度：{}".format(test_data_size))


#运用datalodar加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

#创建网络模型
Model = model()

#损失函数

loss_fn = nn.CrossEntropyLoss()

#优化器
 
learning_rate = 1e-2
optimizer = torch.optim.SGD(Model.parameters(), lr=learning_rate)

#训练网络的一些参数

total_train_step = 0

total_test_step = 0

epoch = 10

#tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("-----第{}轮训练-----".format(i+1))
    Model.train()
    for data in train_dataloader:
        imgs,targets = data
        outputs = Model(imgs)
        loss = loss_fn(outputs,targets)

        #优化器
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step+1
        if total_train_step %100 == 0 :
            print("训练次数：{}, loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    #测试
    Model.eval()
    total_test_loss = 0
    total_accuracy_size = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = Model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss+loss
            accuracy_size = (outputs.argmax(1) ==targets).sum()
            total_accuracy_size += accuracy_size
    print("整体测试集的loss:{}".format(total_test_loss))
    print("整体测试集的正确率：{}".format(total_accuracy_size/test_data_size))
    writer.add_scalar("test_loss",total_test_loss.item(),total_test_step)
    writer.add_scalar("test_acc",total_accuracy_size/test_data_size,total_test_step)
    total_test_step += 1


    #torch.save(Model,"Model{}.pth".format(i))
    torch.save(Model.state_dict(),"Model{}.pth".format(i))
    print("模型已保存")
writer.close()