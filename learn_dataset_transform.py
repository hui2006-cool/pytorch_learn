import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
train_set = torchvision.datasets.CIFAR10(root='learn_pytorch', train=True, download=True,transform=transforms.ToTensor()) 
test_set = torchvision.datasets.CIFAR10(root='learn_pytorch', train=False, download=True,transform=transforms.ToTensor())
# img,target = train_set[0]
# print(target)
# print(train_set.classes[target])
# img.show()
# print(train_set[0][0].shape)
writer = SummaryWriter('log1')
for i in range(10):
    img,target = train_set[i]
    writer.add_image('train_set',img,i)
    print(target)
    print(train_set.classes[target])
writer.close()
