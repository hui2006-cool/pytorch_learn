import torchvision
import torch.nn
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print("ok")
print(vgg16_true)


train_data = torchvision.datasets.CIFAR10(root='learn_pytorch/cifar-10-python', train=True, download=True, transform=torchvision.transforms.ToTensor())

vgg16_true.add_module('add_linear', torch.nn.Linear(1000, 10))

#vgg16_true.classifier.add_module('add_linear', torch.nn.Linear(1000, 10))
print(vgg16_true)

vgg16_false.classifier[6] =torch.nn.Linear(4096,10)
print (vgg16_false)