import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained = False)
#1
torch.save(vgg16, "vgg16_method1.path")

#2(推荐)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")