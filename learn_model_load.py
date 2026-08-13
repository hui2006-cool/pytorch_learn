import torch
import torchvision
import torchvision.models.vgg as vgg
torch.serialization.add_safe_globals([vgg.VGG])
model = torch.load("vgg16_method1.path",weights_only=True)
print(model)
#(推荐第二种)
vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)