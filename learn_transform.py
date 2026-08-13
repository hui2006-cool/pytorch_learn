from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_path = r"C:\Users\48596\Desktop\寒假学习\deeplearn\ants&bees_data\val\ants\35558229_1fa4608a7a.jpg"
img_PIL = Image.open(img_path)

# 1. 基础变换：转 Tensor + 归一化
base_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 2. resize + crop 的组合变换
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 3. 生成不同版本图片
img_tensor = base_transforms(img_PIL)
img_resized = transforms.Resize((256, 256))(img_PIL)
img_cropped = transforms.RandomCrop(224)(img_resized)
img_crop_tensor = transforms.ToTensor()(img_cropped)
img_norm_crop = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img_crop_tensor)

writer = SummaryWriter("logs")
writer.add_image("origin", img_tensor, 0, dataformats="CHW")
writer.add_image("resize", transforms.ToTensor()(transforms.Resize((256, 256))(img_PIL)), 1, dataformats="CHW")
writer.add_image("random_crop", img_norm_crop, 2, dataformats="CHW")
writer.close()
