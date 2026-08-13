import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
writer = SummaryWriter("logs")
img_path = r"C:\Users\48596\Desktop\寒假学习\deeplearn\ants&bees_data\val\ants\35558229_1fa4608a7a.jpg"
img_PIL = Image.open(img_path)
img_np = np.array(img_PIL)

writer.add_image("test", img_np, 1, dataformats="HWC")
for i in range(100):
    writer.add_scalar("y=x", i, i)
    writer.add_scalar("y=2x", 2*i, i)

writer.close()

