from torch.utils.data import Dataset
from PIL import Image
import os  
class myData(Dataset):
    def __init__(self, root_dir, label_dir):
        # Initialize your dataset, e.g., load data from files
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.image_path = [f for f in os.listdir(self.path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    def __getitem__(self,index):
        # Load and return a single data point (image and label)
        image_name = self.image_path[index]
        image_path = os.path.join(self.path, image_name)
       
        # Load the image
        image = Image.open(image_path)
        
        # Load the label
        label = self.label_dir       
        return image, label
    def __len__(self):
        # Return the total number of data points
        return len(self.image_path)

root_dir = 'C:\\Users\\48596\\Desktop\\寒假学习\\deeplearn\\ants&bees_data\\train'
ants_label_dir = "ants"
ants_dataset = myData(root_dir, ants_label_dir)
root_dir = 'C:\\Users\\48596\\Desktop\\寒假学习\\deeplearn\\ants&bees_data\\train'
bees_label_dir = "bees"
bees_dataset = myData(root_dir, bees_label_dir)
train_dataset = ants_dataset + bees_dataset
root,label_dir = train_dataset[0]
print(root,label_dir)

root.show()
