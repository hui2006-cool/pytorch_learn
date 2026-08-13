# PyTorch 自定义数据集代码总结

## 一、代码概述

本代码实现了一个基于 PyTorch 的自定义图像数据集类，用于加载和处理蚂蚁/蜜蜂二分类数据集。

---

## 二、使用的库

| 库 | 导入方式 | 作用 |
|----|----------|------|
| **torch.utils.data** | `from torch.utils.data import Dataset` | 提供数据集基类，用于构建可迭代的数据集对象 |
| **PIL (Pillow)** | `from PIL import Image` | 图像处理库，用于打开、读取和显示图像 |
| **os** | `import os` | 操作系统接口，用于路径拼接和文件列表获取 |

---

## 三、核心语法

### 3.1 类定义与继承

```python
class myData(Dataset):
```

- 使用 `class` 关键字定义类
- 继承自 `torch.utils.data.Dataset`，必须实现三个核心方法：
  - `__init__`: 初始化数据集
  - `__getitem__`: 获取单个数据样本
  - `__len__`: 返回数据集大小

### 3.2 魔法方法（Dunder Methods）

| 方法 | 作用 | 调用方式 |
|------|------|----------|
| `__init__(self, ...)` | 构造函数，初始化对象 | `myData(root_dir, label_dir)` |
| `__getitem__(self, index)` | 返回指定索引的数据 | `dataset[0]` |
| `__len__(self)` | 返回数据集长度 | `len(dataset)` |

### 3.3 列表推导式

```python
self.image_path = [f for f in os.listdir(self.path) if f.endswith(('.png', '.jpg', '.jpeg'))]
```

- 遍历目录下所有文件，筛选出图像文件
- 条件判断：文件名以 `.png`、`.jpg` 或 `.jpeg` 结尾

### 3.4 路径拼接

```python
self.path = os.path.join(root_dir, label_dir)
```

- `os.path.join()` 自动处理跨平台路径分隔符（Windows: `\`，Linux/Mac: `/`）

---

## 四、变量说明

### 4.1 实例变量（类内部）

| 变量 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `self.root_dir` | `str` | 数据集根目录路径 | `'deeplearn\\ants&bees_data\\train'` |
| `self.label_dir` | `str` | 类别名称（子目录名） | `"ants"` |
| `self.path` | `str` | 拼接后的完整路径 | `'deeplearn\\ants&bees_data\\train\\ants'` |
| `self.image_path` | `list[str]` | 图像文件名列表 | `['0013035.jpg', '0013036.jpg', ...]` |

### 4.2 局部变量（方法内部）

| 变量 | 类型 | 说明 |
|------|------|------|
| `image_name` | `str` | 当前图像文件名 |
| `image_path` | `str` | 当前图像完整路径 |
| `image` | `PIL.Image` | 加载的图像对象 |
| `label` | `str` | 图像标签（当前为字符串，需改进为数值） |

### 4.3 全局变量

| 变量 | 类型 | 说明 |
|------|------|------|
| `root_dir` | `str` | 训练集根目录 |
| `ants_label_dir` | `str` | 蚂蚁类别目录名 |
| `ants_dataset` | `myData` | 蚂蚁数据集实例 |
| `bees_dataset` | `myData` | 蜜蜂数据集实例 |
| `train_dataset` | `ConcatDataset` | 合并后的训练数据集 |

---

## 五、逻辑流程

### 5.1 初始化流程（`__init__`）

```
输入参数: root_dir, label_dir
        ↓
拼接完整路径: self.path = root_dir + label_dir
        ↓
扫描目录: os.listdir(self.path)
        ↓
筛选图像文件: 过滤出 .png/.jpg/.jpeg 文件
        ↓
保存到 self.image_path
```

### 5.2 数据获取流程（`__getitem__`）

```
输入: index（数据索引）
        ↓
获取文件名: self.image_path[index]
        ↓
拼接完整路径: os.path.join(self.path, image_name)
        ↓
打开图像: Image.open(image_path)
        ↓
获取标签: label = self.label_dir
        ↓
返回: (image, label)
```

### 5.3 数据集使用流程

```
创建数据集实例 → 合并数据集 → 按索引访问 → 使用数据
    ↓               ↓              ↓              ↓
myData()      ConcatDataset()  dataset[i]    训练/测试
```

---

## 六、注意事项

### 6.1 路径问题 ⚠️

**错误示例：**
```python
image_path = os.path.join(self.root_dir, image_name)  # 缺少子目录
# 生成: 'train\\0013035.jpg' ❌
```

**正确示例：**
```python
image_path = os.path.join(self.path, image_name)  # 使用完整路径
# 生成: 'train\\ants\\0013035.jpg' ✅
```

### 6.2 标签问题 ⚠️

**当前问题：** 返回字符串标签 `"ants"` 或 `"bees"`，深度学习模型需要整数标签。

**改进方案：**
```python
label_map = {"ants": 0, "bees": 1}
label = label_map[self.label_dir]  # 返回 0 或 1
```

### 6.3 数据集合并 ⚠️

**错误示例：**
```python
train_dataset = ants_dataset + bees_dataset  # Dataset 不支持 + 运算符 ❌
```

**正确示例：**
```python
from torch.utils.data import ConcatDataset
train_dataset = ConcatDataset([ants_dataset, bees_dataset])  # ✅
```

### 6.4 图像显示 ⚠️

**错误示例：**
```python
image, label = train_dataset[0]
image = Image.open(image)  # image 已是 PIL 对象，无需再次打开 ❌
```

**正确示例：**
```python
image, label = train_dataset[0]
image.show()  # 直接显示 ✅
```

### 6.5 文件格式兼容性

确保 `os.listdir()` 只筛选图像文件：
```python
f.endswith(('.png', '.jpg', '.jpeg'))  # 支持多种格式
```

---

## 七、完整正确代码

```python
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import os  

class myData(Dataset):
    def __init__(self, root_dir, label_dir, label_map=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.image_path = [f for f in os.listdir(self.path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.label_map = label_map if label_map else {}

    def __getitem__(self, index):
        image_name = self.image_path[index]
        image_path = os.path.join(self.path, image_name)
        image = Image.open(image_path)
        
        if self.label_map:
            label = self.label_map[self.label_dir]
        else:
            label = self.label_dir
        
        return image, label

    def __len__(self):
        return len(self.image_path)

root_dir = 'deeplearn\\ants&bees_data\\train'
label_map = {"ants": 0, "bees": 1}

ants_dataset = myData(root_dir, "ants", label_map=label_map)
bees_dataset = myData(root_dir, "bees", label_map=label_map)
train_dataset = ConcatDataset([ants_dataset, bees_dataset])

image, label = train_dataset[0]
print(f"Image: {image}, Label: {label}")
image.show()
```

---

## 八、扩展建议

### 8.1 添加图像预处理

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = myData(root_dir, "ants", label_map=label_map, transform=transform)
```

### 8.2 使用 DataLoader

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for images, labels in dataloader:
    # 训练逻辑
    pass
```
