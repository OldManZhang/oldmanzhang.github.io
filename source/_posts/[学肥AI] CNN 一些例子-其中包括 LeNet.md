---
title: CNN 一些例子-其中包括 LeNet
date: 2024-08-02 10:18:38
tags: 
    - 学肥AI
    - CNN
    - 深度学习
    - LeNet
categories: 学肥AI
description: "直接应用 CNN 来解决实际问题，动手操作才是学习的根本"
cover: https://qiniu.oldzhangtech.com/cover/%E5%A4%A7%E8%B0%B7%E7%BF%94%E5%B9%B3%202.jpg
---

## 背景问题
前文中已经对 CNN 组件 以及整体的数据流动有了解，有需要的可以翻看之前的[文章](https://blog.csdn.net/weixin_49113487/article/details/140717493?spm=1001.2014.3001.5501)，后面我们直接应用 CNN 来解决实际问题，动手操作才是学习的根本。

本篇的所有的例子，详细的 `ipynb` 都可以从下面链接中找到。 NNvsCNN [ipynb 链接](https://gitee.com/oldmanzhang/resource_machine_learning/blob/master/deep_learning/cnn_compareNN.ipynb)，识别图形 [ipynb 链接](https://gitee.com/oldmanzhang/resource_machine_learning/blob/master/deep_learning/cnn_classify_triangleCycleRectangle.ipynb)。
## NN vs CNN
本例子的目的是：同样面对 `mnist` 手写数字数据集的时候，对比 使用 `NN` 和 使用 `CNN` ，看一下他们表现的差异。可运行文件 `ipynb` 的[链接](https://gitee.com/oldmanzhang/resource_machine_learning/blob/master/deep_learning/cnn_compareNN.ipynb)。
### prepare data
```python
import torch
import torchvision

# %%
transformation = torchvision.transforms.ToTensor()
# 执行调整文件的位置
train_dataset = torchvision.datasets.MNIST(root='../data/mnist/', train=True, download=True, transform=transformation)
test_dataset = torchvision.datasets.MNIST(root='../data/mnist/', train=False, download=True, transform=transformation)

batch_size = 64
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```
### NN 的实现
```python


import matplotlib.pyplot as plt
from torchinfo import summary
import torch.nn as nn
from tqdm import * # tqdm用于显示进度条并评估任务时间开销
import numpy as np
import sys


class NNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        
        self.fcin = nn.Linear(input_size, hidden_sizes[0]) if hidden_sizes else nn.Linear(input_size, output_size)
        self.fcout = nn.Linear(hidden_sizes[-1], output_size) if hidden_sizes else nn.Identity()
        self.relu = nn.ReLU()

        # Create a list of intermediate layers
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
        # Convert the list of layers into a Sequential module
        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, x):
     
        out = self.fcin(x)
        out = self.relu(out)
        
        # Pass through the hidden layers
        if len(self.hidden_layers) > 0:
            out = self.hidden_layers(out)
        
        # 将上一步结果传递给fcout
        out = self.fcout(out)
        # 返回结果
        return out

# %%
input_size = 28*28
output_size = 10
model = NNet(input_size, [512, 512], output_size)
print(summary(model))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def evaluate(model, data_loader):
#     评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.view(-1, input_size)
            logits = model(x)
#             _ 是 value ， predicted 是 index
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

loss_history = []  # 创建损失历史记录列表
acc_history = []   # 创建准确率历史记录列表
num_epochs = 10
for epoch in tqdm(range(num_epochs), file=sys.stdout):
    # 记录损失和预测正确数
    total_loss = 0
    total_correct = 0
    
    model.train()
    for images, labels in train_dataloader:
        # 将图像和标签转换成张量
        # [64, 1, 28, 28] -> [64, 784]
        images = images.view(-1, 28*28)  
        labels = labels.long()
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 记录训练集loss
        total_loss += loss.item()
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_history.append(np.log10(total_loss))    
    accuracy = evaluate(model, test_dataloader)
    acc_history.append(accuracy)
    if(epoch%3 == 0):
        print(f'Epoch {epoch+1}: test accuracy = {acc_history[-1]:.2f}')


# 使用Matplotlib绘制损失和准确率的曲线图
import matplotlib.pyplot as plt
plt.plot(loss_history, label='loss')
plt.plot(acc_history, label='accuracy')
plt.legend()
plt.show()

# 输出准确率
print("Accuracy:", acc_history[-1])
```
![image.png](https://qiniu.oldzhangtech.com/mdpic/55ff4334-4439-494f-aa82-709af8e79db0_d98edf1a-29fd-456c-8a2e-8e4b28464ff6.png)
```python
# 部分的输出
Accuracy: 0.9422
```
### CNN 的实现
```python
# 导入必要的库，torchinfo用于查看模型结构
import torch
import torch.nn as nn
from torchinfo import summary

# 定义LeNet的网络结构
class SimpleCnnNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCnnNet, self).__init__()
#         步长默认为1，填充默认为0 一定要记得
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # 卷积层2：输入6个通道，输出16个通道，卷积核大小为5x5
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 全连接层1：输入16x4x4=256个节点，输出120个节点
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        # 全连接层2：输入120个节点，输出84个节点
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        # 输出层：输入84个节点，输出10个节点
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        # 使用ReLU激活函数，并进行最大池化
        x = torch.relu(self.conv1(x)) 
        # input: 1,28,28, output: 6,24,24
        # output 计算逻辑：(28-5+2*0)/1 + 1 = 24
        x = nn.functional.max_pool2d(x, kernel_size=2)  # output: 6,12,12
        # 使用ReLU激活函数，并进行最大池化
        x = torch.relu(self.conv2(x))  
        # input: 6,12,12, output: 16,8,8
        # output 计算逻辑：(12-5+2*0)/1 + 1 = 8
        x = nn.functional.max_pool2d(x, kernel_size=2)  
        # output: 16,4,4
        
        # 将多维张量展平为一维张量
        x = x.view(-1, 16 * 4 * 4)
        # 全连接层
        x = torch.relu(self.fc1(x))
        # 全连接层
        x = torch.relu(self.fc2(x))
        # 全连接层
        x = self.fc3(x)
        return x
    
# 查看模型结构及参数量，input_size表示示例输入数据的维度信息
print(summary(SimpleCnnNet(), input_size=(1, 1, 28, 28)))

# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import * # tqdm用于显示进度条并评估任务时间开销
import numpy as np
import sys

# 设置随机种子
torch.manual_seed(0)

# 定义模型、优化器、损失函数
model = SimpleCnnNet()
optimizer = optim.SGD(model.parameters(), lr=0.02)
criterion = nn.CrossEntropyLoss()

# 设置数据变换和数据加载器
transform = transforms.Compose([
    transforms.ToTensor(),  # 将数据转换为张量
])


# 设置epoch数并开始训练
num_epochs = 10  # 设置epoch数
loss_history = []  # 创建损失历史记录列表
acc_history = []   # 创建准确率历史记录列表

# tqdm用于显示进度条并评估任务时间开销
for epoch in tqdm(range(num_epochs), file=sys.stdout):
    # 记录损失和预测正确数
    total_loss = 0
    total_correct = 0
    
    # 批量训练
    model.train()
    for inputs, labels in train_dataloader:

        # 预测、损失函数、反向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # 记录训练集loss
        total_loss += loss.item()
    
    # 测试模型，不计算梯度
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:

            # 预测
            outputs = model(inputs)
            # 记录测试集预测正确数
            total_correct += (outputs.argmax(1) == labels).sum().item()
        
    # 记录训练集损失和测试集准确率
    loss_history.append(np.log10(total_loss))  # 将损失加入损失历史记录列表，由于数值有时较大，这里取对数
    acc_history.append(total_correct / len(test_dataset))# 将准确率加入准确率历史记录列表
    
    # 打印中间值
    if epoch % 2 == 0:
        tqdm.write("Epoch: {0} Loss: {1} Acc: {2}".format(epoch, loss_history[-1], acc_history[-1]))

# 使用Matplotlib绘制损失和准确率的曲线图
import matplotlib.pyplot as plt
plt.plot(loss_history, label='loss')
plt.plot(acc_history, label='accuracy')
plt.legend()
plt.show()

# 输出准确率
print("Accuracy:", acc_history[-1])
```

![image.png](https://qiniu.oldzhangtech.com/mdpic/4413e664-0d64-4025-8085-61dfb55dd43d_069afa8c-7ef4-416f-ac1b-9f499a31c7a5.png)
```python
# 部分的输出
Accuracy: 0.9832
```
### 对比效果
```python
# 使用Matplotlib绘制损失和准确率的曲线图
import matplotlib.pyplot as plt
plt.plot(nn_loss_history, label='nn loss')
plt.plot(nn_acc_history, label='nn accuracy')
plt.plot(cnn_loss_history, label='cnn loss')
plt.plot(cnn_acc_history, label='cnn accuracy')
plt.legend()
plt.show()

# 输出准确率
print("ACCURACY:")
print("nn:", nn_acc_history[-1])
print("cnn:", cnn_acc_history[-1])

# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

nn_total_params = count_parameters(nn_model)
cnn_total_params = count_parameters(cnn_model)


print("count of PARAMETERS:")
print("nn:", nn_total_params)
print("cnn:", cnn_total_params)
```
![image.png](https://qiniu.oldzhangtech.com/mdpic/3e483fa6-5eec-4eb5-8ce9-6cfc2967bd9b_3874b442-7387-4608-bf5a-0ff2e2b34337.png)
```python
ACCURACY:
nn: 0.9422
cnn: 0.9832

count of PARAMETERS:
nn: 669706
cnn: 44426
```

🔥 从图中可以看到 `CNN` 比 `NN` 的 **准确度更加高**， 同时他的使用的 **参数量会更加的少**。

💡 当然模型应用不同的超参数会有不同的效果，可以对 `NN` 和 `CNN` 进行修改，看一下效果如何。同时 [链接](https://gitee.com/oldmanzhang/resource_machine_learning/blob/master/deep_learning/cnn_compareNN.ipynb) 中已经有 `girdSearch` 的方法，用于寻找最佳的 `NN` 参数。

## LeNet 网络 识别手写数字
参考了别人的各种卷积神经网络的时间线贴图。
![](https://qiniu.oldzhangtech.com/mdpic/20431588-e607-4106-817c-68d54848a902_83fc8c68-cce1-4663-85a0-e94f939f9052.png)
`LeNet` 为卷积神经网网络奠定了基石。所有的后续的网络都是基于这个网络进行扩展的。
### 整体框架
![](https://qiniu.oldzhangtech.com/mdpic/b990ca3c-7939-4610-b9d8-cafc9586a8bd_5119f828-a66b-4e1d-a03b-7703add2107a.png)
`LeNet` 是由 多个卷积层，池化层，全连接层 组合构建而来。
结构并不复杂，下面就可以手动去实现这个网络。如果对 `卷积`，`池化` 并不熟悉的，可以前往 [文章](https://blog.csdn.net/weixin_49113487/article/details/140717493) 进行了解。
### 手写 LeNet 网络
其实手写 `LeNet`  就是 上述例子中的 `CNN` 的 `SimpleCnnNet`。下面列出部分的代码片段。

```python
class SimpleCnnNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCnnNet, self).__init__()
#         步长默认为1，填充默认为0 一定要记得
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # 卷积层2：输入6个通道，输出16个通道，卷积核大小为5x5 
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 全连接层1：输入16x4x4=256个节点，输出120个节点
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        # 全连接层2：输入120个节点，输出84个节点
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        # 输出层：输入84个节点，输出10个节点
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        # 使用ReLU激活函数，并进行最大池化
        x = torch.relu(self.conv1(x)) 
        # input: 1,28,28, output: 6,24,24
        # output 计算逻辑：(28-5+2*0)/1 + 1 = 24
        x = nn.functional.max_pool2d(x, kernel_size=2)  # output: 6,12,12
        # 使用ReLU激活函数，并进行最大池化
        x = torch.relu(self.conv2(x))  
        # input: 6,12,12, output: 16,8,8
        # output 计算逻辑：(12-5+2*0)/1 + 1 = 8
        x = nn.functional.max_pool2d(x, kernel_size=2)  
        # output: 16,4,4
        
        # 将多维张量展平为一维张量
        x = x.view(-1, 16 * 4 * 4)
        # 全连接层
        x = torch.relu(self.fc1(x))
        # 全连接层
        x = torch.relu(self.fc2(x))
        # 全连接层
        x = self.fc3(x)
        return x
```
由上面定义的模型的结构得知：
`Feature extraction`： 由 2 *【卷积 *  激活   * 池化】组成，每层的卷积和池化的参数有不同。用于抽取特征。
`Classification`： 由 2 *【全连接层 * 激活】 组成。对特征进行逻辑关联。

## 识别三角形/圆形/正方形
目的：生成随机的 三角形/圆形/正方形 的黑白图数据集，并用 CNN 的模型识别他们。可运行 ipynb 的文件[链接](https://gitee.com/oldmanzhang/resource_machine_learning/blob/master/deep_learning/cnn_classify_triangleCycleRectangle.ipynb)。
_Prepare data_
```python
# !pip install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_shape(shape, img_size=1000):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    if shape == 'circle':
        radius = np.random.randint(img_size // 8, img_size // 4)
        center = (np.random.randint(radius, img_size - radius), np.random.randint(radius, img_size - radius))
        cv2.circle(img, center, radius, 255, -1)
    elif shape == 'triangle':
        pt1 = (np.random.randint(0, img_size), np.random.randint(0, img_size))
        pt2 = (np.random.randint(0, img_size), np.random.randint(0, img_size))
        pt3 = (np.random.randint(0, img_size), np.random.randint(0, img_size))
        points = np.array([pt1, pt2, pt3])
        cv2.drawContours(img, [points], 0, 255, -1)
    elif shape == 'rectangle':
        pt1 = (np.random.randint(0, img_size // 2), np.random.randint(0, img_size // 2))
        pt2 = (np.random.randint(img_size // 2, img_size), np.random.randint(img_size // 2, img_size))
        cv2.rectangle(img, pt1, pt2, 255, -1)
    return img

def generate_dataset(num_samples=1000, img_size=1000):
    shapes = ['circle', 'triangle', 'rectangle']
    data = []
    labels = []
    for _ in range(num_samples):
        shape = np.random.choice(shapes)
        img = create_shape(shape, img_size)
        data.append(img)
        labels.append(shapes.index(shape))
    return np.array(data), np.array(labels)
```

可视化图像。
```python
# Visualize some samples
# img_size = 200 是比较好的体现出 形状
fig, axes = plt.subplots(1, 3, figsize=(10, 5))
for i, shape in enumerate(['circle', 'triangle', 'rectangle']):
    axes[i].imshow(create_shape(shape, img_size=200), cmap='gray')
    axes[i].set_title(shape)
    axes[i].axis('off')
plt.show()
```
![image.png](https://qiniu.oldzhangtech.com/mdpic/4a51309b-3fe5-4280-9bbd-da97b8c6925d_bf9181d6-8b22-499c-a013-52ec993b5c47.png)

随机检查数据集，检查数据是否合理。
```python
num = 5
ran_indexes = np.random.randint(0, len(data), num)
fig, axes = plt.subplots(1, num, figsize=(10, 5))
for i in range(num):
    axes[ i].imshow(data[ran_indexes[i]], cmap='gray')  ## 'Axes' object is not subscriptable
    axes[ i].set_title(f'index : {ran_indexes[i]} {labels[ran_indexes[i]]}')
    axes[ i].axis('off')
plt.show()
# 0: cycle, 1: triangle, 2: rectangle
```
![image.png](https://qiniu.oldzhangtech.com/mdpic/0e26b843-b82c-4cb3-8617-54f7eb265e20_f96bca9b-901b-4724-bd06-6b9fab029a9a.png)

_define model_
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ShapeDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # output:32*200*200
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # output:32*50*50
        self.pool = nn.MaxPool2d(2, 2)
        # output:64*50*50
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # output:128
        self.fc1 = nn.Linear(64 * 50 * 50, 128)  # Adjust input size for the new image size
        # output: 3
        self.fc2 = nn.Linear(128, 3)  # 3 classes: circle, triangle, rectangle

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 50 * 50)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

_train model_
```python
# Data transformation
transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),  # Ensure single channel for grayscale images
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create the dataset and data loaders
train_dataset = ShapeDataset(data, labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)



# Instantiate the model
model = SimpleCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# validate dataset
val_data, val_labels = generate_dataset(num_samples=200, img_size=200)
val_dataset = ShapeDataset(val_data, val_labels, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



# Training method
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for _data, _labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(_data)

            # print(_data.shape, _labels.shape, outputs.shape)

            loss = criterion(outputs, _labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
        validate(model, val_loader, criterion)

    print('Finished Training')

# Validation method
def validate(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for _data, _labels in val_loader:
            

            # Forward pass
            outputs = model(_data)

            # print(_data.shape, _labels.shape, outputs.shape)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += _labels.size(0)
            correct += (predicted == _labels).sum().item()

            # Optional: Calculate loss if needed
            # loss = criterion(outputs, _labels)
            # val_loss += loss.item()

    # Optional: Calculate average loss if needed
    # avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')


# Train the model with validation
num_epochs = 20
train(model, train_loader, val_loader, criterion, optimizer, num_epochs)

'''
Epoch [17/20], Loss: 0.0005
Accuracy: 92.00%
Epoch [18/20], Loss: 0.0005
Accuracy: 92.00%
Epoch [19/20], Loss: 0.0004
Accuracy: 92.00%
Epoch [20/20], Loss: 0.0003
Accuracy: 92.00%
Finished Training
'''

```
_evalue_
```python
# Generate test data
test_data, test_labels = generate_dataset(num_samples=100, img_size=200)

# Create the test dataset and data loader
test_dataset = ShapeDataset(test_data, test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluate the model
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        # Ensure inputs have the correct shape
        inputs, labels = inputs.float(), labels
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: { correct / total:.2f}%')
'''
Accuracy of the model on the test images: 0.90%

'''
```
_NOTE_： 准确率还是可以的。

_推理并且视觉验证_
```python

# Visualize some test samples and their predictions
def visualize_predictions(inputs, labels, predictions):
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        axes[i].imshow(inputs[i].squeeze(), cmap='gray')
        axes[i].set_title(f'True: {labels[i]}, Pred: {predictions[i]}')
        axes[i].axis('off')
    plt.show()

# Generate test data
visual_data, visual_labels = generate_dataset(num_samples=5, img_size=200)

# Create the test dataset and data loader
visual_dataset = ShapeDataset(visual_data, visual_labels, transform=transform)
visual_loader = DataLoader(visual_dataset, batch_size=5, shuffle=False)

# torch.from_numpy(visual_data).float().unsqueeze(1).shape = 5,1,200,200
outputs = model(torch.from_numpy(visual_data).float().unsqueeze(1))
_, predictions = torch.max(outputs, 1)
visualize_predictions(visual_data, visual_labels, predictions.numpy())
```
![image.png](https://qiniu.oldzhangtech.com/mdpic/03db3b3d-0d90-4dac-9888-fcf3f9c809c2_39713376-9e7f-4693-93da-b60856f1a06f.png)



## NOTE
`CNN` 是一种有效的 `解决空间` 信息的深度学习网络，特别适用于图像问题的解决中。

