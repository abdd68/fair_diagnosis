import torch.nn as nn
import torchvision.models as models

# 生成器
class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)
    
class Deployed_model(nn.Module):
    def __init__(self, num_classes=10):
        super(Deployed_model, self).__init__()
        
        # 加载 ResNet18 的特征提取部分
        resnet = models.resnet18(pretrained=True)
        # 保存每一层的引用
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的全连接层
        
        # 标签预测器 (f)
        self.label_predictor = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # 展平
        
        # 标签预测
        label_output = self.label_predictor(features)
        return label_output
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 卷积层定义
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # 假设输入为 224x224
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x, label = None):
        # 卷积层 + 批归一化 + 激活 + 池化
        if not (label is None):
            x = x + label.view(-1, 1, 1, 1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        # 卷积层定义
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # 假设输入为 224x224
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # 卷积层 + 批归一化 + 激活 + 池化
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    