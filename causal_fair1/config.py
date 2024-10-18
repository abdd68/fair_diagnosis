from torchvision import transforms

# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

