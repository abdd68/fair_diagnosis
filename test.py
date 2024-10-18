from pytorch_grad_cam import GradCAM
from torchvision.models import resnet50
 
# 加载预训练的 ResNet50 模型
model = resnet50(pretrained=True)
 
# 选择目标层
target_layers = [model.layer4[-1]]
 
# 创建 GradCAM 对象
cam = GradCAM(model=model, target_layers=target_layers)
 
# 输入图像（假设您已经有一个输入张量 input_tensor）
# grayscale_cam = cam(input_tensor=input_tensor)
 
# 打印 GradCAM 对象
print(cam)