import torch
from torchviz import make_dot
import torch.nn as nn

# 假设你的模型已经定义好，这里假设为 UNetModel
class UNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv1(x)

model = UNetModel()
x = torch.randn(1, 3, 256, 256)  # 输入的随机张量，根据实际情况修改
y = model(x)
dot = make_dot(y, params=dict(model.named_parameters()))
dot.render('unet_model', format='png')