import math

import torch.nn as nn  # pytorch torch.nn
import torch.nn.functional as F  # pytorch torch.nn.functional
from torch.nn import init
from torchvision import models   # torchvision

from pytorchcv.model_provider import get_model as ptcv_get_model  # pytorchcv package

# torch.nn.Module 所有神经网络模块的基类；
class ComboNet(nn.Module):
    """
    definition of ComboNet
    """

    def __init__(self, num_out=5, backbone_net_name='SEResNeXt50'):
        super(ComboNet, self).__init__()  # 在对子类进行赋值之前，必须先对父类进行__init__()调用。

        if backbone_net_name == 'SEResNeXt50':
            # 获取预训练的模型
            seresnext50 = ptcv_get_model("seresnext50_32x4d", pretrained=True)

            num_ftrs = seresnext50.output.in_features
            self.backbone = seresnext50.features
        elif backbone_net_name == 'ResNet18':
            # torchvision.models
            resnet18 = models.resnet18(pretrained=True)
            num_ftrs = resnet18.fc.in_features
            self.backbone = resnet18

        self.backbone_net_name = backbone_net_name
        self.regression_branch = nn.Linear(num_ftrs, 1)
        self.classification_branch = nn.Linear(num_ftrs, num_out)

    def forward(self, x):
        if self.backbone_net_name == 'SEResNeXt50':
            feat = self.backbone(x)
            feat = feat.view(-1, self.num_flat_features(feat))
        elif self.backbone_net_name == 'ResNet18':
            for name, module in self.backbone.named_children():
                if name != 'fc':
                    x = module(x)
            feat = x.view(-1, self.num_flat_features(x))

        regression_output = self.regression_branch(feat)
        classification_output = self.classification_branch(feat)

        return regression_output, classification_output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
