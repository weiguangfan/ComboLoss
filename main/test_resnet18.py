from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.autograd import Variable

net = ptcv_get_model("resnet18", pretrained=True)
print(net)
print("*" * 80)
print(net.output)
print("*" * 80)
print(net.features)
print("*" * 80)
x = Variable(torch.randn(1, 3, 224, 224))
print(x)
print("*" * 80)
y = net(x)
print(y)











