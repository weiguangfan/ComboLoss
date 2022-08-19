from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.autograd import Variable

net = ptcv_get_model("seresnext50_32x4d", pretrained=True)
# print(net)
print("*" * 80)
print(net.output)
print("*" * 80)
# print(net.features)
# print("*" * 80)
# for name,module in net.features.named_children():
#     print(name,'-->',module)
# print("*" * 80)
for index_,module in net.features.named_modules():
    print(index_,'-->',module)
print("*" * 80)
# for index_,name in enumerate(net.features.named_modules()):
#     print(index_,'-->',name)
# print("*" * 80)



