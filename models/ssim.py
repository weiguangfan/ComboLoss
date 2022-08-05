import sys
import torch
import torch.nn.functional as F  # pytorch torch.nn.functional
from math import exp
from torch.autograd import Variable  # pytorch torch.autograd.Variable

sys.path.append('../')
from config.cfg import cfg


def gaussian(window_size, sigma):
    # 高斯分布公式
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])

    return gauss / gauss.sum()


def create_window(window_size, channel):
    """创建通道过滤器"""
    # 调用函数gaussian()
    # Tensor.unsqueeze():返回一个新的张量，在指定的位置插入一个尺寸为1的张量。
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    # Tensor.mm():对输入的矩阵和mat2进行矩阵乘法。
    # tensor.t():希望输入的是<=2-D张量，并将0和1维转置。
    # tensor.float():数据类型为torch.float32
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # Tensor.expand():返回一个新的自我张量的视图，并将单子维度扩展到一个更大的尺寸。
    # Tensor.contiguous():返回一个连续的内存中的张量，包含与self张量相同的数据。
    # torch.autograd.Variable():Autograd自动支持将requires_grad设置为True的张量。
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    # 返回张量
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # 预测输入进入卷积层：torch.nn.functional.conv2d():对一个由多个输入平面组成的输入图像进行二维卷积。
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    # 真实输入进入卷积层：torch.nn.functional.conv2d():对一个由多个输入平面组成的输入图像进行二维卷积。
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    # Tensor.pow():读取输入的每个元素的幂，并返回一个带有结果的张量。
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    # 相当于torch.mul():计算两个张量的乘积
    mu1_mu2 = mu1 * mu2
    # 预测输入进入sigmoid层
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    # 真实输入进入sigmoid层
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    # 预测和真实输入进入sigmoid层
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    # 计算映射值
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        # Tensor.mean():返回输入张量中所有元素的平均值。
        return ssim_map.mean()
    else:
        # Tensor.mean():返回输入张量的每一行在给定维度dim中的平均值。
        return ssim_map.mean(1).mean(1).mean(1)

# torch.nn.Module 所有神经网络模块的基类；
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        # 在对子类进行赋值之前，必须先对父类进行__init__()调用。
        super(SSIMLoss, self).__init__()
        # 将window_size赋值给类属性self.window_size
        self.window_size = window_size
        # 将size_average赋值给类属性self.size_average
        self.size_average = size_average

        self.channel = 1
        # 调用函数create_window():
        self.window = create_window(window_size, self.channel)

    def forward(self, pred, gt):
        # Tensor.size():返回自我张量各个维度的大小。
        (_, channel, _, _) = pred.size()
        # 条件判断：通道和数据类型符合条件
        # Tensor.type():如果没有提供type，则返回类型，否则将此对象转换为指定类型。
        if channel == self.channel and self.window.data.type() == pred.data.type():
            window = self.window
        else:
            # 调用函数create_window():返回张量
            window = create_window(self.window_size, channel)
            # 条件判断：cfg字典中键‘use_gpu’的值是否为True
            # torch.cuda.is_available():返回一个bool，表示CUDA当前是否可用。
            if cfg['use_gpu'] and torch.cuda.is_available():
                # Tensor.get_device():对于CUDA张量，该函数返回张量所在的GPU的设备序号。
                # Tensor.cuda():返回此对象在CUDA内存中的副本。
                window = window.cuda(pred.get_device())
            # Tensor.type_as():返回这个张量，并将其转换为给定张量的类型。
            window = window.type_as(pred)

            self.window = window
            self.channel = channel
        # 调用函数__ssim():返回映射张量
        return _ssim(pred, gt, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    # Tensor.size():返回自我张量各个维度的大小。
    (_, channel, _, _) = img1.size()
    # 调用函数create_window():返回张量
    window = create_window(window_size, channel)
    # torch.Tensor.is_cuda():如果张量存储在GPU上，则为真，否则为假。
    if img1.is_cuda:
        # Tensor.get_device():对于CUDA张量，该函数返回张量所在的GPU的设备序号。
        # Tensor.cuda():返回此对象在CUDA内存中的副本。
        window = window.cuda(img1.get_device())
    # Tensor.type_as():返回这个张量，并将其转换为给定张量的类型。
    window = window.type_as(img1)
    # 调用__ssim()函数：
    return _ssim(img1, img2, window, window_size, channel, size_average)
