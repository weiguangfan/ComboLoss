"""
@Author: LucasX
@Time: 2021/01/17
@Desc: extract compressed deep features via residual conv auto encoder
"""
import os
import sys
import argparse
import pickle

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from skimage import io

sys.path.append('../')
from models.resconvdae import *

args = argparse.ArgumentParser()
args.add_argument('-ckpt', help='checkpoint of pretrained ResDAE', type=str, default='./model/ResConvDAE.pth')
args.add_argument('-img_dir', help='image directory', type=str, default='/home/xulu/DataSet/Face/SCUT-FBP/Crop')
args.add_argument('-save_to_dir', help='image directory', type=str, default='./features')
args = vars(args.parse_args())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def extract_features_with_dae(img_f, model):
    """
    extract deep features with ResConvDAE
    :param img_f:
    :param model:
    :return:
    """
    # torch.nn.Module.eval():将模块设置为评估模式。
    model.eval()
    # skimage.io.imread():从文件中加载一个图像，返回一个数组
    img = io.imread(img_f)
    # numpy.ndarray.astype():数组的拷贝，映射成指定的类型。
    # PIL.Image.fromarray:从一个输出数组接口的对象中创建一个图像存储器（使用缓冲区协议）。
    img = Image.fromarray(img.astype(np.uint8))
    # torchvision.transforms.Compose():将几个变换组合在一起。
    # transforms.Resize():将输入的图像调整到给定的尺寸。
    # transforms.ToTensor():将PIL图像或numpy.ndarray转换为张量。
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # 将输入图像进行组合变换：尺寸改变、转换为tensor
    img = preprocess(img)
    # torch.unsqueeze():返回一个新的张量，在指定的位置插入一个尺寸为1的张量。
    # 返回的张量与这个张量共享相同的基础数据。
    img.unsqueeze_(0)
    # torch.to():从self.to(*args, **kwargs)的参数中推断出Torch.dtype和Torch.device。
    img = img.to(device)
    output = model.encoder(img)
    # torch.to():从self.to(*args, **kwargs)的参数中推断出Torch.dtype和Torch.device。
    # torch.Tensor.detach():返回一个新的张量，从当前图形中分离出来。结果将永远不需要梯度。
    # torch.Tensor.numpy():将自我张量作为NumPy的ndarray返回。
    # numpy.astype():数组的拷贝，映射成指定的类型。
    # numpy.ndarray.transpose():返回一个转轴的数组视图。
    # numpy.ndarray.ravel():返回一个连续的扁平化的数组。
    feat = output.to("cpu").detach().numpy().astype(np.float)[0].transpose([1, 2, 0]).ravel()

    os.makedirs(args['save_to_dir'], exist_ok=True)
    with open('./{}/gen_{}'.format(args['save_to_dir'], img_f.split(os.path.sep)[-1].split('.')[0] + '.pkl'),
              'wb') as f:

        pickle.dump(feat, f)
        print(f'extract deep features for {os.path.basename(img_f)} successfully...')


if __name__ == '__main__':
    # 实例化对象
    resconvdae = ResConvDAE()
    model_name = resconvdae.__class__.__name__
    # Module.float():将所有的浮点参数和缓冲区转换为float数据类型。
    resconvdae = resconvdae.float()
    # Module.to():这个方法将只把浮点或复杂的参数和缓冲区转换为dtype（如果给定）。
    resconvdae = resconvdae.to(device)
    print('Start extracting deep features...')
    # Module.load_state_dict():missing_keys是一个包含缺失键的str列表。
    # unexpected_keys 是一个包含意外键的str列表。
    resconvdae.load_state_dict(torch.load(args['ckpt']))
    for img_f in os.listdir(args['img_dir']):
        extract_features_with_dae(os.path.join(args['img_dir'], img_f), resconvdae)
