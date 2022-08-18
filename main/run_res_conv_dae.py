import copy  # python copy module
import os  # pytho os module
import sys  # python sys module
import time  # python time module
import math  # python math module

import cv2  # pypi
import numpy as np  # numpy package
import pandas as pd  # pandas package
from PIL import Image  # python Pillow package
from scipy import spatial  # scipy package
import torch  # PyTorch torch module
import torch.nn as nn
import torch.nn.functional as F  # PyTorch torch.nn.functional module
import torch.optim as optim  # PyTorch torch.optim module
from torch.optim import lr_scheduler  # PyTorch torch.optim.lr_scheduler
from torchvision import transforms  # torchvision package

sys.path.append('../')
from models import ssim
from models.resconvdae import *   # 从路径文件 models/resconvdae.py 导入类和函数
from models.losses import ReconstructionLoss
from data.data_loaders import load_reconstruct_scutfbp, load_reconstruct_hotornot, load_reconstruct_scutfbp5500_64, \
    load_reconstruct_scutfbp5500_cv
from util.file_util import mkdirs_if_not_exist
from config.cfg import cfg


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, inference=False):
    """
    train model
    :param model:ResConvDAE()的一个实例对象
    :param dataloaders:一系列数据加载类的实例
    :param criterion:实例化一个损失函数类
    :param optimizer:实例化一个优化算法类
    :param scheduler:实例化一个学习率调度类
    :param num_epochs:遍历完一遍所有样本称为一个epoch，epoch的数目
    :param inference:推断状态False
    :return:
    """
    # model:ResConvDAE类的一个实例对象
    print(model)
    # 获取类名
    # instance.__class__:一个类实例所属的类。
    # definition.__name__:类、函数、方法、描述符或生成器实例的名称。
    model_name = model.__class__.__name__
    # 转换模块的数据类型
    # torch.nn.Module.float():将所有的浮点参数和缓冲区转换为float数据类型。
    model = model.float()
    # 判定运行设备类型，并设定
    # torch.cuda.is_available():返回一个bool，表示CUDA当前是否可用。
    # torch.device()：是一个代表设备的对象，torch.Tensor已经或将要被分配到这个设备上。
    # device全局变量：条件判断cuda是否可用
    device = torch.device('cuda:0' if torch.cuda.is_available() and cfg['use_gpu'] else 'cpu')
    # 根据设备类型和数量，决定模块是否采用并行方式
    # torch.cuda.device_count():返回可用的GPU的数量。
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # torch.nn.DataParaller():在模块层面上实施数据并行化。
        model = nn.DataParallel(model)
    # 参数移动到指定设备
    # torch.nn.Moudle.to():移动 and\or 映射参数parameters和缓冲区buffers。
    model = model.to(device)
    # 获取样本集大小，构造键值对，返回字典
    # 遍历列表内字符串，作为键值，到dataloaders字典中查找，返回对应键值的值；值为训练、测试、验证等数据集的划分；
    # torch.utils.data.DataLoader().dataset:.dataset是属性吗？
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}
    # 遍历键值对，并打印信息
    for k, v in dataset_sizes.items():
        print('Dataset size of {0} is {1}...'.format(k, v))
    # 默认执行，inference=False
    if not inference:
        print('Start training %s...' % model_name)
        ##############################################
        # 计时开始
        # time.time():以浮点数的形式返回自纪元以来的时间（秒）。
        since = time.time()
        # torch.nn.Module.state_dict():返回一个包含模块整体状态的字典。
        # copy.deepcopy():返回x的一个深度拷贝。
        best_model_wts = copy.deepcopy(model.state_dict())
        # 定义初始值
        best_ssim = 0.0
        best_cosine_similarity = 0.0  # 余弦相似度
        best_l2_dis = float('inf')  # 无穷大
        # 配置文件设置的num_epochs=200
        # 遍历0-199
        for epoch in range(num_epochs):
            print('-' * 100)
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            # 每一个epoch分为train，评估两个阶段
            for phase in ['train', 'val']:
                ##############################################
                # 训练阶段
                if phase == 'train':
                    # 版本小于1.1.0，就启用学习率调度程序
                    # torch.__version__:获取当前torch的版本号
                    if torch.__version__ <= '1.1.0':
                        # torch.optim.lr_scheduler.StepLR.step():学习率调度器，每个调度器都是在前一个调度器获得的学习率上一个接一个地应用。
                        scheduler.step()
                    # torch.nn.Module.train():将模块设置为训练模式。
                    model.train()
                # 评估阶段
                else:
                    # torch.nn.Module.eval():将模块设置为评估模式。
                    model.eval()
                ##############################################
                # 设置初始参数
                running_loss = 0.0  # 运行损失值
                running_ssim = 0.0
                running_l2_dis = 0.0  # l2范数
                running_cos_sim = 0.0  # 余弦相似度

                # 在数据集上面迭代
                # dataloaders[phase]:阶段字符串作为键值，在字典dataloaders中查找键对应的值，返回训练、测试、评估划分数据集
                # enumerate():返回一个元组，该元组包含一个计数（从开始，默认为 0）和在 iterable 上迭代得到的值。
                for i, data in enumerate(dataloaders[phase], 0):
                    # 获取数据集的真实标签
                    inputs = data['image']
                    # torch.nn.Moudle.to():移动 and\or 映射参数parameters和缓冲区buffers。
                    inputs = inputs.to(device)

                    # 将参数梯度置零
                    # Optimizer.zero_grad():将所有优化的Torch.Tensor s的梯度设置为零。
                    optimizer.zero_grad()

                    # 前向传播
                    # 训练过程，追踪历史
                    # torch.set_grad_enabled():将根据其参数模式来启用或禁用梯度。
                    with torch.set_grad_enabled(phase == 'train'):
                        # 计算预测值
                        outputs = model(inputs)
                        # 计算损失值
                        loss = criterion(outputs, inputs)

                        # 训练过程，反向传播 + 优化
                        if phase == 'train':
                            # 损失求和并反向传播
                            loss.sum().backward()
                            # Optimizer.step():优化器更新，执行一个单一的优化步骤（参数更新）。
                            optimizer.step()

                    # 统计:单个样本的损失和，乘以总样本数
                    # torch.tensor.size(dim=None):返回自我张量的大小。
                    # torch.tensor.sum():返回输入张量中所有元素的总和。
                    running_loss += loss.sum() * inputs.size(0)
                    # torch.to():从self.to(*args, **kwargs)的参数中推断出Torch.dtype和Torch.device。
                    # torch.Tensor.detach():返回一个新的张量，从当前图形中分离出来。结果将永远不需要梯度。
                    # torch.Tensor.numpy():将自我张量作为NumPy的ndarray返回。
                    # numpy.ndarray.ravel():返回一个压平的数组。
                    # scipy.spatial.distance.cosine():计算一维数组之间的余弦距离。
                    # 相似度和相似距离之和，等于1；
                    running_cos_sim += 1 - spatial.distance.cosine(outputs.to('cpu').detach().numpy().ravel(),
                                                                   inputs.to('cpu').detach().numpy().ravel())
                    # numpy.linalg.norm()：矩阵或向量规范。
                    running_l2_dis += np.linalg.norm(
                        outputs.to('cpu').detach().numpy().ravel() - inputs.to('cpu').detach().numpy().ravel())
                    # 从导入文件中，调用函数ssim():返回什么？
                    running_ssim += ssim.ssim(outputs, inputs)
                ##############################################
                # 是否重复？另外，版本号，究竟是小于还是大于？
                # 训练阶段
                if phase == 'train':
                    # torch.__version__:获取当前torch的版本号
                    # 版本小于1.1.0，就启用学习率调度程序
                    if torch.__version__ >= '1.1.0':
                        # torch.optim.lr_scheduler.StepLR.step():学习率调度器，每个调度器都是在前一个调度器获得的学习率上一个接一个地应用。
                        scheduler.step()
                ##############################################
                # 求单个epoch上述指标的平均值？
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_l2_dis = running_l2_dis / dataset_sizes[phase]
                epoch_cos_sim = running_cos_sim / dataset_sizes[phase]
                epoch_ssim = running_ssim / dataset_sizes[phase]

                print('{} Loss: {:.4f} L2_Distance: {} Cosine_Similarity: {} SSIM: {}'
                      .format(phase, epoch_loss, epoch_l2_dis, epoch_cos_sim, epoch_ssim))
                ##############################################
                # 评估阶段
                # deep copy the model
                if phase == 'val' and epoch_l2_dis <= best_l2_dis:
                    # 重新赋值best_l2_dis
                    best_l2_dis = epoch_l2_dis
                    ##############################################
                    # torch.nn.Module.state_dict():返回一个包含模块整体状态的字典。
                    # copy.deepcopy():返回x的一个深度拷贝。
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # torch.nn.Module.load_state_dict():将state_dict的参数和缓冲区复制到本模块及其子模块中。
                    model.load_state_dict(best_model_wts)
                    # 确定路径
                    model_path_dir = './model'
                    # 打开导入文件函数mkdirs_if_not_exist():条件判断路径非目录和路径不存在成立，就递归目录创建
                    mkdirs_if_not_exist(model_path_dir)
                    # torch.cuda.device_count():返回可用的GPU的数量。
                    # torch.nn.Module().state_dict():返回一个包含模块整体状态的字典。
                    state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
                    # torch.save:将一个对象保存到一个磁盘文件。
                    torch.save(state_dict, './model/{0}_best_epoch-{1}.pth'.format(model_name, epoch))
        ##############################################
        # 计算结束
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best L2_Distance: {:4f}'.format(best_l2_dis))
        ##############################################
        # 跟评估模式代码一致，是要覆盖？
        # load best model weights
        # torch.nn.Module().load_state_dict:将state_dict的参数和缓冲区复制到本模块及其子模块中。
        model.load_state_dict(best_model_wts)
        # 确定路径
        model_path_dir = './model'
        # 打开导入文件函数mkdirs_if_not_exist():条件判断路径非目录和路径不存在成立，就递归目录创建
        mkdirs_if_not_exist(model_path_dir)
        # torch.cuda.device_count():返回可用的GPU的数量。
        # torch.nn.Module().state_dict():返回一个包含模块整体状态的字典。
        state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
        # torch.save:将一个对象保存到一个磁盘文件。
        torch.save(state_dict, './model/%s.pth' % model_name)
        ##############################################
    else:
        print('Start testing %s...' % model.__class__.__name__)
        # os.path.join():智能地连接一个或多个路径组件。
        # torch.load():从文件中加载一个用 torch.save() 保存的对象。
        # torch.nn.Module().load_state_dict():将state_dict的参数和缓冲区复制到本模块及其子模块中。
        model.load_state_dict(torch.load(os.path.join('./model/%s.pth' % model_name)))
    ##############################################
    # torch.nn.Module.eval():将模块设置为评估模式。
    model.eval()
    # 初始值为0
    cos_sim, l2_dist, ssim_ = 0.0, 0.0, 0.0
    # torch.no_grad():禁用梯度计算的上下文管理器。
    with torch.no_grad():
        # 加载测试集
        for data in dataloaders['test']:
            images = data['image']
            # 参数移动到指定设备
            # torch.nn.Moudle.to():移动 and\or 映射参数parameters和缓冲区buffers。
            images = images.to(device)
            outputs = model(images)
            # torch.to():从self.to(*args, **kwargs)的参数中推断出Torch.dtype和Torch.device。
            # torch.Tensor.detach():返回一个新的张量，从当前图形中分离出来。结果将永远不需要梯度。
            # torch.Tensor.numpy():将自我张量作为NumPy的ndarray返回。
            # numpy.ndarray.ravel():返回一个压平的数组。
            # scipy.spatial.distance.cosine():计算一维数组之间的余弦距离。
            # 相似度和相似距离之和，等于1；
            cos_sim += 1 - spatial.distance.cosine(outputs.to('cpu').detach().numpy().ravel(),
                                                   images.to('cpu').detach().numpy().ravel())
            # numpy.linalg.norm()：矩阵或向量规范。
            l2_dist += np.linalg.norm(
                outputs.to('cpu').detach().numpy().ravel() - images.to('cpu').detach().numpy().ravel())
            # 从导入文件中，调用函数ssim():返回什么？
            ssim_ += ssim.ssim(outputs, images)

    print('*' * 200)
    print('Avg L2 Distance of {0} on test set: {1}'.format(model_name, l2_dist / dataset_sizes['test']))
    print('Avg CosineSimilarity of {0} on test set: {1}'.format(model_name, cos_sim / dataset_sizes['test']))
    print('Avg SSIM of {0} on test set: {1}'.format(model_name, ssim_ / dataset_sizes['test']))
    print('*' * 200)


def main(model, data_name):
    """
    train model
    :param model:ResConvDAE()的一个实例对象
    :param data_name: SCUT-FBP/HotOrNot/SCUT-FBP5500/SCUT-FBP5500CV，其中之一的字符串
    :return:
    """
    # criterion = ReconstructionLoss()
    # 实例化一个损失函数类
    # torch.nn.MSELoss():创建一个标准，测量输入x和目标y中每个元素之间的平均平方误差（平方L2准则）。
    criterion = nn.MSELoss()
    # 实例化一个优化算法类
    # torch.optim.Adam():实现Adam算法。
    # torch.nn.Module.parameters:返回一个模块参数的迭代器。这通常被传递给一个优化器。
    optimizer_ft = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    # 实例化一个学习率调度类
    # torch.optim.lr_scheduler.StepLR():每隔step_size epochs对每个参数组的学习率进行伽玛衰减。
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
    # 调用函数，返回一系列数据加载类的实例
    # 根据data_name字符串，加载相应的数据：
    if data_name == 'SCUT-FBP':
        print('start loading SCUTFBPDataset...')
        dataloaders = load_reconstruct_scutfbp()
    elif data_name == 'HotOrNot':
        print('start loading HotOrNotDataset...')
        dataloaders = load_reconstruct_hotornot(cv_split_index=cfg['cv_index'])
    elif data_name == 'SCUT-FBP5500':
        print('start loading SCUTFBP5500Dataset...')
        dataloaders = load_reconstruct_scutfbp5500_64()
    elif data_name == 'SCUT-FBP5500CV':
        print('start loading SCUTFBP5500Dataset Cross Validation...')
        dataloaders = load_reconstruct_scutfbp5500_cv(cfg['cv_index'])
    else:
        print('Invalid data name. It can only be SCUT-FBP or HotOrNot...')
        # sys.exit():引发一个SystemExit异常，表示打算退出解释器。
        sys.exit(0)
    # 进入训练阶段
    # 调用函数train_model():
    train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer_ft,
                scheduler=exp_lr_scheduler, num_epochs=cfg['epoch'], inference=False)


def ext_res_dae_feat(img, res_dae):
    """
    extract deep features from Residual Deep AutoEncoder's encoder module
    :param img:
    :param res_dae:
    :return:
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if isinstance(img, str):
        img = Image.open(img)
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = preprocess(img)
    img.unsqueeze_(0)
    img = img.to(device)
    encoder = res_dae.module.encoder if torch.cuda.device_count() > 1 else res_dae.encoder
    feat = encoder(img).to("cpu").detach().numpy().ravel()

    return feat


if __name__ == '__main__':
    # 实例化对象
    res_conv_dae = ResConvDAE()
    # 调用函数main():传入一个实例对象，一个字符串
    main(res_conv_dae, 'SCUT-FBP5500')

    # resConvDAE = ResConvDAE()
    # model_name = resConvDAE.__class__.__name__
    # resConvDAE = resConvDAE.float()
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #
    # resConvDAE = resConvDAE.to(device)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     resConvDAE = nn.DataParallel(resConvDAE)
    # print('[INFO] loading pretrained weights for %s...' % model_name)
    # resConvDAE.load_state_dict(torch.load(os.path.join('./model/%s.pth' % model_name)))
    # resConvDAE.eval()
    #
    # img_dir = '/home/xulu/DataSet/SCUT-FBP/Crop'
    # for img_f in os.listdir(img_dir):
    #     # generate_img_with_dae(os.path.join(img_dir, img_f))
    #     feat = ext_res_dae_feat(os.path.join(img_dir, img_f), resConvDAE)
    #     print(feat)
