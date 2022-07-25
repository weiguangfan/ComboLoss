from collections import OrderedDict  # python的collections包
# 生成有序字典
cfg = OrderedDict()
# 使用gpu的状态
cfg['use_gpu'] = True
# 数据加载的绝对路径
cfg['scut_fbp_dir'] = '/home/xulu/DataSet/Face/SCUT-FBP/Crop'
cfg['hotornot_dir'] = '/home/xulu/DataSet/Face/eccv2010_beauty_data/hotornot_face'
cfg['cv_index'] = 1
cfg['scutfbp5500_base'] = '/home/xulu/DataSet/Face/SCUT-FBP5500'
# 批处理大小
cfg['batch_size'] = 64
# 遍历完一遍所有样本称为一个epoch
cfg['epoch'] = 200
# 随机数种子，固定的话，切割数据比例保持不变
cfg['random_seed'] = 40
