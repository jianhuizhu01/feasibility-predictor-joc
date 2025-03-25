# -*- coding:utf-8 -*-
'''
# @author: Jianhui Zhu
# @email: jianhuizhu01@163.com
# @date: 2024/10/07
'''

import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import pickle as pkl

# 自定义Dataset类
class ItemDataset(Dataset):
    def __init__(self, dataset_file_path):
        super(ItemDataset, self).__init__()
        # open the .pkl file, the dataset 
        with open(dataset_file_path, "rb") as fp:
            dataset = pkl.load(fp)
        self.dataset = {k: v for k, v in dataset.items() if v['label'] == 1 or v['label'] == 0}
    
    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.dataset.keys())

    def __getitem__(self, i):
        """
        返回第i个样本的特征和标签
        """
        sample_idx = list(self.dataset.keys())[i]
        sample = self.dataset[sample_idx]
        items, label = sample['items'], sample['label']
        max_width, max_height = sample['bin']
        manual_features = np.asarray(extractManualFeature(items, max_width, max_height))
        return manual_features, label

def createTrainDataloader(class_Config, dataset):
    """
    为数据集准备DataLoader
    使用DataLoader加载数据集，实现批量训练
    :param class_Config: 配置类
    :param dataset: 数据集
    :return: DataLoader
    """
    # 按照80%训练，20%验证划分数据集
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=class_Config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=class_Config.batch_size, shuffle=False)
    return train_loader, validation_loader

def extractManualFeature(items, bin_width, bin_height):
    """
    解析数据集
    The method is to extract features from a set of items
    Parameters:
    ----------
    items (a list of np.arrays)
    bin_width (int): the width of a bin
    bin_height (int): the height of a bin
    Returns (a list of metrics)
    -------
    notes:
    we extract five types of features:
    1) the ratio between width and height, and four statistical metrics (mean, min, max, std) as the features
    2) the ratio between width and the bin width, four statistical metrics (mean, min, max, std) as the features
    3) the ratio between height and the bin height, four statistical metrics (mean, min, max, std) as the features
    4) the ratio between area of a item and the bin capacity, four statistical metrics (mean, min, max, std) as the features
    5) the ratio between total area of the items and the bin capacity, a single metric
    """
    MAX_W_H_RATIO = 18 # 定义了一个常量，用于归一化
    capacity = bin_width * bin_height
    w_h_ratios = np.asarray(list(map(lambda x: x[0] / x[1], items))) / MAX_W_H_RATIO
    w_bin_ratios = np.asarray(list(map(lambda x: x[0] / bin_width, items)))
    h_bin_ratios = np.asarray(list(map(lambda x: x[1] / bin_height, items)))
    area_capacity_ratios = np.asarray(list(map(lambda x: (x[0] * x[1]) / capacity, items)))
    total_area = np.asarray(list(map(lambda x: x[0] * x[1], items))).sum()
    w_h_features = [w_h_ratios.mean(), w_h_ratios.min(), w_h_ratios.max(), w_h_ratios.std()]
    w_bin_features = [w_bin_ratios.mean(), w_bin_ratios.min(), w_bin_ratios.max(), w_bin_ratios.std()]
    h_bin_features = [h_bin_ratios.mean(), h_bin_ratios.min(), h_bin_ratios.max(), h_bin_ratios.std()]
    area_capacity_features = [area_capacity_ratios.mean(), area_capacity_ratios.min(), area_capacity_ratios.max(), area_capacity_ratios.std()]
    total_area_capacity_features = [total_area / capacity]
    extracted_features = [w_h_features, w_bin_features, h_bin_features, area_capacity_features, total_area_capacity_features]
    result = []
    for x in extracted_features:
        result.extend(x)
    return result

if __name__ == "__main__":
    pass
