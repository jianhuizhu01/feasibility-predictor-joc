# -*- coding:utf-8 -*-
'''
# @author: Jianhui Zhu
# @email: jianhuizhu01@163.com
# @date: 2024/10/07
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import confusion_matrix
from data_process import ItemDataset

def createSampler(dataset):
    dataset_indexs = list(range(len(dataset)))
    return SubsetRandomSampler(dataset_indexs)

def createTestDataloader(class_Config, dataset, sampler):
    """
    使用DataLoader加载数据集，实现批量训练
    :param class_Config: 配置类
    :param dataset: 数据集
    :param sampler: 数据集采样器
    :return: DataLoader
    """
    class_Config.batch_size = 1
    test_loader = DataLoader(dataset, batch_size=class_Config.batch_size, sampler=sampler)
    return test_loader

def inferenceMode(class_Config, model, dataloader, criterion):
    """
    在给定的模型上进行评估，并计算损失和准确率
    :param class_Config: 配置类
    :param model: 神经网络模型
    :param dataloader: 数据加载器
    :param criterion: 损失函数
    :return: 无
    """
    model.eval()
    total_loss = 0 # 总损失
    total_correct = 0 # 正确的样本数
    labels = [] # 所有真实标签
    outputs = [] # 所有预测标签
    with torch.no_grad():
        for batch_index, (input, label) in enumerate(dataloader):
            input, label = input.to(class_Config.device).float(), label.to(class_Config.device).float() # 将数据和标签放到设备上
            output = model(input) # 模型对输入数据进行预测
            output = output.reshape(-1) # 将预测结果展平
            labels.append(label) # 将真实标签添加到列表中
            outputs.append((output >= 0.5).float()) # 将预测结果转换为0或1并添加到列表中
            loss = criterion(output, label) # 计算损失
            total_loss += (input.shape[0] * loss.item()) # 计算测试损失
            total_correct += ((output >= 0.5).float() == label).sum().item() # 计算正确的样本数
    test_loss = total_loss / len(dataloader.dataset) # 计算平均测试损失
    test_accuracy = total_correct / len(dataloader.dataset) # 计算平均测试准确率
    # 打印测试损失和测试准确率，保留三位小数
    print(f"test_loss: {test_loss:.3f}, test_accuracy: {test_accuracy:.3f}")
    # 将Tensor元素转换为Python标量
    labels = [x.item() for x in labels]
    outputs = [x.item() for x in outputs]
    # 打印混淆矩阵
    print(confusion_matrix(labels, outputs))
    true_positive_ratio = confusion_matrix(labels, outputs)[0][0] / (confusion_matrix(labels, outputs)[0][0] + confusion_matrix(labels, outputs)[0][1]) # 计算真正率
    true_negative_ratio = confusion_matrix(labels, outputs)[1][1] / (confusion_matrix(labels, outputs)[1][0] + confusion_matrix(labels, outputs)[1][1]) # 计算真负率
    print(f"true_positive_ratio: {true_positive_ratio:.4f}, true_negative_ratio: {true_negative_ratio:.4f}")

def testModel(class_Config, dataset_file_path):
    """
    测试模型
    :param class_Config: 配置类
    :param dataset_file_path: 数据集文件路径
    :return: 无
    """
    test_dataset = ItemDataset(dataset_file_path)
    sampler = createSampler(test_dataset)
    test_loader = createTestDataloader(class_Config, test_dataset, sampler)
    print("Loading the model...")
    model = torch.load("./model/" + class_Config.save_dir + ".pth")
    model  = model.to(class_Config.device)
    criterion = nn.BCELoss()
    inferenceMode(class_Config, model, test_loader, criterion)


if __name__ == "__main__":
    pass