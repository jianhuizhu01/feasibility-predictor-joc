# -*- coding:utf-8 -*-
'''
# @author: Jianhui Zhu
# @email: jianhuizhu01@163.com
# @date: 2024/10/07
'''

import torch
import pandas as pd
import matplotlib.pyplot as plt

def validateModel(class_Config, dataloader, model, criterion):
    """
    在给定的模型上进行评估，并计算损失和准确率
    :param class_Config: 配置类
    :param model: 神经网络模型
    :param dataloader: 数据加载器
    :param criterion: 损失函数
    :return: 平均损失，平均准确率、所有真实标签和所有预测标签
    """
    model.eval()
    total_loss = 0 # 总损失
    total_correct = 0 # 正确的样本数
    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(class_Config.device).float(), labels.to(class_Config.device).float() # 将数据和标签放到设备上
            outputs = model(inputs) # 模型对输入数据进行预测
            outputs = outputs.reshape(-1) # 将预测结果展平
            loss = criterion(outputs, labels) # 计算损失
            total_loss += (inputs.shape[0] * loss.item()) # 计算损失
            total_correct += ((outputs >= 0.5).float() == labels).sum().item() # 计算正确的样本数
    validation_loss = total_loss / len(dataloader.dataset) # 计算平均损失
    validation_accuracy = total_correct / len(dataloader.dataset) # 计算平均准确率
    return validation_loss, validation_accuracy

def trainModel(class_Config, train_loader, validation_loader, model, optimizer, criterion):
    best_validation_loss = float('inf') # 初始化最好的验证损失
    patience_counter = 0 # 初始化耐心计数器
    print("Start training...")
    train_loss_list = [] # 训练损失列表
    train_accuracy_list = [] # 训练正确率列表
    validation_loss_list = [] # 验证损失列表 
    validation_accuracy_list = [] # 验证正确率列表
    for epoch in range(class_Config.number_of_epoches):
        # 训练模式
        model.train()
        total_loss = 0 # 总损失
        total_correct = 0 # 正确率
        for batch_index, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(class_Config.device).float(), labels.to(class_Config.device).float() # 将数据和标签放到设备上
            optimizer.zero_grad() # 梯度清零
            outputs = model(inputs) # 前向传播
            outputs = outputs.reshape(-1) # 将预测结果展平
            loss = criterion(outputs, labels) # 计算损失
            loss.backward() # 反向传播，计算当前损失的梯度
            optimizer.step() # 使用计算的梯度更新模型参数
            total_loss += (inputs.shape[0] * loss.item()) # 累加损失
            total_correct += ((outputs >= 0.5).float() == labels).sum().item() # 累加正确的个数
        train_loss = total_loss / len(train_loader.dataset) # 计算平均损失
        train_loss_list.append(train_loss)
        train_accuracy = total_correct / len(train_loader.dataset) # 计算正确率
        train_accuracy_list.append(train_accuracy)
        # 在验证集上评估模型
        validation_loss, validation_accuracy = validateModel(class_Config, validation_loader, model, criterion)
        validation_loss_list.append(validation_loss)
        validation_accuracy_list.append(validation_accuracy)
        print("Epoch: {}, train_loss: {:.6f}, train_accuracy: {:.6f}, validation_loss: {:.6f}, validation_accuracy: {:.6f}".format(epoch+1, train_loss, train_accuracy, validation_loss, validation_accuracy))
        # 提前停止机制
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            patience_counter = 0  # 重置耐心计数
        else:
            patience_counter += 1
        if patience_counter >= class_Config.early_stopping_epoches:
            print(f'Early stopping at epoch {epoch+1}')
            break
    # 保存模型
    torch.save(model, "./model/" + class_Config.save_dir + ".pth")
    # 将输出的结果保存到csv文件中
    results_df = pd.DataFrame({'Epoch': list(range(1, class_Config.number_of_epoches+1)), 'train_loss': train_loss_list, 'train_accuracy': train_accuracy_list, 'validation_loss': validation_loss_list, 'validation_accuracy': validation_accuracy_list})
    results_df.to_csv("./data/output/" + class_Config.save_dir + ".csv", index=False)

def plotProcess(class_Config):
    # 读取csv文件中的数据
    df = pd.read_csv("./data/output/" + class_Config.save_dir + ".csv")
    train_loss_list = df['train_loss'].tolist()
    train_accuracy_list = df['train_accuracy'].tolist()
    validation_loss_list = df['validation_loss'].tolist()
    validation_accuracy_list = df['validation_accuracy'].tolist()
    # 绘制正确率变化曲线
    plt.figure()
    plt.plot(train_accuracy_list, label=f"{class_Config.save_dir} train_accuracy")
    plt.plot(validation_accuracy_list, label=f"{class_Config.save_dir} validation_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xlim(0, class_Config.number_of_epoches) 
    plt.legend()
    plt.show()
    # 绘制训练损失变化曲线
    plt.figure()
    plt.plot(train_loss_list, label=f"{class_Config.save_dir} train_loss")
    plt.plot(validation_loss_list, label=f"{class_Config.save_dir} validation_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(0, class_Config.number_of_epoches) 
    plt.legend()
    plt.show()

if __name__ == "__main__":
    pass