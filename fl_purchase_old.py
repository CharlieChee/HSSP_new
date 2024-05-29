import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import *
from torch.utils.data import TensorDataset, DataLoader

# 数据加载器
def federated_loader(batch_size):

    data = np.load("/Users/jichanglong/Desktop/hssp_All/data/dataset_purchase.npy")
    all_data = data[:60000, 1:]  # 数据
    all_labels = data[:60000, 0]  # 标签
    all_labels -= 1  # 调整标签



    all_data_tensor = torch.tensor(all_data, dtype=torch.float)
    all_labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    train_dataset = TensorDataset(all_data_tensor[:50000], all_labels_tensor[:50000])
    test_dataset = TensorDataset(all_data_tensor[50000:], all_labels_tensor[50000:])  # 示例中使用了相同的数据集作为训练集和测试集


    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return trainloader, testloader


# 训练
def train(model, device, train_loader, optimizer, epoch, log_interval, save_batch_idx):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        # 获取fc1的输出和fc1_relu的输出
        fc1_output = model.fc1_outputs[-1]
        fc1_relu_output = model.fc1_relu_outputs[-1]
        output_activation = model.fc1_relu_outputs[-1]
        # 计算梯度
        # gradients_loss_to_fc1_relu_output = torch.autograd.grad(loss, fc1_relu_output, retain_graph=True, allow_unused=True)

        # grad_fc1_output_weight = torch.autograd.grad(loss, model.fc1.weight, retain_graph=True)[0]
        # grad_fc1_output_bias = torch.autograd.grad(loss, model.fc1.bias, retain_graph=True)[0]

        # 在指定的batch保存梯度
        if batch_idx == save_batch_idx and epoch % 2 == 0:
            np.save(f'purchase_batch20/images_epoch_{epoch}_batch_{save_batch_idx}.npy', data.cpu().numpy())
            # np.save(f'gradients_fc1_output_weight_epoch_{epoch}_batch_{batch_idx}.npy', grad_fc1_output_weight.cpu().numpy())
            # np.save(f'gradients_fc1_output_bias_epoch_{epoch}_batch_{batch_idx}.npy', grad_fc1_output_bias.cpu().numpy())
            # np.save(f'gradients_fc1_relu_output_weight_epoch_{epoch}_batch_{batch_idx}.npy', grad_fc1_relu_output_weight.cpu().numpy())
            # np.save(f'gradients_fc1_relu_output_bias_epoch_{epoch}_batch_{batch_idx}.npy', grad_fc1_relu_output_bias.cpu().numpy())
            np.save(f'purchase_batch20/gradients_fc1_relu_binary_epoch_{epoch}_batch_{batch_idx}.npy',
                    ((output_activation > 0).int()).cpu().numpy())

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# 测试
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net_purchase().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_loader, test_loader = federated_loader(10)

    log_interval = 1000  # 每训练200批次，打印一次训练的进度信息
    save_batch_idx = 10  # 保存第一个批次的数据
    epochs = 2
    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, epoch, log_interval, save_batch_idx)
        test(model, device, test_loader)







