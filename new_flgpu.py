import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold  # 使用KFold确保平均分配
from random import choice
import json
from itertools import islice, cycle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from itertools import cycle
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)

        # Define a container to hold intermediate activations
        self.fc1_outputs = []
        self.fc2_outputs = []
        self.fc1_relu_outputs = []
        self.fc2_relu_outputs = []

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = self.fc1(x)
        self.fc1_outputs.append(x.clone().detach())  # Save the output of fc1
        x = F.relu(x)
        self.fc1_relu_outputs.append(x.clone())  # Save the output after ReLU
        x = self.fc2(x)
        self.fc2_outputs.append(x.clone())  # Save the output of fc2
        x = F.relu(x)
        self.fc2_relu_outputs.append(x.clone())  # Save the output after ReLU
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def clear_activations(self):
        self.fc1_outputs.clear()
        self.fc2_outputs.clear()
        self.fc1_relu_outputs.clear()
        self.fc2_relu_outputs.clear()


# 数据加载器，现在返回分割后的多个训练数据集
def federated_datasets(num_clients=10, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    full_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 获取数据集的总大小
    num_items = len(full_trainset)
    # 计算每个客户端的数据量
    items_per_client = num_items // num_clients
    # 生成随机排列的索引
    indices = torch.randperm(num_items)

    client_trainloaders = []
    for i in range(num_clients):
        # 计算每个客户端的索引范围
        start_idx = i * items_per_client
        end_idx = start_idx + items_per_client if i != num_clients - 1 else num_items
        # 选择索引
        client_indices = indices[start_idx:end_idx]
        # 创建子集
        client_dataset = Subset(full_trainset, client_indices.tolist())
        trainloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        client_trainloaders.append(trainloader)


    testloader = DataLoader(full_testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return client_trainloaders, testloader

def adjust_keys_from_dataparallel(state_dict):
    """从DataParallel状态字典中移除'module.'前缀"""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith('module.') else key  # 从第7个字符开始切片，移除'module.'
        new_state_dict[new_key] = value
    return new_state_dict

def remove_module_prefix(state_dict):
    """从状态字典的键中移除'module.'前缀"""
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict



def average_models(global_model, client_models):
    with torch.no_grad():
        # 获取全局模型的设备
        if isinstance(global_model, torch.nn.DataParallel):
            global_device = next(global_model.module.parameters()).device
        else:
            global_device = next(global_model.parameters()).device

        # 获取全局模型的状态字典
        if isinstance(global_model, torch.nn.DataParallel):
            global_dict = global_model.module.state_dict()
        else:
            global_dict = global_model.state_dict()

        # 创建新的状态字典以存储平均后的权重
        new_state_dict = {}

        for key in global_dict:
            # 提取出对应的前缀处理后的key
            if key.startswith('module.'):
                stripped_key = key[len('module.'):]
            else:
                stripped_key = key

            # 确保所有张量都移到了全局模型的设备上
            tensor_list = [client_model.state_dict()[key].to(global_device) if key in client_model.state_dict() else client_model.state_dict()['module.' + stripped_key].to(global_device) for client_model in client_models]
            averaged_tensor = torch.stack(tensor_list).mean(dim=0)

            # 保存平均后的权重到新字典中
            new_state_dict[key] = averaged_tensor

        # 如果全局模型是DataParallel，需要加载平均后的状态字典
        if isinstance(global_model, torch.nn.DataParallel):
            global_model.module.load_state_dict(new_state_dict)
        else:
            global_model.load_state_dict(new_state_dict)


def federated_train_multiGPU(num_epochs, client_trainloaders, test_loader, global_model):
    num_clients = 10
    device_ids = list(range(torch.cuda.device_count()))  # 可用GPU列表
    model_cls = type(global_model.module) if isinstance(global_model, nn.DataParallel) else type(global_model)

    # 在所有客户端上复制全局模型
    client_models = [model_cls().to(torch.device(f'cuda:{i % len(device_ids)}')) for i in range(num_clients)]
    optimizers = [optim.SGD(model.parameters(), lr=conf["lr"]) for model in client_models]

    for epoch in range(num_epochs):
        for client_idx, train_loader in enumerate(client_trainloaders):
            device = torch.device(f'cuda:{client_idx % len(device_ids)}')
            print(f"Training on client {client_idx + 1} on GPU {client_idx % len(device_ids)}")

            model = client_models[client_idx]
            optimizer = optimizers[client_idx]

            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

            model.clear_activations()  # 清理激活数据

        average_models(global_model, client_models)  # 聚合模型参数

        global_model.to('cuda:0')  # 确保全局模型在cuda:0上
        for model in client_models:
            if isinstance(global_model, nn.DataParallel):
                adjusted_state_dict = adjust_keys_from_dataparallel(global_model.state_dict())
                model.load_state_dict(adjusted_state_dict)
            else:
                model.load_state_dict(global_model.state_dict())

        torch.cuda.empty_cache()  # 清理GPU缓存
        print("Testing aggregated global model")
        test(global_model, 'cuda:0', test_loader)  # 测试全局模型性能

        global_model.clear_activations()  # 清理全局模型激活数据



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
    conf_path = './utils/conf.json'

    # 读取配置文件
    with open(conf_path, 'r') as f:
        conf = json.load(f)

    global_model = nn.DataParallel(Net().to('cuda:0'))
    num_clients = 10
    num_epochs = 100


    client_trainloaders, test_loader = federated_datasets(num_clients, conf["batch_size"])

    federated_train_multiGPU(conf["global_epochs"], client_trainloaders, test_loader, global_model)

