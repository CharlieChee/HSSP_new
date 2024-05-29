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
from pathlib import Path
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


def create_full_train_loader(batch_size=32):
    # 定义预处理和归一化的转换操作
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载 CIFAR-10 训练集
    full_train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # 创建 DataLoader
    full_train_loader = DataLoader(full_train_set, batch_size=batch_size, shuffle=True)

    return full_train_loader

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
        trainloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        client_trainloaders.append(trainloader)


    testloader = DataLoader(full_testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return client_trainloaders, testloader


# 模型聚合函数
def average_models(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))],
                                     0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


# 训练函数，现在需要处理多个加载器
def federated_train_old(num_epochs, client_trainloaders, test_loader, global_model, device,full,save_batch_idx):
    # 在所有客户端上复制全局模型
    client_models = [type(global_model)().to(device) for _ in client_trainloaders]
    optimizers = [optim.SGD(model.parameters(), lr=conf["lr"]) for model in client_models]

    log_file = Path('./expdata/acc/new_fl_shuffle_false.log')
    log_file.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):

        for client_idx, train_loader in enumerate(client_trainloaders):
            print(f"Training on client {client_idx + 1}")
            for batch_idx,(data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                client_models[client_idx].train()
                optimizers[client_idx].zero_grad()
                output = client_models[client_idx](data)
                loss = F.nll_loss(output, target)
                loss.backward()
                output_activation = client_models[client_idx].fc1_relu_outputs[-1]

                # Save gradients in specified batch
                if batch_idx == save_batch_idx and client_idx == 0:
                    np.save(f'./expdata/truefl_shuffle_false/images_epoch_{epoch}_batch_{save_batch_idx}.npy',
                            data.cpu().numpy())
                    np.save(f'./expdata/truefl_shuffle_false/gradients_fc1_relu_binary_epoch_{epoch}_batch_{batch_idx}.npy',
                            ((output_activation > 0).int()).cpu().numpy())

                optimizers[client_idx].step()

                # Calculate and print training set accuracy


            client_models[client_idx].clear_activations()

            test(client_models[client_idx], device, test_loader)

        average_models(global_model, client_models)

        train_accuracy = test(global_model, device, full, "Training")
        test_accuracy = test(global_model, device, test_loader, "Test")

        # Log accuracies
        with log_file.open('a') as f:
            f.write(f'Epoch {epoch}, Training Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}\n')

        print("Testing aggregated global model")
        test(global_model, device, test_loader)

        for model in client_models:
            model.load_state_dict(global_model.state_dict())

        torch.cuda.empty_cache()  # 清理 GPU 缓存
        print("Testing aggregated global model")
        test(global_model, device, test_loader)

        global_model.clear_activations()


def federated_train(num_epochs, client_trainloaders, test_loader, global_model, device):
    # 在所有客户端上复制全局模型
    client_models = [type(global_model)().to(device) for _ in client_trainloaders]
    optimizers = [optim.SGD(model.parameters(), lr=conf["lr"]) for model in client_models]

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        for client_idx, train_loader in enumerate(client_trainloaders):
            # 计算当前epoch应该使用的批次索引
            batch_index = (epoch - 1) % len(train_loader)
            # 从train_loader获取特定的批次
            batch_iterator = iter(train_loader)
            chosen_batch = next(islice(batch_iterator, batch_index, None))
            data, target = chosen_batch
            data, target = data.to(device), target.to(device)

            print(f"Training on client {client_idx + 1} with batch index {batch_index}")
            client_models[client_idx].train()
            optimizers[client_idx].zero_grad()
            output = client_models[client_idx](data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizers[client_idx].step()

            # 清理激活以节省内存
            client_models[client_idx].clear_activations()

            # 选用这一批次训练数据后立即进行测试（可选）
            test(client_models[client_idx], device, test_loader)

        # 所有客户端训练完成后，聚合更新到全局模型
        average_models(global_model, client_models)

        # 将更新后的全局模型状态复制回每个客户端模型
        for model in client_models:
            model.load_state_dict(global_model.state_dict())

        del client_models
        torch.cuda.empty_cache()  # 清理 GPU 缓存
        print("Testing aggregated global model")
        test(global_model, device, test_loader)

        global_model.clear_activations()

def test(model, device, data_loader, set_name="Test"):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(data_loader.dataset)
    print(f'{set_name} set: Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


import subprocess
import os
import torch

def get_idle_gpu():
    # 调用 nvidia-smi 命令获取 GPU 状态
    smi_output = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                                 '--format=csv,nounits,noheader'], capture_output=True, text=True)
    # 解析输出，每一行包括 GPU 编号，空闲内存和 GPU 使用率
    gpu_stats = [line.split(', ') for line in smi_output.stdout.strip().split('\n')]
    # 按空闲内存大小和 GPU 使用率排序（内存降序，使用率升序）
    gpu_stats.sort(key=lambda x: (-int(x[1]), int(x[2])))
    # 返回最空闲的 GPU 编号
    return int(gpu_stats[0][0])
# 测试函数保持不变

# 主程序
if __name__ == "__main__":
    conf_path = './utils/conf.json'

    # 读取配置文件
    with open(conf_path, 'r') as f:
        conf = json.load(f)
    idle_gpu = get_idle_gpu()
    print("Using GPU:", idle_gpu)

    # 设置 PyTorch 使用最空闲的 GPU
    device = torch.device(f"cuda:{idle_gpu}" if torch.cuda.is_available() else "cpu")

    global_model = Net().to(device)


    num_clients = 10
    num_epochs = 100



    client_trainloaders, test_loader = federated_datasets(num_clients, conf["batch_size"])

    full = create_full_train_loader(conf["batch_size"])

    federated_train_old(conf["global_epochs"], client_trainloaders, test_loader, global_model, device,full,save_batch_idx=0)

