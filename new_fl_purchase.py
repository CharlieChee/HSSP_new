import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
# federated_loader

def create_full_train_loader(batch_size=32):
    data = np.load("./data/dataset_purchase.npy")
    all_data = data[:60000, 1:]  # 数据
    all_labels = data[:60000, 0]  # 标签
    all_labels -= 1  # 调整标签以匹配索引

    all_data_tensor = torch.tensor(all_data, dtype=torch.float)
    all_labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    # 将数据分为训练集和测试集
    train_dataset = TensorDataset(all_data_tensor[:50000], all_labels_tensor[:50000])
    test_dataset = TensorDataset(all_data_tensor[50000:], all_labels_tensor[50000:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader

def federated_datasets(num_clients=10, batch_size=64):
    data = np.load("./data/dataset_purchase.npy")
    all_data = data[:60000, 1:]  # 数据
    all_labels = data[:60000, 0]  # 标签
    all_labels -= 1  # 调整标签以匹配索引

    all_data_tensor = torch.tensor(all_data, dtype=torch.float)
    all_labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    # 将数据分为训练集和测试集
    train_dataset = TensorDataset(all_data_tensor[:50000], all_labels_tensor[:50000])
    test_dataset = TensorDataset(all_data_tensor[50000:], all_labels_tensor[50000:])

    num_train_items = len(train_dataset)
    items_per_client = num_train_items // num_clients
    indices = torch.randperm(num_train_items)

    client_datasets = []
    for i in range(num_clients):
        start_idx = i * items_per_client
        end_idx = start_idx + items_per_client if i != num_clients - 1 else num_train_items
        client_indices = indices[start_idx:end_idx]
        client_dataset = Subset(train_dataset, client_indices.tolist())
        client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
        client_datasets.append(client_loader)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return client_datasets, test_loader
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # CIFAR-10图像尺寸为32x32，通道数为3
        self.fc1 = nn.Linear(600, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 100)  # purchase 一共100个类别

        self.fc1_outputs = []
        # 定义容器以保存ReLU激活后的输出
        self.fc1_relu_outputs = []
        self.fc2_relu_outputs = []
        self.fc3_relu_outputs = []

    def forward(self, x):
        x = x.view(-1, 600)  # 展平图像
        x = self.fc1(x)
        self.fc1_outputs.append(x.clone().detach())  # 保存fc1的输出
        x_relu = F.relu(x)
        self.fc1_relu_outputs.append(x_relu.clone().detach())  # 保存ReLU后的fc1输出
        x = self.fc2(x_relu)
        x_relu = F.relu(x)
        #self.fc2_relu_outputs.append(x_relu.clone().detach())  # 保存ReLU后的fc2输出
        x = self.fc3(x_relu)
        x_relu = F.relu(x)
        #self.fc3_relu_outputs.append(x_relu.clone().detach())  # 保存ReLU后的fc3输出
        x = self.fc4(x_relu)
        return F.log_softmax(x, dim=1)

    def clear_activations(self):
        self.fc1_outputs.clear()

        self.fc1_relu_outputs.clear()

def average_models(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))],
                                     0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


# 训练函数，现在需要处理多个加载器
def federated_train_old(num_epochs, client_trainloaders, test_loader, global_model, device,save_batch_idx):
    # 在所有客户端上复制全局模型
    client_models = [type(global_model)().to(device) for _ in client_trainloaders]
    optimizers = [optim.SGD(model.parameters(), lr=conf["lr"]) for model in client_models]

    log_file = Path('./expdata/acc/new_fl_purchase_true.log')
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
                    np.save(f'./expdata/truefl_purchase_true/images_epoch_{epoch}_batch_{save_batch_idx}.npy',
                            data.cpu().numpy())
                    np.save(f'./expdata/truefl_purchase_true/gradients_fc1_relu_binary_epoch_{epoch}_batch_{batch_idx}.npy',
                            ((output_activation > 0).int()).cpu().numpy())

                optimizers[client_idx].step()

                # Calculate and print training set accuracy


            client_models[client_idx].clear_activations()

            test(client_models[client_idx], device, test_loader)

        average_models(global_model, client_models)


        print("Testing aggregated global model")
        test(global_model, device, test_loader)

        for model in client_models:
            model.load_state_dict(global_model.state_dict())

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
    batchsize = 10
    num_epochs = 100


    client_trainloaders, test_loader = federated_datasets(num_clients, batchsize )
    federated_train_old(conf["global_epochs"], client_trainloaders, test_loader, global_model, device,save_batch_idx=0)

