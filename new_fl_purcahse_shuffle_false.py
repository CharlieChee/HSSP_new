import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import json

from torch.utils.data import TensorDataset, DataLoader
# federated_loader
def federated_loader(batch_size):

    data = np.load("./data/dataset_purchase.npy")
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



# train
def train(model, device, train_loader, optimizer, epoch, log_interval, save_batch_idx):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        # Get the output of fc1 and the output of fc1_relu
        output_activation = model.fc1_relu_outputs[-1]

        # Save gradients in specified batch
        if batch_idx == save_batch_idx:
            np.save(f'./expdata/purchase/images_epoch_{epoch}_batch_{save_batch_idx}.npy', data.cpu().numpy())
            np.save(f'./expdata/purchase/gradients_fc1_relu_binary_epoch_{epoch}_batch_{batch_idx}.npy',
                    ((output_activation > 0).int()).cpu().numpy())

        loss.backward()
        optimizer.step()

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        model.clear_activations()

def test(model, device, train_loader, test_loader):
    model.eval()
    test_loss = 0
    correct_test = 0
    correct_train = 0

    # 计算测试集上的准确度
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct_test += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # 计算训练集上的准确度
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct_train += pred.eq(target.view_as(pred)).sum().item()

    # 打印结果
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct_test, len(test_loader.dataset),
        100. * correct_test / len(test_loader.dataset)))

    print('Train set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct_train, len(train_loader.dataset),
        100. * correct_train / len(train_loader.dataset)))

def run():
    conf_path = './utils/conf.json'

    # 读取配置文件
    with open(conf_path, 'r') as f:
        conf = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)
    # output_activation = model.fc1_relu_outputs[-1]
    # np.save(f'./expdata/initial.npy',
    #         ((output_activation > 0).int()).cpu().numpy())

    optimizer = optim.SGD(model.parameters(), lr=conf["lr"])

    train_loader, test_loader = federated_loader(conf["batch_size"])
    log_interval = 1000  # For every 200 batches of training, the training progress information is printed.
    save_batch_idx = conf["save_batch_idx"]  # Save the first batch of data

    # 获取并保存初始激活
    data, _ = next(iter(train_loader))
    data = data.to(device)
    model(data)  # 执行前向传播
    initial_output_activation = model.fc1_relu_outputs[-1].detach().cpu().numpy()
    np.save(f'./expdata/purchase/initial_activation.npy', (initial_output_activation > 0).astype(int))

    model.clear_activations()  # 清理以准备训练
    for epoch in range(conf["global_epochs"]):
        train(model, device, train_loader, optimizer, epoch, log_interval, save_batch_idx)
        test(model, device, train_loader, test_loader)


if __name__ == "__main__":
    run()

