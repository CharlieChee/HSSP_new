import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import json


# federated_loader
# def federated_loader(batch_size):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#
#     # 加载完整的MNIST训练数据集
#     full_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#
#     # 所有图像的索引
#     indices = list(range(len(full_trainset)))
#
#     # 为了确保第一个batch全部是label为8的图像
#     def batch_sampler():
#         # 其他图像的索引
#         other_indices = [idx for idx in indices if full_trainset.targets[idx] != 8]
#         # 使用torch.randperm来打乱其他索引
#         shuffled_indices = torch.randperm(len(other_indices)).tolist()
#         other_indices = [other_indices[i] for i in shuffled_indices]
#
#         # 索引前面放置label为8的图像
#         label_8_indices = [idx for idx in indices if full_trainset.targets[idx] == 8]
#         selected_label_8_indices = label_8_indices[:batch_size]
#
#         # 先产生只有label为8的图像的第一个batch
#         yield selected_label_8_indices
#
#         # 然后产生其他打乱的batch
#         other_batch_start = 0
#         while other_batch_start < len(other_indices):
#             yield other_indices[other_batch_start:other_batch_start + batch_size]
#             other_batch_start += batch_size
#
#     # 创建联合的DataLoader
#     train_loader = torch.utils.data.DataLoader(
#         full_trainset,
#         batch_sampler=batch_sampler(),
#         num_workers=2
#     )
#
#     # 加载测试数据集
#     testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
#
#     return train_loader, testloader

def federated_loader(batch_size):
    # MNIST数据集只有一个颜色通道（灰度），所以均值和标准差是单个值，不是三元组
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载和加载MNIST训练数据集
    trainset = datasets.MNIST(root='./data', train=True,
                              download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    # 下载和加载MNIST测试数据集
    testset = datasets.MNIST(root='./data', train=False,
                             download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader


# model
class Net(nn.Module):
    def __init__(self, conf):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, int(conf["first_layer_neurons"]))  # 调整输入尺寸
        self.fc2 = nn.Linear(int(conf["first_layer_neurons"]), 100)
        self.fc3 = nn.Linear(100, 10)

        # Define a container to hold intermediate activations
        self.fc1_outputs = []
        self.fc2_outputs = []
        self.fc1_relu_outputs = []
        self.fc2_relu_outputs = []

    def forward(self, x):
        x = x.view(-1, 28*28)
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
            np.save(f'./expdata/mnist_ini/images_epoch_{epoch}_batch_{save_batch_idx}.npy', data.cpu().numpy())
            np.save(f'./expdata/mnist_ini/gradients_fc1_relu_binary_epoch_{epoch}_batch_{batch_idx}.npy',
                    ((output_activation > 0).int()).cpu().numpy())

        loss.backward()
        optimizer.step()

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader), loss.item()))

        model.clear_activations()


# test
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

    log_interval = 1000  # For every 200 batches of training, the training progress information is printed.
    save_batch_idx = conf["save_batch_idx"]  # Save the first batch of data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net(conf).to(device)
    # output_activation = model.fc1_relu_outputs[-1]
    # np.save(f'./expdata/initial.npy',
    #         ((output_activation > 0).int()).cpu().numpy())

    optimizer = optim.SGD(model.parameters(), lr=conf["lr"])

    train_loader, test_loader = federated_loader(conf["batch_size"])


    # 获取并保存初始激活
    data, _ = next(iter(train_loader))
    data = data.to(device)
    model(data)  # 执行前向传播
    initial_output_activation = model.fc1_relu_outputs[-1].detach().cpu().numpy()
    np.save(f'./expdata/mnist_ini/initial_activation.npy', (initial_output_activation > 0).astype(int))

    model.clear_activations()  # 清理以准备训练
    for epoch in range(conf["global_epochs"]):
        train(model, device, train_loader, optimizer, epoch, log_interval, save_batch_idx)
        test(model, device,train_loader, test_loader)


if __name__ == "__main__":
    run()

