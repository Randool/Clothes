import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable


paras = {
    'win_path': 'D:\\data\\FashionMNIST\\',
    'linux_path': '/home/ubuntu/Data/FashionMNIST',
    'lr': 0.00004,
    'momentum': 0.4,
    'epochs': 1,
    'batch_size': 16,
    'cuda': True,
}

path = paras['win_path'] if os.name == 'nt' else paras['linux_path']


# Normalize can make training easier
trainset = datasets.FashionMNIST(
    root=path,
    train=True,
    download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)
testset = datasets.FashionMNIST(
    root=path,
    train=True,
    download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=paras['batch_size'])
testloader = torch.utils.data.DataLoader(testset, batch_size=paras['batch_size'])


class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, inputs):
        batch_size, _, x_dim, y_dim = inputs.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        xx_channel = xx_channel.float() / (x_dim - 1) * 2 - 1  # 归一化
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)
        yy_channel = yy_channel.float() / (y_dim - 1) * 2 - 1
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat(
            [inputs, xx_channel.type_as(inputs), yy_channel.type_as(inputs)], dim=1
        )  # 在channel上合并

        if self.with_r:
            # 为什么要-0.5?
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(inputs) - 0.5, 2)
                + torch.pow(yy_channel.type_as(inputs) - 0.5, 2)
            )
            ret = torch.cat([ret, rr], dim=1)  # 在channel上合并

        return ret


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super(CoordConv, self).__init__()
        self.main = nn.Sequential(
            AddCoords(with_r=with_r),
            nn.Conv2d(in_channels + 2 + int(with_r), out_channels, **kwargs),
        )

    def forward(self, inputs):
        return self.main(inputs)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            CoordConv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            CoordConv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.main(x) + x
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block_type):
        super(ResNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            block_type(8, 8), block_type(8, 8),
            block_type(8, 8), block_type(8, 8),
            block_type(8, 8), block_type(8, 8),
        )
        self.fcs = nn.Sequential(
            nn.Linear(8 * 26 * 26, 128),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)  # -> 8 * 28 * 28
        return self.fcs(x)

# resnet = ResNet(ResidualBlock)


# Training
def train_model():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=paras['lr'])
    # optimizer = optim.SGD(resnet.parameters(), lr=paras['lr'], momentum=paras['momentum'])

    start = time.time()
    losses = []
    for epoch in range(paras['epochs']):
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = Variable(inputs), Variable(labels)
            if paras['cuda']:
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                losses.append(loss.item())
                print(f'Epoch {epoch+1} | {i+1:#3d} batch, Loss: {loss.item():#.4f}')
    print(f'Finished! Cost {((time.time()-start)/60):#.2f} min')


# import matplotlib.pyplot as plt
# plt.plot(losses)

# Testing and Saving the Trained Model
def test_model():
    print('Start testing...')
    start = time.time()

    total, correct = 0, 0
    for inputs, labels in testloader:
        if paras['cuda']:
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = resnet(Variable(inputs))
        _, predict = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predict == labels).sum()

    end = time.time()
    print(f'Cost {((end-start)/60):#.2f} min.\nAccuracy: {(100*correct/total):#.2f} %')

    if input('Save? (Y/N)') in ['Y', 'y']:
        torch.save(resnet.state_dict(), 'resnet_coord.pkl')
        print('Saved!')
    return


if __name__ == '__main__':
    resnet = ResNet(ResidualBlock)
    if paras['cuda']:
        resnet.cuda()

    while True:
        action = input("1: load\n2: train\n3: test\n4: exit\n> ")

        if action == "1":
            print("Loading parameters...")
            try:
                resnet.load_state_dict(torch.load('resnet_coord.pkl'))
                print("Successful!")
            except:
                print("Can't load parameters!")

        elif action == "2":
            train_model()

        elif action == "3":
            test_model()

        elif action == "4":
            print("Exit!")
            break
