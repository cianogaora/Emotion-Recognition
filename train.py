import os
import shutil

import numpy

import split_data
import dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNN, Net
import torch.nn.functional as F


def main():
    train_path = 'all_train'

    my_transforms = transforms.Compose([
        ToTensor()
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    train_labels = []
    for name in os.listdir('all_train'):
        train_labels.append(int(name[1]))

    print(train_labels[0])
    train_dataset = dataset.MyDataset(train_path, train_labels, my_transforms, None)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    cnn = CNN()
    net = Net()
    # criterion = F.nll_loss()
    # params = cnn.parameters()
    params = net.parameters()
    optimiser = optim.Adam(params=params, lr=3e-4)
    log_interval = 150

    for epoch in range(10):
        print('epoch: ', epoch)
        # cnn.train()
        net.train()
        running_loss = 0.0
        j = 1
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # labels = torch.tensor(labels)

            optimiser.zero_grad()

            # outputs = cnn(inputs)
            outputs = net(inputs)
            loss = F.nll_loss(outputs, labels)
            # loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()

            optimiser.step()

            running_loss += loss.item()
            if i % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(inputs), len(train_loader.dataset),
                    100. * i / len(train_loader), running_loss/j))

            j+=1
    print('training complete')
    model_count = len(os.listdir('models'))
    # torch.save(cnn.state_dict(), f'models/model{model_count}.pth')
    torch.save(net.state_dict(), f'models/model{model_count}.pth')

if __name__ == '__main__':
    main()
