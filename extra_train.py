#!/usr/bin/env python3
import os
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

def main(modelChoice):
    train_path = 'all_train'

    my_transforms = transforms.Compose([
        ToTensor()
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

    train_labels = []
    for im_name in os.listdir(train_path):
        train_labels.append(int(im_name[1]))

    train_dataset = dataset.MyDataset(train_path, train_labels, my_transforms, None)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)

    models = os.listdir('models')
    model_path = models[-1]
    cnn = CNN()
    net = Net()
    # cnn.load_state_dict(torch.load(f'models/model{modelChoice}.pth'))
    net.load_state_dict(torch.load(f'models/model{modelChoice}.pth'))

    print(f'training model {model_path}')
    #criterion = nn.CrossEntropyLoss()
    # params = cnn.parameters()
    params = net.parameters()
    optimiser = optim.Adam(params=params, lr=3e-4)
    log_interval = 200

    for epoch in range(20):
        print('epoch: ', epoch + 1)
        # cnn.train()
        net.train()
        running_loss = 0.0
        j = 1
        for i, data in enumerate(train_loader):
            inputs, labels = data

            optimiser.zero_grad()

            # outputs = cnn(inputs)
            outputs = net(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            if i % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, i * len(inputs), len(train_loader.dataset),
                    100. * i / len(train_loader), running_loss/j))
            j+=1
    print('training complete')
    # torch.save(cnn.state_dict(), f'models/model{modelChoice}.pth')
    torch.save(net.state_dict(), f'models/model{modelChoice}.pth')



if __name__ == '__main__':
    modelChoice = str(input("Enter model number: "))
    main(modelChoice)
