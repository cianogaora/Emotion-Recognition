import os

import split_data
import dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNN


def main():
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

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    cnn = CNN()
    cnn.load_state_dict(torch.load('models/model0.pth'))

    criterion = nn.CrossEntropyLoss()
    params = cnn.parameters()
    optimiser = optim.Adam(params=params, lr=1e-3)
    log_interval = 80

    for epoch in range(50):
        print('epoch: ', epoch + 1)
        cnn.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data

            optimiser.zero_grad()

            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            if i % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, i * len(inputs), len(train_loader.dataset),
                    100. * i / len(train_loader), loss.item()))

    print('training complete')
    torch.save(cnn.state_dict(), f'models/model0.pth')


if __name__ == '__main__':
    main()
