from torch.utils.data import dataloader
from model import CNN, Net
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
import torch.nn as nn
from dataset import MyDataset
import os
from torchvision import transforms
from torchvision.transforms import ToTensor

def check_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

def main(modelChoice):
    path = 'all_test'
    my_transforms = transforms.Compose([
        ToTensor()
    ])
    test_labels = []
    for im_name in os.listdir(path):
        test_labels.append(int(im_name[1]))

    test_dataset = MyDataset(path, test_labels, my_transforms, None)

    test_loader = dataloader.DataLoader(test_dataset, batch_size=8, shuffle=True)

    net = Net()
    # net = nn.DataParallel(net)

    models = os.listdir('models')
    print(modelChoice)

    net.load_state_dict(torch.load(f'models/model{modelChoice}.pth'))

    criterion = nn.CrossEntropyLoss()
    params = net.parameters()
    optimiser = optim.Adam(params=params, lr=1e-3)

    net.eval()
    with torch.no_grad():
        net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs, labels = data
                output = net(inputs)
                test_loss += criterion(output, labels).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(labels.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))


if __name__=='__main__':
    # path = 'all_test/'
    # my_transforms = transforms.Compose([
    #     ToTensor()
    # ])
    # test_labels = []
    # for im_name in os.listdir(path):
    #     test_labels.append(int(im_name[1]))
    # test_dataset = MyDataset(path, test_labels, my_transforms, None)
    #
    # test_loader = dataloader.DataLoader(test_dataset, batch_size=4, shuffle=True)
    # models = os.listdir('models')
    # model_path = models[-1]
    # cnn = CNN()
    # cnn.load_state_dict(torch.load(f'models/{model_path}'))
    # check_accuracy(cnn, test_loader)
    modelChoice = str(input("Enter model number: "))
    main(modelChoice)
