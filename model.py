import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.fc1 = nn.Linear(256 * 2 * 2, 50)
        self.fc2 = nn.Linear(50, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.tanh(x)

    '''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(1296, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        '''


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = conv_block(1, 64)

        self.conv1 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 64), conv_block(64, 128))
        # self.drop1 = nn.Dropout(0.6)

        self.conv2 = conv_block(128, 256, pool=True)
        self.res2 = nn.Sequential(conv_block(256, 128), conv_block(128, 256))
        # self.drop2 = nn.Dropout(0.6)

        self.conv3 = conv_block(256, 512, pool=True)
        self.res3 = nn.Sequential(conv_block(512, 256), conv_block(256, 512))
        self.drop3 = nn.Dropout(0.7)

        self.classifier = nn.Sequential(nn.MaxPool2d(6),
                                        nn.Flatten(),
                                        nn.Linear(512, 6))

    def forward(self, xb):
        out = self.input(xb)

        out = self.conv1(out)
        out = self.res1(out) + out
        # out = self.drop1(out)

        out = self.conv2(out)
        out = self.res2(out) + out
        # out = self.drop2(out)

        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.drop3(out)

        return self.classifier(out)
