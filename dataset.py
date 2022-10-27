import os
import cv2
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, img_dir, labels, transform, label_transform):
        self.labels = labels
        self.images = []
        self.transforms = transform
        self.label_transform = label_transform
        for img_name in os.listdir(img_dir):
            self.images.append(cv2.imread(img_dir + '/' + img_name, 0))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transforms:
            image = self.transforms(image)
        if self.label_transform:
            label = self.label_transform(label)

        return image, label
