import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


class MyDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_labels = pd.read_csv()

    def __len__(self)
        return len(self.img_labels)
