import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class MyDataSet(Dataset):
    """自定义数据集"""
    
    def __init__(self, images_path, labels_path, transform=None):
        self.images = [images_path + f for f in os.listdir(images_path) if f.endswith('.png')]
        self.labels = [labels_path + f for f in os.listdir(labels_path) if f.endswith('.png')]
        self.transform = transform

    def __getitem__(self, item):
        img = Image.open(self.images[item])
        lab = Image.open(self.labels[item])
        if self.transform is not None:
            img = self.transform(img)
            lab = self.transform(lab)
        # img = np.array(img)
        # lab = np.array(lab)
        # img = torch.from_numpy(img.astype(np.float32))
        # lab = torch.from_numpy(lab.astype(np.float32))
        return img, lab

    def __len__(self):
        return len(self.images)

class testDataSet(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images = [images_path + f for f in os.listdir(images_path) if f.endswith('.png')]
        self.labels = [labels_path + f for f in os.listdir(labels_path) if f.endswith('.png')]
        self.transform = transform
        self.names = [f for f in os.listdir(images_path) if f.endswith('.png')]

    def __getitem__(self, item):
        img = Image.open(self.images[item])
        lab = Image.open(self.labels[item])
        name = self.names[item]
        if self.transform is not None:
            img = self.transform(img)
            lab = self.transform(lab)
        # img = np.array(img)
        # lab = np.array(lab)
        # img = torch.from_numpy(img.astype(np.float32))
        # lab = torch.from_numpy(lab.astype(np.float32))
        return img, lab, name

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        return images, labels
