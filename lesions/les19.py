import os
import json

from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as tfs


class DigitDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, 'train' if train else 'test')
        self.transform = transform

        with open(os.path.join(path, "format.json"), "r") as f:
            self.data = json.load(f)

        self.length = 0
        self.files =[]
        self.targets = torch.eye(10)

        for _dir, _target in self.data.items():
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path) #Список файлов в каждом катологе
            self.length += len(list_files) #Считаем кол-во файлов
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target),list_files))

    def __getitem__(self, index):
        path_file, target = self.files[index]
        t = self.targets[index]
        img = Image.open(path_file)

        if self.transform is not None:
            img = self.transform(img).ravel().float() / 255.0

        return img, t

    def __len__(self):
        return self.length

d_train = DigitDataset('../Datasets/dataset')
train_dat = data.DataLoader(d_train, batch_size=32, shuffle=True)
print()