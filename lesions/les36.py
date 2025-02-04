import os
import json
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

class DogDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, "train" if train else "test")
        self.transform = transform

        with open(os.path.join(self.path, "format.json"), "r") as fp:
            self.format = json.load(fp)

        self.length = 0
        self.files = []
        self.targets = torch.eye(10)

        for _dir, _target in self.format.items():
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path)
            self.length += len(list_files)
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))

    def __getitem__(self, item):
        path_file, target = self.files[item]
        t = self.targets[target]
        img = Image.open(path_file)

        if self.transform:
            img = self.transform(img)

        return img, t

    def __len__(self):
        return self.length

path_to_json = Path('..') / 'Datasets' / 'dogs'

rs_wg = models.ResNet50_Weights.DEFAULT
transforms = rs_wg.transforms()

model = models.resnet50(weights=rs_wg)
model.requires_grad_(False)
model.fc = nn.Linear(512*4,10) #Заменяем последний слой для обучения
model.fc.requires_grad_(True)

d_train = DogDataset(path_to_json, transform=transforms)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

optimizer = optim.Adam(params=model.fc.parameters(), lr=0.001, weight_decay=0.001)
loss_function = nn.CrossEntropyLoss()
epochs = 3
model.train()

for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_function(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")

st = model.state_dict()
torch.save(st, 'model_transfer_resnet.tar')
# st = torch.load('model_transfer_resnet.tar', weights_only=False)
# model.load_state_dict(st)

d_test = DogDataset(path_to_json, train=False, transform=transforms)
test_data = data.DataLoader(d_test, batch_size=50, shuffle=False)

# тестирование обученной НС
Q = 0
P = 0
count = 0
model.eval()

test_tqdm = tqdm(test_data, leave=True)
for x_test, y_test in test_tqdm:
    with torch.no_grad():
        p = model(x_test)
        p2 = torch.argmax(p, dim=1)
        y = torch.argmax(y_test, dim=1)
        P += torch.sum(p2 == y).item()
        Q += loss_function(p, y_test).item()
        count += 1

Q /= count
P /= len(d_test)
print(Q)
print(P)