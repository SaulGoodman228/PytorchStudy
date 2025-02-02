import os
import json

import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as tfs
from torch import optim

class RavelTransform(nn.Module):
    def forward(self, item):
        return item.ravel()


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
        t = self.targets[target]
        img = Image.open(path_file)

        if self.transform is not None:
            img = self.transform(img).ravel().float() / 255.0

        return img, t

    def __len__(self):
        return self.length

class DigitNN(nn.Module):
    def __init__(self, input_size, num_classes, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_size, num_classes)
        self.layer2 = nn.Linear(num_classes, output_dim)
        self.bm_1 = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        x = self.layer1(x)  # Получение значений суммы нейронов
        x = nn.functional.relu(x)  # Функция активации (Гиперболический тангенс)
        x = self.bm_1(x)
        x = self.layer2(x)
        return x

model = DigitNN(28*28, 128, 10)

transforms = tfs.Compose([tfs.ToImage(),tfs.Grayscale(),
                         tfs.ToDtype(torch.float32, scale = True),
                        RavelTransform()])  # tfs.Grayscale() - для преобразования в один цветовой канал

#Стандартный датасет MNIST
dataset_mnist = torchvision.datasets.MNIST(r'C:\datasets\mnist', download=True, train=True, transform=transforms)
d_train, d_val = data.random_split(dataset_mnist, [0.7, 0.3]) #Разделение на обучающую и валидационную выборку
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True) #Создание итераторов для получения батчей
train_data_val = data.DataLoader(d_val, batch_size=32, shuffle=False)


optimizer = optim.Adam(params=model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()
epochs = 20

loss_lst_val = [] # список значений потерь при валидации
loss_lst = [] # список значений потерь при обучении


for e in range(epochs):
    model.train()
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=False)
    for x_train, y_train in train_tqdm:

        predict = model(x_train)
        loss = loss_function(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1/ lm_count * loss.item() + (1-1/lm_count)*loss_mean #Вычисление среднего значения функции потерь
        train_tqdm.set_description(f'Epoch [{e+1}/{epochs}], Loss_mean: {loss_mean:.4f}] ')

    model.eval()
    Q_val = 0
    count_val = 0

    for x_val, y_val in train_data_val:
        with torch.no_grad():
            p = model(x_val)
            loss = loss_function(p, y_val)
            Q_val += loss.item()
            count_val += 1

    Q_val /= count_val

    loss_lst.append(loss_mean)
    loss_lst_val.append(Q_val)

    print(f" | loss_mean={loss_mean:.3f}, Q_val={Q_val:.3f}")

d_test = ImageFolder("dataset/test", transform=transforms)
test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)

Q = 0

model.eval()

for x_test, y_test in test_data:
    with torch.no_grad():
        p = model(x_test) #Формируются тензоры 500 на 10
        p = torch.argmax(p, dim=1) #Во второй оси выбирается вектор с макс значением, из-за того что все кроме определнного 0, то выбирается значение
        Q += torch.sum(p==y_test).item() # Подсчитываем количество правильных классификаций

Q /= len(d_test)
print(Q)

plt.plot(loss_lst)
plt.plot(loss_lst_val)
plt.grid()
plt.show()