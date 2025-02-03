import  torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torchvision
import torch.nn as nn
import torch.optim as optim

class DigitNN(nn.Module):
    def __init__(self, input_size, num_classes, output_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_size, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, output_dim),
        )

    def forward(self, x):
        return self.net(x)

block = nn.Sequential(
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
    nn.LeakyReLU(),
)

#Последовательно запсываем модель
model = nn.Sequential()
model.add_module('layer1',nn.Linear(28*28,32))
model.add_module('relu',nn.ReLU())
model.add_module('block',block)
model.add_module('layer2',nn.Linear(32,10))
