from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from PIL.ImageOps import scale
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim




class ModelStyle(nn.Module):
    def __init__(self):
        super().__init__()
        _model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.mf = _model.features #Выделяем в модели только сверточные слои
        self.mf.requires_grad_(False) #Отключаем градиенты
        self.requires_grad_(False)
        self.mf.eval() #Переводим в режим эксплуатации
        #Вспомогательные переменные
        self.idx_out = (0, 5, 10, 19, 28, 34) #Номера слоев с которых будем брать данные
        self.num_style_layers = len(self.idx_out) - 1 # Поможет вычислить потери по стилю

    def forward(self, x):
        outputs = []
        for indx, layer in enumerate(self.mf): #Перебираем слои и сохраняем нужные без батча
            x = layer(x)
            if indx in self.idx_out:
                outputs.append(x.squeeze(0))

        return outputs

#Функция потерь ориг изображ.
def get_content_loss(base_content, target):
    return torch.mean( torch.square(base_content - target) )

#Функция потерь по стилю на слое
def gram_matrix(x):
  channels = x.size(dim=0) #Число каналов
  g = x.view(channels, -1) #Вытягивание в вектор
  gram = torch.mm(g, g.mT) / g.size(dim=1) # М * Мт / размер карты признаков
  return gram

#Общая функция потерь стиля
def get_style_loss(base_style, gram_target):
    style_weights = [1.0, 0.8, 0.5, 0.3, 0.1]

    _loss = 0
    i = 0
    for base, target in zip(base_style, gram_target):
        gram_style = gram_matrix(base)
        _loss += style_weights[i] * torch.mean(torch.square(gram_style - target))
        i += 1

    return _loss

img = Image.open('img/cat_1.jpg').convert('RGB')
img_style = Image.open('img/style_2.jpg').convert('RGB')

transforms = tfs_v2.Compose([tfs_v2.ToImage(),
                            tfs_v2.ToDtype(torch.float32, scale = True),
                            ])

img = transforms(img).unsqueeze(0) #Добавляем первую ось, размер пакетов (Batch)
img_style = transforms(img_style).unsqueeze(0)
img_create = img.clone()
img_create.requires_grad_(True)

model = ModelStyle()

outputs_img = model(img)
output_img_style = model(img_style)

#Матрица Грама
gram_matrix_style = [gram_matrix(x) for x in output_img_style[:model.num_style_layers]]
content_weight = 1 #Alfa
style_weight = 1000 #Beta
best_loss = -1 #Хранит лучшие потери
epochs = 50

optimizer = optim.Adam(params=[img_create], lr=0.01) #В качестве параметров указываются пиксели изображения
best_img = img_create.clone()

for _e in range(epochs):
    outputs_img_create = model(img_create) #Пропускаем изображение через модель

    loss_content = get_content_loss(outputs_img_create[-1], outputs_img[-1]) #Потери контента
    loss_style = get_style_loss(outputs_img_create, gram_matrix_style) #Потери стиля
    loss = content_weight * loss_content + style_weight * loss_style #Потери общие

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    img_create.data.clamp_(0, 1)
    #Лучший вариант стилизации
    if loss < best_loss or best_loss < 0:
      best_loss = loss
      best_img = img_create.clone()

    print(f'Iteration: {_e}, loss: {loss.item(): .4f}')

x = best_img.detach().squeeze() #Убираем слой батчей
low, hi = torch.amin(x), torch.amax(x) #Мин макс по пикселям
x = (x - low) / (hi - low) * 255.0 #Нормировка
x = x.permute(1, 2, 0) #Меняем оси для вывода
x = x.numpy()
x = np.clip(x, 0, 255).astype('uint8')

image = Image.fromarray(x, 'RGB')
image.save("result.jpg")

print(best_loss)
plt.imshow(x)
plt.show()