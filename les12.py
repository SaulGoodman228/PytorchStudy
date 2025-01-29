#Градиентный спуск(Back propagation)
import torch
from random import randint

# 2 слоя нейронов
#Входные данные


#Функция активации - гиперболический тангенс
def act(x):
    return torch.tanh(x)

#Производная
def df(x):
    s = act(x)
    return 1 - s*s

#пропускает вектор x_inp через нейронную сеть
#x_inp содержит три компоненты [x1,x2,x3]
def go_forward(x_inp,w1,w2):
    z1 = torch.mv(w1[:,:3], x_inp) + w1[:,3] #добавляем байес(смещение)
    s1 =act(z1)

    z2 = torch.dot(w2[:2], s1) + w2[2]
    s2 = act(z2)
    return s2,z1,s1,z2

#Задаем значения случайно в диапазоне [-0.5,0.5] для первого вектора, который преобразуем в матрицу 2х4
W1 = torch.rand(8).view(2,4) - 0.5
W2 = torch.rand(3) - 0.5

torch.manual_seed(1)

#Обучающая выборка х и соответствующие для нее значения у
x_train = torch.FloatTensor([(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)])
y_train = torch.FloatTensor([-1, 1, -1, 1, -1, 1, -1, -1])

lmd = 0.05  # шаг обучения
N = 1000  # число итераций при обучении
total = len(y_train) # размер обучающей выборки

for _ in range(N):
    k = randint(0,total-1)
    x = x_train[k] #Выбираем случайный образ з обучающей выборки

    y, z1, s1, out = go_forward(x,W1,W2) #Получаем выходные значения нейронов
    e = y - y_train[k] #Расчет функции потерь
    delta = e * df(out) #Расчет локального градиента
    delta2 = W2[:2] * delta * df(z1) #Вектор из двух локальных градиентов скрытого слоя

    W2[:2] = W2[:2] - lmd * delta * df(z1) #Корректировка весов полей для слоя
    W2[2] = W2[2] - lmd * delta # корректировка смещения

    # Корректировка связей первого слоя
    W1[0, 3] = W1[0, 3] - lmd * delta2[0]
    W1[1, 3] = W1[1, 3] - lmd * delta2[1]

for x,d in zip(x_train,y_train):
    y,z1,s1,out = go_forward(x,W1,W2)
    print(f'Выходное значение НС: {y} => {d}')

print(W1)
print(W2)


