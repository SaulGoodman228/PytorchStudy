import torch
import torch.optim as optim

from random import randint
import matplotlib.pyplot as plt

#Результат матричного умножения
def model(X,w):
    return X @ w

N=2
w = torch.FloatTensor(N).uniform_(-1e-5, 1e-5)
w.requires_grad_(True) #Указывает на автоматическое вычисление градиентов
x =torch.arange(0,3,0.1)

y_train = 0.5 * x + 0.2 * torch.sin(2*x) - 3.0
x_train = torch.tensor([[_x ** _n for _n in range(N)] for _x in x])

total = len(x) #размер выборки
lr = torch.tensor([0.1,0.01]) #Шаг сходbмости

loss_funk = torch.nn.MSELoss() #Стандартная функция потерь
optimum =optim.SGD([w], lr=0.01, momentum=0.8, nesterov=True) #Оптимизатор

#Цикл для стохастического градиентного спуска
for _ in range(1000):
    k = randint(0,total-1) #выбираем случайный образ из выборки
    y = model(x_train[k],w) #Пропускаем образ через модель  получаем значение
    loss = loss_funk(y,y_train[k]) #Функция потерь

    loss.backward()
    # w.data -= lr*w.grad #В свойстве .grad хранятся прозводные
    # w.grad.zero_()
    optimum.step()
    optimum.zero_grad()

print(w)
predict = model(x_train,w)

plt.plot(x,y_train.numpy())
plt.plot(x, predict.data.numpy())
plt.grid()
plt.show()
