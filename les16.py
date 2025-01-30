import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randint




class MyModule(nn.Module):
    def __init__(self, input_size,num_hiden, output_size):
        #Обязательный инициализатор родительского класса
        super(MyModule, self).__init__()
        self.layer1 = nn.Linear(input_size, num_hiden)
        self.layer2 = nn.Linear(num_hiden, output_size)


    #Реализация прямого прохода по слою или нейросети
    def forward(self, x):
        x = self.layer1(x)  # Получение значений суммы нейронов
        x = F.tanh(x)  # Функция активации (Гиперболический тангенс)
        x = self.layer2(x)
        x = F.tanh(x)
        return x

#Фунция что пропускает значения через сеть
def forward(inp,l1: nn.Linear, l2:nn.Linear):
    u1 = l1.forward(inp)  #Получение значений суммы нейронов
    s1 = F.tanh(u1) #Функция активации (Гиперболический тангенс)
    print(s1)
    u2 = l2.forward(s1)
    s2 = F.tanh(u2)
    print(s2)
    return s2

#Обучающая выборка х и соответствующие для нее значения у
x_train = torch.FloatTensor([(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)])
y_train = torch.FloatTensor([-1, 1, -1, 1, -1, 1, -1, -1])
total = len(y_train) # размер обучающей выборки


model = MyModule(input_size=3,num_hiden=2,output_size=1)

optimizer = optim.RMSprop(model.parameters(), lr=0.01)
loss_funk = nn.MSELoss()

model.train()

for _ in range(1000):
    k = randint(0,total-1) #выбираем случайный образ из выборки
    y = model(x_train[k]) #Пропускаем образ через модель  получаем значение
    y = y.squeeze() #Добавляет в соответствие значению дополнительную ось [Кол_во батчей, значение]
    loss = loss_funk(y,y_train[k]) #Функция потерь

    #Шаг стохастического градиентного спуска
    optimizer.zero_grad() #Обнуление
    loss.backward()       #Поиск значений частных производных
    optimizer.step()      #Корректировка весов

model.eval()
for x,d in zip(x_train,y_train):
    y=model(x)
    print(f'Выходное значение НС: {y.data} => {d}')


# layer1 = nn.Linear(in_features=3, out_features=2)
# layer2 =nn.Linear(2,1)
#
# #Задаем заранее полученные веса из прошлых обучений
# layer1.weight.data = torch.tensor([[0.7402,0.6008,0.-1.334],[0.2098,0.4537,-0.7692]])
# layer1.bias.data = torch.tensor([0.5505,0.3719])
#
# layer2.weight.data = torch.tensor([[-2.0719,-0.9485]])
# layer2.bias.data = torch.tensor([-0.1461])

# x = torch.FloatTensor([1,-1,1]) #Параметры входного вектора
# y = forward(x,layer1,layer2) #Пропускаем через нейронную сеть
# print(y.data)