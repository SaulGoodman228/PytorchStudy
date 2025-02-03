#Персептрон
import torch
import matplotlib.pyplot as plt

N=5
b=-3

x1 = torch.rand(N)
x2 = x1 + torch.randint(1,10,[N]) / 10 +b
C1 = torch.vstack([x1,x2, torch.ones(N)]).mT #Объединение в один тензор

x1 = torch.rand(N)
x2 = x1 - torch.randint(1,10,[N]) / 10 +b
C2 = torch.vstack([x1,x2, torch.ones(N)]).mT #Объединение в один тензор

f = [0+b,1+b]

w1 = -0.5
w2 = -w1
w3 = -b*w2

w = torch.FloatTensor([-w1,w2,w3]) #Веса

# Функция классифицирует принадлежность точек к одному из классов
for i in range(N):
    x=C1[:][i]
    y=torch.dot(w,x) #Скал. произв. векторов
    print(x,y)
    if y >= 0:
        print('C1')
    else:
        print('C2')

plt.scatter(C1[:,0],C1[:,1],s=10,c='red')
plt.scatter(C2[:,0],C2[:,1],s=10,c='blue')
plt.plot(f)
plt.grid()
plt.show()

