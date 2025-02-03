import torch

#Специальное свойство обозначает что будут вычисляться производные по этим точкам
x = torch.tensor([2.], requires_grad=True)
y = torch.tensor([-4.], requires_grad=True)
#Тензоры это програмные еденицы, что неявно взаимодействуют с собой

f = (x+y)**2 +2*x*y
#вычисляет числовое значение по точке
f.backward()

print(f)
print(x.data,x.grad)
print(y.data,y.grad)