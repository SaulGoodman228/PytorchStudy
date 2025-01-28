import torch

#Функция активации
def act(x):
    return 0 if x>=0 else 1

w_hiden = torch.FloatTensor([[1,1,-1.5],[1,1,-0.5]])
w_out = torch.FloatTensor([-1,-1,-0.5]) #Веса для выходного нейрона

data_x=[0,0]
x = torch.FloatTensor(data_x+[1])

z_hiden = torch.matmul(w_hiden,x)
print(z_hiden)

u_hiden = torch.FloatTensor([act(x) for x in z_hiden]+[1])
print(u_hiden)

z_out = torch.dot(w_out,u_hiden)
y = act(z_out)
print(y)