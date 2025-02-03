import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def act(x): #пороговая функция актвации
    return 0 if x<0.5 else 1

def go(house, rock, attr):
    X =torch.tensor([house, rock, attr],dtype=torch.float32) # первый тензор
    Wh= torch.tensor([[0.3,0.3,0],[0.4,-0.5,1]],dtype=torch.float32) #второй тензор(веса)
    Wout = torch.tensor([-1.,1.]) #Тензор результирующей функции

    Zh=torch.mv(Wh,X) #Умножение Mатрицы на вектор
    print(f'Значение сумм на нейронах скрытого слоя: {Zh}')

    Uh =torch.tensor([act(x) for x in Zh],dtype=torch.float32) #Вывод результирующей функциии для скрытого нейрона
    print(f'Значение сумм на нейронах скрытого слоя: {Uh}')

    Zout = torch.dot(Wout,Uh)
    Y = act(Zout)
    print(f'Выходное значение нейросети: {Y}')


go(1,0,1)

