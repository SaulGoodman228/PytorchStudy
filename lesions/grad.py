import numpy as np
import matplotlib.pyplot as plt
import time

def f(x):
    return np.sin(x)+0.5*x

def df(x):
    return np.cos(x)+0.5

N=20    #iter
xx=2    #start numb
lmd=0.3 #shag shod
mn=100

x_plt=np.arange(-5,5,0.1)
f_plt=[f(x) for x in x_plt]

plt.ion()
fig,ax= plt.subplots()
ax.grid(True)

ax.plot(x_plt,f_plt)
point = ax.scatter(xx,f(xx),c='red')

for i in range(N):
    lmd = 1/min(i+1,mn)
    xx = xx - lmd*np.sign(df(xx))

    point.set_offsets([xx,f(xx)])

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.02)

plt.ioff()
print(xx)
ax.scatter(xx,f(xx),c='blue')
plt.show()