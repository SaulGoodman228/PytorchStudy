import torch
import torch.nn as nn

rnn = nn.LSTM(10,16,batch_first=True,bidirectional=True)

x = torch.randn(1,5,10)
y, (h,c) = rnn(x)

print(y.size())
print(h.size())
print(c.size())