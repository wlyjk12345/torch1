import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from loss_func import MyLoss
x = np.mat('0 0;'
           '0 1;'
           '1 0;'
           '1 1')
x = torch.tensor(x).float()
y = np.mat('1;'
           '0;'
           '0;'
           '1')
y = torch.tensor(y).float()
z = np.mat('0 0;'
           '0 0;'
           '1 1;'
           '1 1')
z = torch.tensor(z).float()
'''
myNet = nn.Sequential(         #  相当于 test4中Perceptron族
    nn.Linear(2,10),
    nn.ReLU(),
    nn.Linear(10,1),
    nn.Sigmoid()
    )'''
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        return x
myNet = MyNet()
print(myNet)

optimzer = torch.optim.SGD(myNet.parameters(),lr=1e-2)  #优化器
myloss = MyLoss()  #损失函数

for epoch in range(5000):
    out = myNet(x)
    loss = myloss(out,y)   #  loss = torch.mean((out - y)**2)
    optimzer.zero_grad()   # 梯度清零，等价于net.zero_grad()
    loss.backward() # fake backward
    optimzer.step() # 执行优化

print(myNet(x).data.storage())
#print(myNet(z).data)
#print(optimzer)