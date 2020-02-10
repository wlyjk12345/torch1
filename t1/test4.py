import torch
import torch as t
from torch import nn
import numpy as np
from loss_func import MyLoss

class Linear(nn.Module):  # 继承nn.Module
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()  # 等价于nn.Module.__init__(self)
        self.w = nn.Parameter(t.randn(in_features, out_features))
        self.b = nn.Parameter(t.randn(out_features))

    def forward(self, x):
        x = x.mm(self.w)  # x.@(self.w)
        return x + self.b.expand_as(x)
'''
layer = Linear(4, 3)   #等价于nn.Linear
input = t.randn(2, 4)
output = layer(input)
'''
class Perceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        nn.Module.__init__(self)
        self.layer1 = Linear(in_features, hidden_features) # 此处的Linear是前面自定义的全连接层
        self.layer2 = nn.Linear(hidden_features, out_features)
    def forward(self,x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        return self.layer2(x)
perceptron = Perceptron(2,10,1)

# Sequential的三种写法
net1 = nn.Sequential()
net1.add_module('conv', nn.Conv2d(3, 3, 3))
net1.add_module('batchnorm', nn.BatchNorm2d(3))
net1.add_module('activation_layer', nn.ReLU())

net2 = nn.Sequential(
        nn.Conv2d(3, 3, 3),
        nn.BatchNorm2d(3),
        nn.ReLU()
        )

from collections import OrderedDict
net3= nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(3, 3, 3)),
          ('bn1', nn.BatchNorm2d(3)),
          ('relu1', nn.ReLU())
        ]))
print('net1:', net1)
print('net2:', net2)
print('net3:', net3)

import torch
torch.cuda.is_available()

