import numpy as np
import torch
'''
x = torch.empty(5, 3) #未初始化的矩阵
#x = torch.rand(5, 3)  #随机数初始化的矩阵，里面的数值为限定时是在[0,1]
#x = torch.zeros(5, 3, dtype=torch.long)
#x = torch.tensor([5.5, 3, 3])#张量
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes 全一阵
x = torch.randn_like(x, dtype=torch.float)    # override dtype! # result has the same size
print(x.size())
print(x.type())
print(x.item())    #打印数字
# add operation
y = torch.rand(5, 3)
result = torch.empty(5, 3)
torch.add(x, y, out=result)  #x + y
print(result)
# adds x to y
y.add_(x)
print(y)
x.copy_(y)  # same as np with an _
print(x[:,1]) # 从零开始
# resize
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
''''''
x = torch.randn(1)
print(x)
print(x.item())
# numpy
a = torch.ones(5)
b = a.numpy()
print(b)
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
'''
x = torch.ones(2, 2, requires_grad=True) #当设置它的属性 .requires_grad=True，那么就会开始追踪在该变量上的所有操作，而完成计算后，可以调用 .backward() 并自动计算所有的梯度，得到的梯度都保存在属性 .grad 中。
print(x)
y = x - 2
#print(x.requires_grad)
#print(y.grad_fn) #was created as a result of an operation
#with torch.no_grad():  #希望防止跟踪历史
#    print((x ** 2).requires_grad)
x.requires_grad_(True)
#out = y.mean()
#out.backward()
v = torch.tensor([[1., 1.],[1., 1.]], dtype=torch.float)
y.backward(v)
print(y)
# 输出梯度 d(out)/dx
print(x.grad)