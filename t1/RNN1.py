import torch as t
import torch.nn as nn
t.manual_seed(1000)
# 输入：batch_size=3，序列长度都为2，序列中每个元素占4维
input = t.randn(2, 3, 4)

# lstm输入向量4维，隐藏元3，1层
lstm = nn.LSTM(4, 3, 1)
# 初始状态：1层，batch_size=3，3个隐藏元
h0 = t.randn(1, 3, 3)
c0 = t.randn(1, 3, 3)
out, hn = lstm(input, (h0, c0))

# 一个LSTMCell对应的层数只能是一层
lstm = nn.LSTMCell(4, 3)
hx = t.randn(3, 3)
cx = t.randn(3, 3)
out = []
for i_ in input:
    hx, cx=lstm(i_, (hx, cx))
    out.append(hx)
t.stack(out)

print(out)