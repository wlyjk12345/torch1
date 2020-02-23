import torch
import torch.nn as nn

class NetG(nn.Module):
    """
    生成器定义
    """
    def __init__(self,opt):
        super(NetG, self).__init__()
        ngf = opt.ngf  # 生成器feature map数

        self.main = nn.Sequential(
            # 输入是一个nz维度的噪声，我们可以认为它是一个nz*1*1的feature map
            nn.ConvTranspose2d(in_channels=opt.nz, out_channels=ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8), #批 规范化 层
            nn.ReLU(True),   # (True)会把输出直接覆盖到输入中
            # 上一步的输出形状：(ngf*8) x 4 x 4
            #nn.ConvTranspose2d(in_channels, out_channel, kernel_size, stride, padding，output_padding=, bias)
            # 逆卷积 卷积核 步长(扩大倍数) 输入填充(加边) 输出填边 添加偏离
            #out =output_padding + (in - 1 )* Stride - 2 * padding + kernel_size
            # 输入是一个nz维度的噪声，我们可以认为它是一个1*1*nz的feature map

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 上一步的输出形状： (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 上一步的输出形状： (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 上一步的输出形状：(ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()  # 输出范围 -1~1 故而采用Tanh
            # 输出形状：3 x 96 x 96
        )

    def forward(self, input):
        return self.main(input)


class NetD(nn.Module):
    """
    判别器定义
    """

    def __init__(self, opt):
        super(NetD, self).__init__()
        ndf = opt.ndf

        self.main = nn.Sequential(
            # 输入 3 x 96 x 96
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # 输出 (ndf) x 32 x 32

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(ndf * 2, 0.8),
            # 输出 (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(ndf * 4, 0.8),
            # 输出 (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(ndf * 8, 0.8),
            # 输出 (ndf*8) x 4 x 4


            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),

            nn.Sigmoid()  # 输出一个数(概率)
        )
    def forward(self, input):
        return self.main(input).view(-1)

