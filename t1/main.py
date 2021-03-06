import os
import ipdb
import torch as t
import torchvision as tv
import tqdm
from model import NetG, NetD
from torchnet.meter import AverageValueMeter
from torch.autograd import Variable
import numpy as np

class Config(object):
    data_path = 'data/'  # 数据集存放路径
    num_workers = 4  # 多进程加载数据所用的进程数
    image_size = 96  # 图片尺寸
    batch_size = 256
    max_epoch = 200
    lr1 = 2e-4  # 生成器的学习率
    lr2 = 2e-4  # 判别器的学习率
    beta1 = 0.9  #  RMSprop优化器的alphe参数
    gpu = True  # 是否使用GPU
    nz = 100  # 噪声维度
    ngf = 64  # 生成器feature map数
    ndf = 64  # 判别器feature map数

    save_path = 'imgs/'  # 生成图片保存路径

    vis = False  # 是否使用visdom可视化
    env = 'GAN'  # visdom的env
    plot_every = 20  # 每间隔20 batch，visdom画图一次

    debug_file = '/tmp/debuggan'  # 存在该文件则进入debug模式
    d_every = 1  # 每1个batch训练一次判别器
    g_every = 5  # 每5个batch训练一次生成器
    save_every = 10# 没10个epoch保存一次模型
    save_new = None
    netd_path = None
    netg_path = None
    '''
    if save_new :
        netd_path = None
        netg_path = None
    else :
        netd_path = 'checkpoints/netd_%s.pth' %save_new  #'#'checkpoints/netd_%s.pth' %save_new#'checkpoints/netd_%s.pth' %save_new#'checkpoints/netd_200.pth'   # 'checkpoints/netd_.pth' #预训练模型
        netg_path = 'checkpoints/netg_%s.pth' %save_new  #'#'checkpoints/netg_%s.pth' %save_new#'checkpoints/netg_%s.pth' %save_new#'checkpoints/netg_200.pth'    # 'checkpoints/netg_211.pth'
    '''
    # 只测试不训练
    gen_img = 'result.png'
    # 从512张生成的图片中保存最好的64张
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0  # 噪声的均值
    gen_std = 1  # 噪声的方差


opt = Config()

def compute_gradient_penalty(D, real_samples, fake_samples):

    alpha = t.rand((opt.batch_size, 1, 1, 1))
    if opt.gpu:
        alpha = alpha.cuda()
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates =D(interpolates)
    #fake = Variable(t(opt.batch_size, 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = t.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=t.ones(d_interpolates.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device('cuda') if opt.gpu else t.device('cpu')
    if opt.vis:
        from visualize import Visualizer
        vis = Visualizer(opt.env)

    # 数据
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    dataloader = t.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=opt.num_workers,
                                         drop_last=True
                                         )

    # 网络
    netg, netd = NetG(opt), NetD(opt)
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    netd.to(device)
    netg.to(device)

    # 定义优化器和损失
    optimizer_g = t.optim.RMSprop(netg.parameters(), opt.lr1, alpha= opt.beta1)
    optimizer_d = t.optim.RMSprop(netd.parameters(), opt.lr2, alpha= opt.beta1)
    for p in NetD.parameters(netd):
        p.data.clamp_(-0.01, 0.01)
    one = t.FloatTensor([1])
    mone = -1 * one
    criterion = t.nn.BCELoss().to(device)

    # 真图片label为1，假图片label为0
    # noises为生成网络的输入
    true_labels = Variable(t.ones(opt.batch_size).to(device))
    fake_labels = Variable(t.zeros(opt.batch_size).to(device))
    fix_noises = Variable(t.randn(opt.batch_size, opt.nz, 1, 1).to(device))
    noises = Variable(t.randn(opt.batch_size, opt.nz, 1, 1).to(device))

    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()

    epochs = range(opt.max_epoch)
    for epoch in iter(epochs):
        if (epoch + 1) % opt.save_every == 0:
            # 保存模型、图片
            # tv.utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (opt.save_path, epoch), normalize=True,
            #  range=(-1, 1))
            t.save(netd.state_dict(), 'checkpoints/netd_%s.pth' % epoch)
            t.save(netg.state_dict(), 'checkpoints/netg_%s.pth' % epoch)
            errord_meter.reset()
            errorg_meter.reset()
            print(epoch)
        for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
            real_img = img.to(device)

            #real_imgs = Variable(t.from_numpy(real_img).cuda())
            z = Variable(t.randn(opt.batch_size, opt.nz, 1, 1).to(device))

            # Generate a batch of images
            gen_imgs = netg(z)

            if ii % opt.d_every == 0:
                # 训练判别器
                optimizer_d.zero_grad()
                ## 尽可能的把真图片判别为正确
                output = netd(real_img)
                error_d_real = criterion(output, true_labels)
                #error_d_real = output
                #error_d_real.backward()

                ## 尽可能把假图片判别为错误
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises).detach()  # 根据噪声生成假图
                output = netd(fake_img)
                error_d_fake = criterion(output, fake_labels)
                #error_d_fake=output
               # error_d_fake.backward(mone)
                optimizer_d.step()

                gradient_penalty = compute_gradient_penalty(netd, real_img.data, gen_imgs.data)


                error_d = (error_d_fake + error_d_real) / 2 + (gradient_penalty * 10)
                error_d.backward()
                optimizer_d.step()

                #error_d = error_d_fake + error_d_real

                errord_meter.add(error_d.item())

            if ii % opt.g_every == 0:
                # 训练生成器
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()
                errorg_meter.add(error_g.item())
'''
            if opt.vis and ii % opt.plot_every == opt.plot_every - 1:
                ## 可视化
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                fix_fake_imgs = netg(fix_noises)
                vis.images(fix_fake_imgs.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
                vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
                vis.plot('errord', errord_meter.value()[0])
                vis.plot('errorg', errorg_meter.value()[0])
'''
@t.no_grad()
def generate(**kwargs):
    """
    随机生成动漫头像，并根据netd的分数选择较好的
    """
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device('cuda') if opt.gpu else t.device('cpu')

    netg = NetG(opt).eval()
    netd = NetD(opt).eval()
    noises = t.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
    noises = noises.to(device)

    map_location = lambda storage, loc: storage
    netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    netd.to(device)
    netg.to(device)

    # 生成图片，并计算图片在判别器的分数
    fake_img = netg(noises)
    scores = netd(fake_img).detach()

    # 挑选最好的某几张
    indexs = scores.topk(opt.gen_num)[1]
    result = []
    for ii in indexs:
        result.append(fake_img.data[ii])
    # 保存图片
    tv.utils.save_image(t.stack(result), '%s/%s————'%(opt.save_path, opt.save_new) + opt.gen_img, normalize=True, range=(-1, 1))
    #torchvision.utils.save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False)

if __name__ == '__main__':
    import fire

    fire.Fire()


train()
#generate()