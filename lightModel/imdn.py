import torch
import torch.nn as nn
import math

'''
    This is PyTorch implementation of IMDN('Lightweight Image Super-Resolution with Information Multi-distillation Network').
    The original code can found in 'https://github.com/Zheng222/IMDN'.
'''

def make_model(args, parent=False):
    return IMDN(args)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * 255. * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, act):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1))
                m.append(nn.PixelShuffle(2))
                if act: m.append(nn.ReLU(True))
        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, 3, padding=1))
            m.append(nn.PixelShuffle(3))
            if act is not None: m.append(act)
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

# contrast-aware channel attention module
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

# Information multi-distillation block
class IMDB(nn.Module):
    def __init__(self, n_feats, kernel_size, distillation_rate, act):
        super(IMDB, self).__init__()
        self.distilled_feats = int(n_feats * distillation_rate)
        self.remaining_feats = int(n_feats - self.distilled_feats)

        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size, 1, kernel_size//2)
        self.conv2 = nn.Conv2d(self.remaining_feats, n_feats, kernel_size, 1, kernel_size//2)
        self.conv3 = nn.Conv2d(self.remaining_feats, n_feats, kernel_size, 1, kernel_size//2)
        self.conv4 = nn.Conv2d(self.remaining_feats, self.distilled_feats, kernel_size, 1, kernel_size//2)
        self.conv5 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        self.cca = CCALayer(self.distilled_feats * 4)
        self.act = act

    def forward(self, x):
        conv1 = self.act(self.conv1(x))
        dis_conv1, rem_conv1 = torch.split(conv1, (self.distilled_feats, self.remaining_feats), dim = 1)

        conv2 = self.act(self.conv2(rem_conv1))
        dis_conv2, rem_conv2 = torch.split(conv2, (self.distilled_feats, self.remaining_feats), dim = 1)

        conv3 = self.act(self.conv3(rem_conv2))
        dis_conv3, rem_conv3 = torch.split(conv3, (self.distilled_feats, self.remaining_feats), dim = 1)

        conv4 = self.act(self.conv4(rem_conv3))
        out = torch.cat([dis_conv1, dis_conv2, dis_conv3, conv4], dim = 1)
        out = self.conv5(self.cca(out)) + x
        return out

# Information multi-distillation network
class IMDN(nn.Module):
    def __init__(self):
        super(IMDN, self).__init__()

        self.n_blocks = 6
        distillation_rate = 0.25
        n_colors = 3
        n_feats = 64
        kernel_size = 3
        scale = 4
        act = nn.LeakyReLU(0.05)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

        # feature shallow extraction layer
        self.head = nn.Conv2d(n_colors, n_feats, kernel_size, padding=kernel_size // 2)

        # middle feature extraction layer
        modules = []
        for i in range(self.n_blocks):
            modules.append(IMDB(n_feats, kernel_size, distillation_rate, act))
        self.body = nn.Sequential(*modules)

        # features fusion
        self.fusion = nn.Sequential(*[nn.Conv2d(n_feats*self.n_blocks, n_feats, 1, 1, 0),
                                      act,
                                      nn.Conv2d(n_feats, n_feats, kernel_size, 1, kernel_size // 2)])

        # upsample and reconstruction layer
        self.tail = nn.Sequential(*[nn.Conv2d(n_feats, n_colors * (scale ** 2), kernel_size, 1, kernel_size//2),
                                    nn.PixelShuffle(scale)])

    def forward(self, x):
        x = self.sub_mean(x)
        res = x = self.head(x)
        middle_feats = []
        for i in range(self.n_blocks):
            x = self.body[i](x)
            middle_feats.append(x)
        x = self.fusion(torch.cat(middle_feats, dim = 1))
        x = self.tail(x + res)
        x = self.add_mean(x)
        return x

from torchstat import stat
net = IMDN()
stat(net, (3, 10, 10))