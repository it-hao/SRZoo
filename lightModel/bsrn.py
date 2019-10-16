import torch
import torch.nn as nn
import math

'''
    This is PyTorch implementation of BSRN('Lightweight and Efficient Image Super-Resolutionwith Block 
    State-based Recursive Network'). The original code is based on TensorFlow and the github is 
    'https://github.com/idearibosome/tf-bsrn-sr'
'''

def make_model(args, parent=False):
    return BSRN(args)

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

class Residual_block(nn.Module):
    def __init__(self, n_feats, kernel_size, act):
        super(Residual_block, self).__init__()

        self.conv1 = nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size, 1, kernel_size // 2)
        self.conv2 = nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size, 1, kernel_size // 2)
        self.conv3 = nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size, 1, kernel_size // 2)
        self.act = act

    def forward(self, x, state):
        concat = torch.cat([x, state], 1)
        res = self.conv1(concat)
        res = self.conv2(self.act(res))
        res, state = torch.split(res, (64, 64), dim = 1)
        res = res + x
        state = self.conv3(torch.cat([res, state], dim=1))
        res2, state = torch.split(state, (64, 64), dim=1)
        return res + res2, state

# recursive residual block (n_recursions = 15)
class RRB(nn.Module):
    def __init__(self, n_recursions, n_feats, kernel_size, act):
        super(RRB, self).__init__()
        self.n_recursions = n_recursions
        self.block = Residual_block(n_feats, kernel_size, act)

    def forward(self, x):
        # state = torch.zeros(x.shape).cuda() # 第一个state初始化为0，并且gpu加速
        state = torch.zeros(x.shape) # 第一个state初始化为0，并且gpu加速
        for i in range(self.n_recursions):
            x, state = self.block(x, state)
        return x

class BSRN(nn.Module):
    def __init__(self):
        super(BSRN, self).__init__()

        n_recursions = 15
        n_colors = 3
        n_feats = 64
        kernel_size = 3
        scale = 4
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

        # feature shallow extraction layer
        self.head = nn.Conv2d(n_colors, n_feats, kernel_size, padding=kernel_size // 2)

        # middle feature extraction layer
        self.body = RRB(n_recursions, n_feats, kernel_size, act)

        # upsample and reconstruction layer
        self.tail = nn.Sequential(*[
            Upsampler(scale, n_feats, act=None),
            nn.Conv2d(n_feats, n_colors, kernel_size=3, padding=kernel_size // 2)
        ])

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = self.add_mean(x)
        return x

from torchstat import stat
net = BSRN()
stat(net, (3, 10, 10))