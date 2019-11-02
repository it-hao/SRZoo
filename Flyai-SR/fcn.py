import torch
import torch.nn as nn
import math

def make_model(args, parent=False):
    return NET(args)

class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

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


class ConcatBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, act):
        super(ConcatBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats*4, kernel_size, padding = kernel_size//2)
        self.conv2 = nn.Conv2d(n_feats*4, n_feats, kernel_size, padding = kernel_size//2)
        self.conv3 = nn.Conv2d(n_feats*2, n_feats, kernel_size = 1, padding = 0)
        self.act = act
        self.w = Scale(1)
        self.u = Scale(1)

    def forward(self, x):
        res = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        out = torch.cat([self.w(x), self.u(res)], 1)
        out = self.conv3(out)
        return out

class ConcatGroup(nn.Module):
    def __init__(self, n_feats, n_block, kernel_size, act):
        super(ConcatGroup, self).__init__()
        self.n_block = n_block
        concatblock = []
        for i in range(n_block):
            concatblock.append(ConcatBlock(n_feats, kernel_size, act))
        self.block = nn.Sequential(*concatblock)
        self.conv = nn.Conv2d(n_feats * 2, n_feats, 1, 1, 0)
        self.w = Scale(1)
        self.u = Scale(1)

    def forward(self, x):
        res = x
        for i in range(self.n_block):
            x = self.block[i](x)
        out = torch.cat([self.w(x), self.u(res)], 1)
        out = self.conv(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.n_group = n_group = 8
        n_block = 16
        n_colors = 3
        n_feats = 32
        kernel_size = 3
        scale = 4
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

        # feature shallow extraction
        self.head = nn.Conv2d(n_colors, n_feats, kernel_size, padding=kernel_size//2)

        # feature deep extraction
        body = nn.ModuleList()
        rescale = []
        for i in range(n_group):
            body.append(ConcatGroup(n_feats, n_block, kernel_size, act))
            rescale.append(Scale(1))
        self.group = nn.Sequential(*body)
        self.rescale = nn.Sequential(*rescale)

        # feature confusion
        self.confusion = nn.Sequential(*[
            nn.Conv2d(n_feats * n_group, n_feats, 1, 1, 0),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding = kernel_size//2)
        ])

        # upsample and reconstruction
        self.tail = nn.Sequential(*[
            Upsampler(scale, n_feats, act=None),
            nn.Conv2d(n_feats, n_colors, kernel_size, padding=kernel_size // 2)
        ])

    def forward(self, x):
        x = self.sub_mean(x)
        res = x = self.head(x)
        out = []
        for i in range(self.n_group):
            x = self.rescale(self.group[i](x))
            out.append(x)
        x = self.confusion(torch.cat(out, 1))
        x += res
        x = self.tail(x)
        x = self.add_mean(x)

        return x

# from torchstat import stat

# net = NET()
# stat(net, (3, 10, 10))

