import torch
import torch.nn as nn
import math

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
    def __init__(self, scale, n_feats, wn, act):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(wn(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1)))
                m.append(nn.PixelShuffle(2))
                if act: m.append(nn.ReLU(True))
        elif scale == 3:
            m.append(wn(nn.Conv2d(n_feats, 9 * n_feats, 3, padding=1)))
            m.append(nn.PixelShuffle(3))
            if act is not None: m.append(act)
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class Attention(nn.Module):
    def __init__(self, channel, wn, reduction=16):
        super(Attention, self).__init__()

        # channel attention
        self.CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            wn(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True)),
            nn.ReLU(inplace=True),
            wn(nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True)),
        )
        # space attention
        self.SA = nn.Sequential(
            wn(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, dilation=1, groups=channel, bias=True)),
            nn.ReLU(True),
            wn(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, dilation=1, groups=channel, bias=True))
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        CA = self.CA(x)
        SA = self.SA(x)
        y = self.sigmoid(CA + SA)
        y = x * y
        return x + y

class ConvLayer(nn.Module):
    def __init__(self, num_convs, n_feats, kernel_size, wn, act):
        super(ConvLayer, self).__init__()
        self.num_convs = num_convs
        self.convs = nn.ModuleList()
        self.attes = nn.ModuleList()
        for i in range(num_convs):
            self.convs.append(nn.Sequential(
                wn(nn.Conv2d(n_feats, n_feats, kernel_size, 1, padding=kernel_size // 2)),
                act
            ))

    def forward(self, x):
        convs_out = []

        for i in range(self.num_convs):
            convs_out.append(self.convs[i](x))
        x = torch.cat(convs_out, 1)

        return x

class Block(nn.Module):
    def __init__(self, n_feats, kernel_size, wn, act):
        super(Block, self).__init__()

        self.num_convs = num_convs = 3
        self.num_layers = num_layers = 4

        self.convs = nn.ModuleList()
        self.confusion = nn.ModuleList()
        self.attes = nn.ModuleList()
        for i in range(1, num_layers + 1):
            self.convs.append(ConvLayer(num_convs, n_feats, kernel_size, wn, act))
            self.confusion.append(wn(nn.Conv2d(n_feats * (i * num_convs + 1), n_feats, 1, 1, 0)))
            self.attes.append(Attention(n_feats, wn))

    def forward(self, x):
        res = concat = x
        for i in range(self.num_layers):
            concat = torch.cat([concat, self.convs[i](x)], 1)
            x = self.attes[i](self.confusion[i](concat))
        return x + res


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.blocks = 15
        n_colors = 3
        n_feats = 64
        kernel_size = 3
        scale = 4
        act = nn.ReLU(True)
        wn = lambda x: torch.nn.utils.weight_norm(x)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

        # feature shallow extraction
        self.head = wn(nn.Conv2d(n_colors, n_feats, kernel_size, padding=kernel_size // 2))

        # feature deep extraction
        body = nn.ModuleList()
        rescale = []
        for i in range(self.blocks):
            body.append(Block(n_feats, kernel_size, wn, act))
            rescale.append(Scale(1))
        self.body = nn.Sequential(*body)
        self.rescale = nn.Sequential(*rescale)

        # feature confusion
        self.confusion = wn(nn.Conv2d(n_feats * self.blocks, n_feats, 1, 1, 0))

        # upsample and reconstruction
        self.tail = nn.Sequential(*[
            Upsampler(scale, n_feats, wn, act=None),
            wn(nn.Conv2d(n_feats, n_colors, kernel_size, padding=kernel_size // 2))
        ])

    def forward(self, x):
        x = self.sub_mean(x)
        res = x = self.head(x)
        blocks_out = []
        for i in range(self.blocks):
            x = self.rescale[i](self.body[i](x))
            blocks_out.append(x)
        x = self.confusion(torch.cat(blocks_out, 1))
        x += res
        x = self.tail(x)
        x = self.add_mean(x)

        return x

# from torchstat import stat
#
# net = Net()
# stat(net, (3, 10, 10))

