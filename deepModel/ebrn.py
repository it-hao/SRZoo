import torch
import torch.nn as nn

def make_model(args, parent=False):
    return EBRN(args)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * 255. * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BRM(nn.Module):
    def __init__(self, channels, scale, bp = True):
        super(BRM, self).__init__()
        self. bp = bp
        kernel_size, stride, padding = {
            2: (6, 2, 2),
            4: (8, 4, 2),
            8: (12, 8, 2)
        }[scale]
        # 先进行上采样
        self.up = nn.ConvTranspose2d(channels, channels, kernel_size, stride=stride, padding=padding)
        # 上采样特征进行重建
        self.sr_flow = nn.Sequential(*[
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
            nn.PReLU(channels)
        ])
        # 再进行下采样
        self.down = nn.Conv2d(channels, channels, kernel_size, stride = stride, padding = padding)
        # 残差进行重建
        self.bp_flow = nn.Sequential(*[
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
            nn.PReLU(channels)
        ])
    def forward(self, x):
        up = self.up(x)
        ox = self.sr_flow(up)
        # 最后一层并没有back-projection flow
        if self.bp:
            down = self.down(up)
            sub = x - down
            ix = self.bp_flow(sub)
            ix += sub
            return ix, ox
        return ox
 
class EBRN(nn.Module):
    def __init__(self):
        super(EBRN, self).__init__()
        self.n_brms = 10
        n_colors = 3
        n_feats = 64
        kernel_size = 3
        scale = 4

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

        # feature shallow extraction
        self.head = nn.Sequential(*[
            nn.Conv2d(n_colors, n_feats * 4, kernel_size, padding=kernel_size // 2),
            nn.PReLU(n_feats * 4),
            nn.Conv2d(n_feats * 4, n_feats, kernel_size, padding=kernel_size // 2),
            nn.PReLU(n_feats),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2),
            nn.PReLU(n_feats)
        ])  
        
        # convolutional layer after fusion
        self.convs = nn.ModuleList()
        for i in range(self.n_brms - 1):
            self.convs.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding = kernel_size//2))

        # embedded block residual learning
        self.brms = nn.ModuleList()
        for i in range(self.n_brms - 1):
            self.brms.append(BRM(n_feats, scale, True))
        self.brms.append(BRM(n_feats, scale, False))

        # reconstruction
        self.tail = nn.Conv2d(n_feats * self.n_brms, n_colors, kernel_size, padding = kernel_size//2)
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        out = []
        sr_sets = []
        # 前面的self.n_brms-1层
        for i in range(self.n_brms - 1):
            x, sr = self.brms[i](x)
            sr_sets.append(sr)
        # 最后的第n_brms层   
        sr = self.brms[self.n_brms - 1](x)
        out.append(sr)

        for i in range(self.n_brms - 1):
            sr = sr + sr_sets[self.n_brms - i - 2]
            sr = self.convs[i](sr)
            out.append(sr)
        x = self.tail(torch.cat(out, dim = 1))
        return x

# from torchstat import stat
# net = EBRN()
# stat(net, (3, 10, 10))
# Total params: 7,632,143
