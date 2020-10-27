import torch.nn as nn
import torch
# from loss import HMLoss

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CB(nn.Module):
    def __init__(self, inp_ch, out_ch):
        super(CB, self).__init__()
        self.inp_ch = inp_ch
        self.out_ch = out_ch

        self.bn1 = nn.BatchNorm2d(inp_ch)
        self.conv1 = nn.Conv2d(inp_ch, out_ch//2, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(out_ch//2)
        self.conv2 = nn.Conv2d(out_ch//2, out_ch//2, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(out_ch//2)
        self.conv3 = nn.Conv2d(out_ch//2, out_ch, 1, 1, 0)
        self.relu = nn.ReLU()

    def forward(self, x):
        assert x.size()[1] == self.inp_ch, "{} {}".format(x.size()[1], self.inp_ch)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        return out

class Residual(nn.Module):
    def __init__(self, inp_ch, out_ch):
        super(Residual, self).__init__()
        self.inp_dim = inp_ch
        self.out_dim = out_ch
        self.CB = CB(inp_ch, out_ch)

        if inp_ch == out_ch:
            self.skip_layer = None
        else:
            self.skip_layer = nn.Conv2d(inp_ch, out_ch, 1)

    def forward(self, x):
        if self.skip_layer:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.CB(x)
        out += residual
        return residual


class Hourglass(nn.Module):
    def __init__(self, inp_ch=256, nReduction=4, nModule=2, poolK=(2,2), poolStride=(2,2), upsampleK=2):
        super(Hourglass, self).__init__()
        self.nReduction = nReduction

        # skip connection
        skip_list = []
        for _ in range(nModule):
            skip_list.append(Residual(inp_ch, inp_ch))
        self.skip = nn.Sequential(*skip_list)


        #max pooling
        self.mp = nn.MaxPool2d(poolK, poolStride)

        #after pool
        afterpool_list = []
        for _ in range(nModule):
            afterpool_list.append(Residual(inp_ch, inp_ch))
        self.afterpool = nn.Sequential(*afterpool_list)

        if (self.nReduction > 1):
            self.hg = Hourglass(inp_ch, self.nReduction-1, nModule, poolK, poolStride, upsampleK)
        else:
            midres_list = []
            for _ in range(nModule):
                midres_list.append(Residual(inp_ch, inp_ch))
            self.midres = nn.Sequential(*midres_list)

        lastres_list = []
        for _ in range(nModule):
            lastres_list.append(Residual(inp_ch, inp_ch))
        self.lastres = nn.Sequential(*lastres_list)

        self.up = nn.Upsample(scale_factor=upsampleK)
    def forward(self, x):
        out_skip = x
        out_skip = self.skip(out_skip)
        out_pool = x
        out_pool = self.mp(out_pool)
        out_pool = self.afterpool(out_pool)
        if self.nReduction>1:
            out_pool = self.hg(out_pool)
        else:
            out_pool = self.midres(out_pool)
        out_pool = self.lastres(out_pool)
        out_pool = self.up(out_pool)
        return out_skip+out_pool



class StackedHourglass(nn.Module):
    def __init__(self, cfg, nReduction=4, nModule=2, poolK=(2,2), poolStride=(2,2), upsampleK=2):
        super(StackedHourglass, self).__init__()
        self.nStack = cfg.MODEL.NUM_STACK
        self.inp_ch = cfg.MODEL.INPUT_CH
        self.nJoint = cfg.MODEL.NUM_JOINTS

        self.bn1 = nn.BatchNorm2d(self.inp_ch)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(self.inp_ch, 64, 7, 2, 3)

        self.res1 = Residual(64, 128)
        self.mp = nn.MaxPool2d(2, 2)
        self.res2 = Residual(128, 128)
        self.res3 = Residual(128, self.inp_ch)

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(inp_ch=self.inp_ch),
            ) for _ in range(self.nStack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(self.inp_ch, self.inp_ch),
                Conv(self.inp_ch, self.inp_ch, 1, bn=True, relu=True)
            ) for _ in range(self.nStack)])

        self.outs = nn.ModuleList([Conv(self.inp_ch, self.nJoint, 1, relu=False, bn=False) for i in range(self.nStack)])
        self.merge_features = nn.ModuleList([Conv(self.inp_ch, self.inp_ch, 1, relu=False, bn=False) for i in range(self.nStack - 1)])
        self.merge_preds = nn.ModuleList([Conv(self.nJoint, self.inp_ch, 1, relu=False, bn=False) for i in range(self.nStack - 1)])

        # self.HMloss = HMLoss()


    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.res1(x)
        x = self.mp(x)
        x = self.res2(x)
        x = self.res3(x)
        combined_hm_preds = []
        for i in range(self.nStack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nStack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return combined_hm_preds
        # return torch.stack(combined_hm_preds, 1)

    # def loss(self, hm_preds, hm):
    #     stacked_loss = []
    #     for i in range(self.nStack):
    #         stacked_loss.append(self.HMloss(hm_preds[i], hm))
    #     stacked_loss = torch.stack(stacked_loss, dim=1)
    #     return stacked_loss



if __name__ == '__main__':
    print("start")
