import torch 
import torch.nn as nn
import torch.nn.functional as F
# 原文 U-net: Convolutional networks for biomedical image segmentation  2015

class double_conv(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )
    def forward(self,x):
        x = self.max_pool_conv(x)
        return x
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2,in_ch//2,2,stride=2)
        self.conv = double_conv(in_ch, out_ch)
    def forward(self,x1,x2):
        x1 = self.up(x1)
        diffX = x1.size()[2]-x2.size()[2]
        diffY = x1.size()[3]-x2.size()[3]
        x2 = F.pad(x2, (diffX//2, int(diffX/2), diffY//2, int(diffY/2)))
        x = torch.cat([x2,x1],dim=1)
        x = self.conv(x)
        return x
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self,x):
        x = self.conv(x)
        return x

class ChannelAttention_ave(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAttention_ave, self).__init__()
        self.convq = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)  # 深度可分离卷积
        self.convk = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)  # 深度可分离卷积
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):
        B = self.convq(x)
        C = self.convk(x)
        D = x
        b,c,h,w = D.size()
        S = self.softmax(torch.matmul(B.view(b,c,h*w), C.view(b,c,h*w).transpose(1,2)))
        E = torch.matmul(S, D.view(b,c,h*w)).view(b,c,h,w)
        E = E+x
        return E

class PatchMerging(nn.Module):
    def __init__(self):
        super(PatchMerging, self).__init__()
    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        x0 = x[:, :, 0::2, 0::2]  # [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]  # [B, C, H/2, W/2]
        x2 = x[:, :, 0::2, 1::2]  # [B, C, H/2, W/2]
        x3 = x[:, :, 1::2, 1::2]  # [B, C, H/2, W/2]
        x = torch.cat([x0, x1, x2, x3], 1)  # [B, 4*C, H/2, W/2]
        return x

class Patch2Image(nn.Module):
    def __init__(self):
        super(Patch2Image, self).__init__()
    def forward(self, x):
        device = torch.device('cuda:0')
        B, C, H, W = x.shape
        C = C//4
        xr = torch.zeros(B, C, H*2, W*2).to(device)
        xr[:,:,0::2, 0::2] = x[:,0:C:,:,:]
        xr[:,:,1::2, 0::2] = x[:,C:2*C:,:,:]
        xr[:,:,0::2, 1::2] = x[:,2*C:3*C:,:,:]
        xr[:,:,1::2, 1::2] = x[:,3*C:4*C:,:,:]
        return xr

class PMPA1_ave(nn.Module):     # 下采样4次   Q K 做3*3卷积
    def __init__(self, in_channels, out_channels):
        super(PMPA1_ave, self).__init__()
        self.pm = PatchMerging()
        self.p2i = Patch2Image()
        self.convq = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)  # 深度可分离卷积
        self.convk = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)  # 深度可分离卷积
        self.softmax = nn.Softmax(dim=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        B, C, H, W = x.shape 
        Q = self.convq(x)
        K = self.convk(x)
        Q = self.pm(Q)
        Q = self.pm(Q)
        Q = self.pm(Q)
        Q = self.pm(Q)
        K = self.pm(K)
        K = self.pm(K)
        K = self.pm(K)
        K = self.pm(K)
        V = self.pm(x)   # B, 4C, H/2, W/2
        V = self.pm(V)   # B, 16C, H/4, W/4
        V = self.pm(V)   # B, 64C, H/8, W/8
        V = self.pm(V)   # B, 256C, H/16, W/16
        b,c,h,w = V.size()
        S = self.softmax(torch.matmul(Q.view(b,c,h*w).transpose(1,2), K.view(b,c,h*w)))
        E = torch.matmul(V.view(b,c,h*w), S.transpose(1,2)).view(b,c,h,w)
        E = self.p2i(E)  # B, 64C, H/8, W/8
        E = self.p2i(E)  # B, 16C, H/4, W/4
        E = self.p2i(E)  # B, 4C, H/2, W/2
        E = self.p2i(E)  # B, C, H, W
        E = E+x   # B, C, H, W
        x = self.conv(E)
        return x

class PMPA2_ave(nn.Module):     # 下采样3次
    def __init__(self, in_channels, out_channels):
        super(PMPA2_ave, self).__init__()
        self.pm = PatchMerging()
        self.p2i = Patch2Image()
        self.convq = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)  # 深度可分离卷积
        self.convk = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)  # 深度可分离卷积
        self.softmax = nn.Softmax(dim=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        B, C, H, W = x.shape 
        Q = self.convq(x)
        K = self.convk(x)
        Q = self.pm(Q)
        Q = self.pm(Q)
        Q = self.pm(Q)
        K = self.pm(K)
        K = self.pm(K)
        K = self.pm(K)
        V = self.pm(x)   # B, 4C, H/2, W/2
        V = self.pm(V)   # B, 16C, H/4, W/4
        V = self.pm(V)   # B, 64C, H/8, W/8
        b,c,h,w = V.size()
        S = self.softmax(torch.matmul(Q.view(b,c,h*w).transpose(1,2), K.view(b,c,h*w)))
        E = torch.matmul(V.view(b,c,h*w), S.transpose(1,2)).view(b,c,h,w)
        E = self.p2i(E)  # B, 16C, H/4, W/4
        E = self.p2i(E)  # B, 4C, H/2, W/2
        E = self.p2i(E)  # B, C, H, W
        E = E+x   # B, C, H, W
        x = self.conv(E)
        return x


class PMPA3_ave(nn.Module):     # 下采样2次
    def __init__(self, in_channels, out_channels):
        super(PMPA3_ave, self).__init__()
        self.pm = PatchMerging()
        self.p2i = Patch2Image()
        self.convq = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)
        self.convk = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)
        self.softmax = nn.Softmax(dim=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        B, C, H, W = x.shape 
        Q = self.convq(x)
        K = self.convk(x)
        Q = self.pm(Q)
        Q = self.pm(Q)
        K = self.pm(K)
        K = self.pm(K)
        V = self.pm(x)   # B, 4C, H/2, W/2
        V = self.pm(V)   # B, 16C, H/4, W/4
        b,c,h,w = V.size()
        S = self.softmax(torch.matmul(Q.view(b,c,h*w).transpose(1,2), K.view(b,c,h*w)))
        E = torch.matmul(V.view(b,c,h*w), S.transpose(1,2)).view(b,c,h,w)
        E = self.p2i(E)  # B, 4C, H/2, W/2
        E = self.p2i(E)  # B, C, H, W
        E = E+x   # B, C, H, W
        x = self.conv(E)
        return x

class PMPA4_ave(nn.Module):     # 下采样1次
    def __init__(self, in_channels, out_channels):
        super(PMPA4_ave, self).__init__()
        self.pm = PatchMerging()
        self.p2i = Patch2Image()
        self.convq = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)
        self.convk = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)
        self.softmax = nn.Softmax(dim=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        B, C, H, W = x.shape 
        Q = self.convq(x)
        K = self.convk(x)
        Q = self.pm(Q)
        K = self.pm(K)
        V = self.pm(x)   # B, 4C, H/2, W/2
        b,c,h,w = V.size()
        S = self.softmax(torch.matmul(Q.view(b,c,h*w).transpose(1,2), K.view(b,c,h*w)))
        E = torch.matmul(V.view(b,c,h*w), S.transpose(1,2)).view(b,c,h,w)
        E = self.p2i(E)  # B, C, H, W
        E = E+x   # B, C, H, W
        x = self.conv(E)
        return x

class Rcnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Rcnet,self).__init__()
        self.inc = double_conv(in_channels, 64)
        self.PMPA1 = PMPA1_ave(64,64)
        self.down1 = down(64,128)
        self.PMPA2 = PMPA2_ave(128,128)
        self.down2 = down(128,256)
        self.PMPA3 = PMPA3_ave(256,256)
        self.down3 = down(256,512)
        self.PMPA4 = PMPA4_ave(512,512)
        self.down4 = down(512,512)
        self.ca = ChannelAttention_ave(512,512)
        self.up1 = up(1024,256)
        self.up2 = up(512,128)
        self.up3 = up(256,64)
        self.up4 = up(128,64)
        self.outc = outconv(64,out_channels)
    def forward(self, x):
        x1 = self.inc(x)
        x1 = x1+self.PMPA1(x1)  # 位置注意力
        x2 = self.down1(x1)
        x2 = x2+self.PMPA2(x2)
        x3 = self.down2(x2)
        x3 = x3+self.PMPA3(x3)
        x4 = self.down3(x3)
        x4 = x4+self.PMPA4(x4)
        x5 = self.down4(x4)
        x5 = self.ca(x5)   # 通道注意力
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        x = self.outc(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda:0')
    x = torch.rand(8,1,256,512)
    x = x.to(device)
    net = Rcnet(in_channels=1, out_channels=1).to(device)
    out = net(x)
    print(out.is_cuda)
    print(out.shape)

