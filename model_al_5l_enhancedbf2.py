# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       model
   Project Name:    AE+aux_loss+5x3+ BF path enhanced2
   Author :         Hengrong LAN
   Date:            2019/2/26
   Device:          GTX1080Ti
-------------------------------------------------
   Change Activity:
                   2019/2/26:
-------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np




def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def conv5x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=(5,3),
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)



class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv5x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv5x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)


        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)

        if self.pooling:
            x = self.pool(x)
        return x

class Bottom(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Bottom, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = conv5x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels,
                               kernel_size=(20, 3), stride=(20, 1),padding=(3,1))
        #self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.conv3 = conv3x3(self.out_channels, self.in_channels)
        self.bn2 = nn.BatchNorm2d(self.in_channels)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.01)

        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, 
                  up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.in_channels,
            mode=self.up_mode)
        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)


    def forward(self, x):
        x = self.upconv(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        return x

class Featurelayer(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Featurelayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        return x

class Inception_block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Inception_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1= conv1x1(self.in_channels,64)
        #self.bn1 = nn.BatchNorm2d(64)

        self.conv2= conv1x1(self.in_channels,32)
        #self.bn2 = nn.BatchNorm2d(32)
        self.conv3= conv3x3(32,64)
        #self.bn3 = nn.BatchNorm2d(64)


        self.conv4= conv1x1(self.in_channels,32)
        #self.bn4 = nn.BatchNorm2d(32)
        self.conv5= conv3x3(32,64)
        #self.bn5 = nn.BatchNorm2d(64)
        self.conv6= conv3x3(64,128)
        #self.bn6 = nn.BatchNorm2d(128)

        self.conv7= conv3x3(256,self.out_channels)
        self.bn7 = nn.BatchNorm2d(self.out_channels)
    def forward(self,x):
        x1=self.conv1(x)

        x2=self.conv2(x)
        x2=self.conv3(x2)

        x3=self.conv4(x)
        x3=self.conv5(x3)
        x3=self.conv6(x3)

        x4 = torch.cat((x1, x2, x3), 1)
        x4 = F.leaky_relu(self.bn7(self.conv7(x4)), 0.01)

        return x4


        

class FinUp(nn.Module):

    def __init__(self, in_channels, out_channels, 
                  up_mode='transpose', merge_mode='concat'):
        super(FinUp, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up_mode = up_mode
        self.merge_mode = merge_mode

        self.upconv = upconv2x2(self.in_channels, self.in_channels,
            mode=self.up_mode)
        if self.merge_mode == 'add':
            self.conv1 = conv3x3(
                self.in_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same,concat
            self.conv1 = conv3x3(2*self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.conv3 = conv3x3(self.out_channels, self.out_channels)
        self.bn3 = nn.BatchNorm2d(self.out_channels)

    def forward(self, bfimg, x):
        x = self.upconv(x)
        if self.merge_mode == 'add':
            x = x + bfimg

        else:
            #concat
            x = torch.cat((x, bfimg), 1)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.01)
        return x

class FinConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FinConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.conv3 = conv3x3(self.out_channels, self.out_channels)
        self.bn3 = nn.BatchNorm2d(self.out_channels)

    def forward(self,  x):        
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.01)
        return x

# Model 1 modified AutoEncoder
class AE_al5long_enhancedbf(nn.Module):


    def __init__(self,  in_channels=3, up_mode='transpose'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(AE_al5long_enhancedbf, self).__init__()
        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.in_channels = in_channels

        self.inputlayer = nn.Sequential(
                            nn.Conv2d(1,1, kernel_size=(3, 1), stride=1, padding=(1,4)),
                            nn.BatchNorm2d(1),
                            nn.LeakyReLU(0.1))
        self.down1 = DownConv(1,32)
        self.down2 = DownConv(32,64)
        self.down3 = DownConv(64,128)
        self.down4 = DownConv(128,256)
        self.bottom = Bottom(256,512)
        self.up1 = UpConv(256,128)
        self.up2 = UpConv(128,64)
        self.up3 = UpConv(64,32)
        self.bfpath1 = Inception_block(1, 64)
        self.bfpath2 = Inception_block(64, 128)
        self.bfpath3 = Inception_block(128, 32)
        self.up4 = FinUp(32,64)
        self.final1 = FinConv(64,128)
        self.final2 = FinConv(128,1)


        #auxiliary loss
        self.flayer1 = Featurelayer(256,1)
        self.flayer2 = Featurelayer(32,1)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x,bfimg):
        out = self.inputlayer(x) # 2560, 128,1

        out = self.down1(out) # 1280, 64,32
        out = self.down2(out) # 640, 32,64
        out = self.down3(out) # 320, 16,128
        out = self.down4(out) # 160, 8,256
        out = self.bottom(out) # 8, 8, 256

        feature = self.flayer1(out) #auxiliary feature


        out = self.up1(out)# 16, 16,128
        out = self.up2(out)# 32, 32,64
        out = self.up3(out)# 64, 64,32
        bfimg = self.bfpath1(bfimg)
        bfimg = self.bfpath2(bfimg)
        bfimg = self.bfpath3(bfimg)
        bf_feature = self.flayer2(bfimg)

        out = self.up4(bfimg,out)# 128, 128,1
        out = self.final1(out)
        out = self.final2(out)

        return out,feature,bf_feature

if __name__ == "__main__":
    """
    testing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device =  torch.device('cuda:1')
    time_step = 6
    hidden_size = 4
    _lambda = 0.5
    batch = 2
    tau = 0.5


    grad_result = []
    grads = {}




    x = Variable(torch.FloatTensor(np.random.random((1, 1, 2560, 120))),requires_grad = True).to(device)
    bf = Variable(torch.FloatTensor(np.random.random((1, 1, 128, 128))),requires_grad = True).to(device)
    model = AE_al5long_enhancedbf(in_channels=1).to(device)
    out,f1,f2 = model(x,bf)
    #loss = torch.mean(out)

    #loss.backward()

    print(out)
    print(f1)
    print(f2)
