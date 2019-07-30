# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       model
   Project Name:    AE+aux_loss+5x3+ Pixel2pixel
   Author :         Hengrong LAN
   Date:            2019/1/22
   Device:          GTX1080Ti
-------------------------------------------------
   Change Activity:
                   2019/1/22:
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
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.conv3 = conv3x3(self.out_channels, self.in_channels)
        self.bn3 = nn.BatchNorm2d(self.in_channels)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.01)

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
        self.conv3 = conv3x3(1, self.in_channels)
        self.bn3 = nn.BatchNorm2d(self.in_channels)

    def forward(self, bfimg, x):
        x = self.upconv(x)
        bfimg = F.leaky_relu(self.bn3(self.conv3(bfimg)), 0.01)
        if self.merge_mode == 'add':
            x = x + bfimg

        else:
            #concat
            x = torch.cat((x, bfimg), 1)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        return x

# Model 1 modified AutoEncoder
class AE_al5long_bf(nn.Module):


    def __init__(self,  in_channels=3, up_mode='transpose'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(AE_al5long_bf, self).__init__()
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
        self.up4 = FinUp(32,1)

        #auxiliary loss
        self.flayer = Featurelayer(256,1)

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

        feature = self.flayer(out) #auxiliary feature


        out = self.up1(out)# 16, 16,128
        out = self.up2(out)# 32, 32,64
        out = self.up3(out)# 64, 64,32
        out = self.up4(bfimg,out)# 128, 128,1
        return out,feature

#  Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 64, normalization=False),
            *discriminator_block(64, 128), 
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )  

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
                  
    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

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
    model = AutoEncoder(in_channels=1).to(device)
    out = model(x)
    loss = torch.mean(out)

    loss.backward()

    print(loss)
