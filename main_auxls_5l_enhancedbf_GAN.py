# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       main
   Project Name:    GAN+aux_loss+5x3+ Pixel2pixel
   Author :         Hengrong LAN
   Date:            2019/3/1
   Device:          GTX1080Ti
-------------------------------------------------
   Change Activity:
                   2019/3/15:
-------------------------------------------------
"""
from model_al_5l_enhancedbf2 import AE_al5long_enhancedbf
from model_al_5l_bf_GAN import Discriminator
from visualizer import Visualizer
from skimage.measure import compare_ssim, compare_psnr
from torch.autograd import Variable
import thindataset
import os
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_pathr = './20181220/'
learning_rate=0.005
batch_size = 32
test_batch = 32
start_epoch=0
loadcp = False
lam_=0.5
lam_2=0.5
lamd_p =25
img_height = 128
img_width = 128
num_epochs = 600

def calc_confidence_interval(samples, confidence_value=0.95):
    # samples should be a numpy array
    if type(samples) is list:
        samples = np.asarray(samples)
    assert isinstance(samples, np.ndarray), 'args: samples {} should be np.array'.format(samples)
    # print('Results List:', samples)
    stat_accu = st.t.interval(confidence_value, len(samples)-1, loc=np.mean(samples), scale=st.sem(samples))
    center = (stat_accu[0]+stat_accu[1])/2
    deviation = (stat_accu[1]-stat_accu[0])/2
    return center, deviation
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, num_top = 10):
        self.reset()
        _array = np.zeros(shape=(num_top)) + 0.01
        self.top_list = _array.tolist()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def top_update_calc(self, val):
        # update the lowest or NOT
        if val > self.top_list[0]:
            self.top_list[0] = val
            # [lowest, ..., highest]
            self.top_list.sort()
        # update mean, deviation
        mean, deviation = calc_confidence_interval(self.top_list)
        best = self.top_list[-1]
        return mean, deviation, best

losses_G = AverageMeter()
losses_D = AverageMeter()
batch_time = AverageMeter()
data_time = AverageMeter()
train_ssim_meter = AverageMeter()
train_psnr_meter = AverageMeter()
train_ssim_top20 = AverageMeter(num_top=20)
train_psnr_top20 = AverageMeter(num_top=20)
test_ssim_meter = AverageMeter()
test_psnr_meter = AverageMeter()
test_ssim_top10 = AverageMeter(num_top=10)
test_psnr_top10 = AverageMeter(num_top=10)
losses_pixel= AverageMeter()




curr_lr_G = learning_rate
curr_lr_D = learning_rate/5

#vis = Visualizer(env='GAN_25_aelam1_5long_enhancedbf')
vis = Visualizer(env='main')

#source activate pytorch

train_dataset = thindataset.ReconDataset(dataset_pathr,train=True)
test_dataset = thindataset.ReconDataset(dataset_pathr,train=False)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True)
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch,
        shuffle=True)



model_G = AE_al5long_enhancedbf(in_channels=1)
model_G = nn.DataParallel(model_G)
model_D = Discriminator()
model_D = nn.DataParallel(model_D)

criterion_GAN = nn.MSELoss()
criterion_pixelwise = nn.L1Loss()

model_G = model_G.to(device)
model_D = model_D.to(device)
criterion_GAN = criterion_GAN.to(device)
criterion_pixelwise = criterion_pixelwise.to(device)

#criterion = nn.MSELoss()
optimizer_G = torch.optim.Adam(model_G.parameters(),lr=curr_lr_G)
optimizer_D = torch.optim.Adam(model_D.parameters(),lr=curr_lr_D)

if loadcp:
   checkpoint = torch.load('reconstruction_Unet_2200.ckpt')
   model.load_state_dict(checkpoint['state_dict'])
   start_epoch=checkpoint['epoch']-1
   curr_lr = checkpoint['curr_lr']
   optimizer.load_state_dict(checkpoint['optimizer'])



# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def reset_grad():
    optimizer_G.zero_grad()
    optimizer_D.zero_grad()

cudnn.benchmark = True


total_step = len(train_loader)
test_total_step = len(test_loader)

epoch = start_epoch
print("start")
print('train_data :{}'.format(train_dataset.__len__()))
print('test_data :{}'.format(test_dataset.__len__()))
end = time.time()
patch = (1, img_height // 2 ** 3, img_width // 2 ** 3)
for epoch in range(num_epochs):
        for batch_idx, (rawdata ,reimage, bfimg) in enumerate(train_loader):
                rawdata = rawdata.to(device)
                reimage = reimage.to(device)
                bfimg = bfimg.to(device)

                # Adversarial ground truths
                valid = Variable(torch.cuda.FloatTensor(np.ones((rawdata.size(0), *patch))), requires_grad=False)
                fake = Variable(torch.cuda.FloatTensor(np.zeros((rawdata.size(0), *patch))), requires_grad=False)
                # ================================================================== #
                #                      Train the Generators                          #
                # ================================================================== #

                fake_img,ae_feature,bf_feature = model_G(rawdata,bfimg)
                pred_fake = model_D(fake_img, bfimg)
                # loss
                ae_loss = F.mse_loss(fake_img, reimage)
                
                reimage_resize = F.upsample(reimage, (8, 8), mode='bilinear')
                aux_loss = F.mse_loss(ae_feature, reimage_resize)
                
                bf_loss = F.mse_loss(bf_feature,reimage)
                loss_pixel = ae_loss+ lam_*aux_loss+lam_2*bf_loss
                loss_GAN = criterion_GAN(pred_fake, valid)
                loss_G = loss_GAN + lamd_p * loss_pixel
                
                # Backward and optimize
                reset_grad()
                loss_G.backward()
                optimizer_G.step()

                losses_pixel.update(loss_pixel.item(), rawdata.size(0))                
                losses_G.update(loss_G.item(), rawdata.size(0))

                if (batch_idx) % 1 ==0:
                # ================================================================== #
                #                      Train the Discriminator                       #
                # ================================================================== #
                # Fake loss          
                   pred_fake = model_D(fake_img.detach(), bfimg)      
                   loss_fake = criterion_GAN(pred_fake, fake)
                
                # Real loss
                   pred_real = model_D(reimage, bfimg)
                   loss_real = criterion_GAN(pred_real, valid)
                
                   loss_D = 0.5*(loss_real + loss_fake)
                # Backward and optimize
                   reset_grad()
                   loss_D.backward()
                   optimizer_D.step()
                
                   losses_D.update(loss_D.item(), rawdata.size(0))
                
                ssim = compare_ssim(np.array(reimage.detach().squeeze()), np.array(fake_img.detach().squeeze()))
                train_ssim_meter.update(ssim)
                psnr = compare_psnr(np.array(reimage.detach().squeeze()), np.array(fake_img.detach().squeeze()), data_range=255)
                train_psnr_meter.update(psnr)
                
                # visualization and evaluation
                if (batch_idx +1) %8 ==0:
                   out_image= fake_img.detach()
                   bfim=bfimg.detach()
                   bf_f =bf_feature.detach()
                   #vis_index= random.randint(0,reimage.size(0)-1)
                   vis.img(name='ground truth',img_=10*reimage[0])
                   vis.img(name='beamform', img_=20*bfim[0])
                   vis.img(name='feature', img_=10*bf_f[0])
                   vis.img(name='reconstruction', img_=10*out_image[0])

                batch_time.update(time.time() - end)
                end = time.time()

                if (batch_idx + 1) % 20 == 0:
                        print('Epoch [{}], Start [{}], Step [{}/{}], Loss_G: {:.4f}, Loss_D: {:.4f},Time [{batch_time.val:.3f}({batch_time.avg:.3f})]'
                              .format(epoch + 1, start_epoch, batch_idx + 1, total_step, loss_G.item(), loss_D.item(),batch_time=batch_time))

        vis.plot_multi_win(
            dict(
                loss_G_val=losses_G.val,
                loss_G_avg=losses_G.avg)) 

        vis.plot_multi_win(
            dict(
                loss_D_val=losses_D.val,
                loss_D_avg=losses_D.avg)) 

        vis.plot_single_win(
            dict(
                pixel_loss=losses_pixel.val), win='pixel_loss')
        
        vis.plot_multi_win(dict(
                                train_ssim=train_ssim_meter.avg, 
                                train_psnr=train_psnr_meter.avg))  
        mean, deviation, best = train_ssim_top20.top_update_calc(train_ssim_meter.avg)
        vis.plot_single_win(dict(mean=mean, deviation=deviation, best=best), win='train_ssim_')
        
        mean, deviation, best = train_psnr_top20.top_update_calc(train_psnr_meter.avg)
        vis.plot_single_win(dict(mean=mean, deviation=deviation, best=best), win='train_psnr_')
        
        # Validata
        if (epoch + 1) % 8 == 0:
            with torch.no_grad():
               for batch_idx, (rawdata ,reimage, bfimg) in enumerate(test_loader):
                   rawdata = rawdata.to(device)
                   reimage = reimage.to(device)
                   bfimg = bfimg.to(device)
                   outputs, _, bf_feature = model_G(rawdata, bfimg)   
                   ssim = compare_ssim(np.array(reimage.squeeze()), np.array(outputs.squeeze()))
                   test_ssim_meter.update(ssim)
                   psnr = compare_psnr(np.array(reimage.squeeze()), np.array(outputs.squeeze()), data_range=255)
                   test_psnr_meter.update(psnr) 
                   if (batch_idx +1) % 8 ==0:
                      out_image= outputs.detach()
                      bfim=bfimg.detach()
                      bf_f =bf_feature.detach()
                      #vis_index= random.randint(0,reimage.size(0)-1)
                      vis.img(name='Test: ground truth',img_=10*reimage[0])
                      vis.img(name='Test: beamform', img_=20*bfim[0])
                      vis.img(name='Test: feature', img_=10*bf_f[0])
                      vis.img(name='Test: reconstruction', img_=10*out_image[0]) 
               
               vis.plot_multi_win(dict(
                                      test_ssim=test_ssim_meter.avg, 
                                      test_psnr=test_psnr_meter.avg))  
               mean, deviation, best = test_ssim_top10.top_update_calc(test_ssim_meter.avg)
               vis.plot_single_win(dict(mean=mean, deviation=deviation, best=best), win='test_ssim_')
        
               mean, deviation, best = test_psnr_top10.top_update_calc(test_psnr_meter.avg)
               vis.plot_single_win(dict(mean=mean, deviation=deviation, best=best), win='test_psnr_')
         
        # Decay learning rate
        if (epoch + 1) % 50 == 0:
                curr_lr_G /= 5
                curr_lr_D /= 5
                update_lr(optimizer_G, curr_lr_G)
                update_lr(optimizer_D, curr_lr_D)

        if (epoch+1) % 100 ==0:
                torch.save({'epoch': epoch + 1,
                            'state_dict_G':model_G.state_dict(),
                            'state_dict_D':model_D.state_dict(),
                            'optimizer_G': optimizer_G.state_dict(),
                            'optimizer_D': optimizer_D.state_dict(),
                            'curr_lr_G': curr_lr_G,
                            'curr_lr_D': curr_lr_D
                           },
                          './checkpoint/25GAN_aelam1_5long_enhancedbf_{}.ckpt'
                           .format(epoch + 1))
                print('Save ckpt successfully!')
        #epoch=epoch+1

