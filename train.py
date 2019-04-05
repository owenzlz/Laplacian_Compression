import argparse
import os
from math import log10
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from vis_tools import *
from data_utils import *
from loss import GeneratorLoss
from model import Generator, Discriminator
import architecture as arch
import pdb 
import torch.nn.functional as F

gpu_id = 0
port_num = 8091
display = visualizer(port=port_num)
report_feq = 10
NUM_EPOCHS = 40

netG = arch.RRDB_Net(4, 3, 64, 12, gc=32, upscale=1, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
netD = Discriminator()

generator_criterion = GeneratorLoss()

if torch.cuda.is_available():
    netG.to(gpu_id)
    netD.to(gpu_id)
    generator_criterion.to(gpu_id)

optimizerG = optim.Adam(netG.parameters(), lr=0.0002)
optimizerD = optim.Adam(netD.parameters(), lr=0.0002)

train_set = MyDataLoader(hr_dir='../data/train_sample/HR/', hr_sample_dir='../data/train_sample/HR_Sample/4/', lap_dir='../data/train_sample/LAP_HR_Norm/')
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=2, shuffle=True)

step = 0
for epoch in range(1, NUM_EPOCHS):
    netG.train()
    netD.train()
    for hr, hr_sample, lap in train_loader:
        hr, hr_sample, lap = hr.to(gpu_id), hr_sample.to(gpu_id), lap.to(gpu_id)
        
        input = torch.cat([hr_sample, lap], 1)
        fake_img = netG(input)
        fake_img = (F.tanh(fake_img) + 1) / 2
        
        real_img = hr
        
        netD.zero_grad()
        real_out = netD(real_img).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        optimizerD.step()
    
        ############################
        netG.zero_grad()
        g_loss = generator_criterion(fake_out, fake_img, real_img)
        g_loss.backward()
        optimizerG.step()
        fake_img = netG(input)
        fake_img = (F.tanh(fake_img) + 1) / 2
        
        step += 1
        
        if step % report_feq == 0:
            vis_high = (real_img*255)[0].detach().cpu().data.numpy()
            vis_laplacian = (lap*255)[0].detach().cpu().data.numpy()
            vis_low = (hr_sample*255)[0].detach().cpu().data.numpy()
            vis_recon = (fake_img*255)[0].detach().cpu().data.numpy()

            display.plot_img_255(vis_high, win=1, caption='high')
            display.plot_img_255(vis_laplacian,  win=2, caption='laplacian')
            display.plot_img_255(vis_low,  win=3, caption='low')
            display.plot_img_255(vis_recon,  win=4, caption='sr')
            
        print(epoch, step)

        ########## Save Models ##########
        if step % 5000 == 0:
            if not os.path.exists('models'): os.mkdir('models')
            torch.save(netG, 'models/G_'+str(step)+'.pt')
            torch.save(netD, 'models/D_'+str(step)+'.pt')


    