from skimage.io import imsave
import argparse
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils import data
from model import Generator
import pdb
from torch.nn import functional as F
from data_utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import os

if not os.path.exists('data/hr'): os.mkdir('data/hr')
if not os.path.exists('data/hr_sample'): os.mkdir('data/hr_sample')
if not os.path.exists('data/sr'): os.mkdir('data/sr')
if not os.path.exists('data/lap'): os.mkdir('data/lap')

gpu_id = 2
CROP_SIZE = 128
UPSCALE_FACTOR = 2
TEST_MODE = True 
# IMAGE_NAME = opt.image_name

MODEL_NAME = 'G_200000.pt'
netG = torch.load('models/' + MODEL_NAME).to(gpu_id)

test_set = MyDataLoader(hr_dir='../data/test_sample/HR/', hr_sample_dir='../data/test_sample/HR_Sample/4/', lap_dir='../data/test_sample/LAP_HR/')
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=True)

cnt = 0
for hr, hr_sample, lap in test_loader:
	hr, hr_sample, lap = hr.to(gpu_id), hr_sample.to(gpu_id), lap.to(gpu_id)

	input = torch.cat([hr_sample, lap], 1)
	fake_img = netG(input)
	fake_img = (F.tanh(fake_img) + 1) / 2
	real_img = hr
	
	lap_np = lap.cpu().data.transpose(1,3).transpose(1,2).squeeze(3).numpy()[0]

	# imsave('SR_results/'+str(cnt)+'.jpg', fake_img_np)
	save_image(fake_img, 'data/sr/'+str(cnt)+'.png', nrow=1, padding=0)
	save_image(hr, 'data/hr/'+str(cnt)+'.png', nrow=1, padding=0)
	save_image(hr_sample, 'data/hr_sample/'+str(cnt)+'.png', nrow=1, padding=0)
	imsave('data/lap/'+str(cnt)+'.png',lap_np)

	cnt += 1

	print(cnt)


