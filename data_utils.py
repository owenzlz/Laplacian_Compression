from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import glob
import pdb

def toTensor_transform():
    return Compose([
        ToTensor()
    ])

class MyDataLoader(Dataset):
    def __init__(self, hr_dir='../data/train_sample/HR/', hr_sample_dir='../data/train_sample/HR_Sample/4/', lap_dir='../data/train_sample/LAP_HR_Norm/'):
        super(MyDataLoader, self).__init__()

        n_imgs = 0
        for file in glob.glob(str(hr_dir)+'*.png'):
            n_imgs += 1

        hr_list = []; hr_sample_list = []; lap_list = []
        for i in range(n_imgs):
            hr_list.append(str(hr_dir)+str(i)+'.png')
            hr_sample_list.append(str(hr_sample_dir)+str(i)+'.png')
            lap_list.append(str(lap_dir)+str(i)+'.jpg')

        self.transform = toTensor_transform()
        self.hr_list = hr_list
        self.hr_sample_list = hr_sample_list
        self.lap_list = lap_list

    def __getitem__(self, idx):
        hr = self.transform(Image.open(self.hr_list[idx]))
        hr_sample = self.transform(Image.open(self.hr_sample_list[idx]))
        lap = self.transform(Image.open(self.lap_list[idx]))
        return hr, hr_sample, lap

    def __len__(self):
        return len(self.hr_list)

'''
train_set = MyDataLoader()
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=100, shuffle=True)
for idx, (hr, hr_sample, lap) in enumerate(train_loader):
    print(idx, hr.shape, hr_sample.shape, lap.shape)
    pdb.set_trace()
'''












