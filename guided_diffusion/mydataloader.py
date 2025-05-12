import os
import sys
sys.path.append("")
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
# import torchvision.transforms.functional as F
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate

CLASSES = ['NT', 'NB', 'bg']


class MyDataset(Dataset):
    
    def __init__(self, args, data_path , transform = None, mode = 'training',plane = False):
        df = pd.read_csv(os.path.join(data_path, 'mydata_' + mode + '_groundtruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.args = args


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')  # L是灰度图像


        if self.transform:
            # state = torch.get_rng_state()
            # img = self.transform(img)
            # torch.set_rng_state(state)
            # mask = self.transform(mask)
            img, mask = self.transform(img, mask)
        
        # mask = torch.tensor(np.array(mask), dtype=torch.long)
        # unique_values = torch.unique(mask)
        # print("mask唯一值:", unique_values)
        # print(mask)
        # print(mask.shape)
        mask = self.load_mask(mask)
        # img_test = torch.rand([3, 256, 256])
        # mask_test = torch.rand([3, 256, 256])
        return (img, mask, name)
    
    def load_mask(self, mask):
        # print("---------------load_mask-----------------")
        # print(mask.shape)
        mask = mask.long()
        one_hot = F.one_hot(mask, num_classes=len(CLASSES)).float()  # [H, W, C]
        one_hot = one_hot.squeeze(0)
        return_mask = one_hot.permute(2, 0,1)
        # test_mask = torch.rand([3, 256, 256])
        return return_mask # [C, H, W]