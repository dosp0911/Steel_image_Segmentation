#!/usr/bin/env python
# coding: utf-8

# In[2]:


from torch.utils.data.dataset import Dataset
import torch
import pathlib

import matplotlib.pyplot as plt
import numpy as np

# import import_ipynb
from util import csv_file_load
from pre_processing import Pre_process_img as p


# In[4]:


class Steel_dataset(Dataset):
    # out_size =  the size of output of the final layer for mask
    def __init__(self, img_f_path, dataframe, out_size=(132, 1476)):
        super(Steel_dataset, self).__init__()
        
        if isinstance(img_f_path, str):
            self.img_f_path = pathlib.Path(img_f_path)
        else:
            self.img_f_path = img_f_path
        self.dataframe = dataframe
        self.out_size = out_size
        
    def __getitem__(self, index):
        img_info = self.dataframe.iloc[index]
        name, i_class, encoded_p = img_info[0], img_info[1], img_info[2] 
        
        # img read and standardize
        img_arr = plt.imread(str(self.img_f_path / name))
        # convert rgv to grayscale
        img_arr = p().rgb_to_gray(img_arr, new_axis=True)
        
        # decode rle 1d into 2d tensor with maksing class number
        mask = p().decode_pixels_to_mask(size=np.shape(img_arr), encoded_p=encoded_p, mask_val=int(i_class))
        mask = p().crop_img(mask, self.out_size[0], self.out_size[1])
        mask = torch.from_numpy(mask)
        
        #overlap-tile strategy
        img_arr = p().overlap_tile(img_arr, 30, 30) 
        img_arr = torch.from_numpy(img_arr) / 255.0
        #convert img shape into(C,H,W)
        img_arr = img_arr.permute(2,0,1)
        
        return img_arr, mask
    
    def __len__(self):
        return len(self.dataframe)


# In[2]:


# !jupyter nbconvert --to script steel_dataset.ipynb

