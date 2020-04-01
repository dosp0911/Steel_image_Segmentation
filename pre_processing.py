#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict

import pathlib
import math


# In[16]:


class Pre_process_img():
    def __init__(self):
        return 
        
    def standardize(self, img_arr):
        if not isinstance(img_arr, np.array):
            raise TypeError('img_arr must be np.array type')
        return img_arr / 255.0
        
        # h, w는 한쪽 부분에서 늘릴 길이
    # data = [height, width, n_channels]
    def overlap_tile(self, data, h, w):
        if not np.ndim(data) == 3:
            raise ValueError('dimension of the data have to be 3. [height, width, n_channels]')
        old_h, old_w, _ = np.shape(data)
        tmp_ = np.concatenate((data[:,(w-1)::-1,:], data, data[:,:(old_w-w-1):-1,:]), axis=1) # mirror 반사 모양으로 width 늘리기 
        new_ = np.concatenate((tmp_[(h-1)::-1,:,:], tmp_, tmp_[:(old_h-h-1):-1,:,:]), axis=0) # mirror 반사 모양으로 height 늘리기
        return new_

        # size = (height,width) of img, 
    # return img with segmentation, 2d coordinates list
    def decode_pixels_to_mask(self, size, encoded_p, mask_val=1):
        mask_val = int(mask_val)
        h, w, _ = size
        p = np.array([int(m) for m in encoded_p.split(" ")])
        mask = np.zeros((h*w))

        starts, lengths = p[::2], p[1::2]
        for s, l in zip(starts, lengths):
            mask[int(s-1):int(s+l-1)] = mask_val  ## masking

        return mask.reshape(h,w, order='F')
    
        # color : which color to draw mask RGB
    def apply_mask_to_img(self, img, mask, color, mask_val=1, alpha=0.5):
        if not np.ndim(color):
            raise ValueError('color must be 3dim.')
        img[mask==mask_val] = ((1-alpha)*img[mask==mask_val] + alpha*np.array(color))
        return img
    
    def crop_img(self, img_arr, h, w):
        """
            img_arr = (h,w,c) or (h,w)
            h , w = img size after cropping
            
        """
        dims = np.ndim(img_arr)
        if dims is 2:
            h_old, w_old = np.shape(img_arr)
        elif dims is 3:
            h_old, w_old, _ = np.shape(img_arr)
        else:
            raise ValueError('img shape does not fit.')
            
        x = math.ceil((h_old - h) / 2)
        y = math.ceil((w_old - w) / 2)
        
        if dims is 2:
            return img_arr[x:(x + h), y:(y + w)]
        elif dims is 3:
            return img_arr[x:(x + h), y:(y + w), :]
    
    def show_images_by_raw(self, img_arr, ncols, figsize=(20,10)):
        if not np.ndim(img_arr)== 4:
            raise ValueError('img_arr must be 4 dims.')
        num = len(img_arr)
        plt.figure(figsize=figsize)
        for i in range(num):
            plt.subplot(num/ncols, ncols, i+1)
            plt.imshow(img_arr_s[i])
    
    def show_images(self, f_path, image_ids, n_col=2, figsize=(10,10)):

        if isinstance(image_ids, (list, pd.Series)):
            img_arr_list = []
            n_imgs = len(image_ids)
            fig, axes = plt.subplots(ncols=n_col, nrows=int(np.ceil(n_imgs/n_col)), figsize=figsize)

            for i, img_id in enumerate(image_ids):
                img_arr = plt.imread(f_path / img_id)
                axes[i//n_col, i%n_col].imshow(img_arr)
                img_arr_list.append(np.array(img_arr)) # np array 수정권한이 없어서 복사
            return img_arr_list

        else:
            img_arr = plt.imread(f_path / image_ids)
            plt.imshow(img_arr)
            return np.arrary(img_arr)
          
    # img_arr shape = (H,W,C)
    def rgb_to_gray(self, img_arr, new_axis=True):
      if np.ndim(img_arr) != 3:
        raise ValueError('img_arr must be 3 dim')
      #  0.299 * R + 0.587 * G + 0.114 * B
      gray_img = (0.299*img_arr[:,:,0] + 0.587 * img_arr[:,:,1] + 0.114 *img_arr[:,:,2])
      if new_axis:
        return gray_img[:,:,np.newaxis]
      else:
        return gray_img


# In[14]:


# !jupyter nbconvert --to script pre_processing.ipynb


# In[17]:


# from util import csv_file_load
# import pathlib
# import matplotlib.pyplot as plt
# import numpy as np
# p = pathlib.Path('steel_images')
# train_pd = csv_file_load(p/'train.csv')

# img_arr = plt.imread(p/'train_images'/train_pd.ImageId[0] )

# plt.figure(figsize=(30,15))
# plt.subplot(1,2,1)
# plt.imshow(img_arr)
# np.shape(img_arr)
# g = Pre_process_img().rgb_to_gray(img_arr)
# np.shape(g)
# plt.subplot(1,2,2)
# plt.imshow(g, cmap='gray')


# In[6]:


# import cv2

# img = cv2.imread(str(p/'train_images'/train_pd.ImageId[0]))
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow('origin',img)
# cv2.imshow('gray', g)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[8]:


# colors_of_classes = [(0,0,0), (255,0,0), (0,255,0), (0,0,255), (255,0,255)]
# masked_imgs_arr = []
# for i,img in enumerate(img_arr_list):
#     mask = p.decode_pixels_to_mask(np.shape(img), encoded_list[i])
#     masked_img = p.apply_mask_to_img(img, mask, colors_of_classes[classes[i]])
#     masked_imgs_arr.append(masked_img)
    
# fig, axes = plt.subplots(ncols=2, nrows=np.ceil(len(masked_imgs_arr)/2).astype(int), figsize=(30,15))
# for i, m_img in enumerate(masked_imgs_arr):
#     row, col = i//2, i%2
#     axes[row, col].set_title(f'Class {classes[i]}')
#     axes[row, col].imshow(m_img)

