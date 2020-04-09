#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict

import cv2

import pathlib
import math


# In[24]:


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
    def apply_mask_to_img(self, img, mask, color, mask_val=1, alpha=0.5, dataformat='NHWC'):
      '''
          img : (N,H,W,C), (N,C,H,W), (N,H,W) (H,W), (H,W,C), (C,H,W) 
          mask : (N,H,W), (H,W)
      '''
      if len(color) != 3:
        raise ValueError('color must be 3dim.')

      if dataformat in ('NHWC', 'HWC'):
        mask = mask[:,:,np.newaxis]
      elif dataformat == 'NCHW':
        img = np.transpose(img, (0,2,3,1)) # convert into 'NHWC'
        mask = mask[:,:,np.newaxis]
      elif dataforamt == 'CHW':
        img = np.transpose(img, (1,2,0)) # convert into 'HWC'
        mask = mask[:,:,np.newaxis]
      elif dataforamt in('HW', 'NHW'):
        pass
      else:
        raise ValueError('dataformat is wrong')

      img = img.copy()
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


# In[25]:


if __name__ == '__main__':
  get_ipython().system('jupyter nbconvert --to script pre_processing.ipynb')


# In[19]:


if __name__ == '__main__':

  from util import csv_file_load
  import pathlib
  import matplotlib.pyplot as plt
  import numpy as np
  import cv2
  import torch

  p = pathlib.Path('steel_images')
  train_pd = csv_file_load(p/'train.csv')
  pre = Pre_process_img()
  img_arr = plt.imread(p/'train_images'/train_pd.ImageId[0] )



  for k in range(10):
    rand_idx = list(np.random.randint(0, high=len(train_pd), size=10))
    img_arr_list = [np.array(plt.imread(p/'train_images'/train_pd.ImageId[i])) for i in rand_idx]
    masked_imgs_arr = []
    colors_of_classes = [(0,0,0), (255,0,0), (0,255,0), (0,0,255), (255,0,255)]

    for i,img in enumerate(img_arr_list):
        mask = pre.decode_pixels_to_mask(np.shape(img), train_pd.EncodedPixels[rand_idx[i]])
        masked_img = pre.apply_mask_to_img(img, mask, colors_of_classes[train_pd.ClassId[rand_idx[i]]])
        masked_imgs_arr.append(masked_img)

    fig, axes = plt.subplots(ncols=2, nrows=len(masked_imgs_arr), figsize=(15,15))
    fig.tight_layout()
    for i, (m_img, o_img) in enumerate(zip(masked_imgs_arr, img_arr_list)):
        axes[i, 0].set_title(f'Class {train_pd.ClassId[rand_idx[i]]}')
        axes[i, 0].imshow(m_img)
        axes[i, 1].set_title(f'Class {train_pd.ClassId[rand_idx[i]]}')
        axes[i, 1].imshow(o_img)


