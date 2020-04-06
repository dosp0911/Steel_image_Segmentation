#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch 
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import math


# In[ ]:


# class Cont2D(nn.Module):
  
#   def __init__(self, in_channel, out_channel, kernel_size, act_f='relu'):
#     super(self,Cont2D).__init__()
    
#     self.in_channel = in_channel
#     self.out
    
    


# In[5]:


class U_net(nn.Module):
    
    def __init__(self, in_channel, n_classes):
        super(U_net, self).__init__()
        
        #input image channels
        self.in_channel = in_channel
        # contractive path conv kernel size:(3x3)
        self.c_kernel_size = 3
        # expansive path conv kernel size:(3x3)
        self.up_kernel_size = 2
        
        self.final_kernel_size = 1
        self.max_pool_kernel_size = 2
        self.max_pool_stride = 2
        
        self.deconv_stride = 2
        
        # activation function
        self.c_activation_f = nn.ReLU(inplace=True)
        
       
        
        # channels
        self.n_out_channels1 = 64
        self.n_out_channels2 = 128
        self.n_out_channels3 = 256
        self.n_out_channels4 = 512
        self.n_out_channels5 = 1024
        self.final_out_channels = n_classes
        
        self.n_skip_channels1 = 128
        self.n_skip_channels2 = 256
        self.n_skip_channels3 = 512
        self.n_skip_channels4 = 1024
        
        self.c_max_pooling =  nn.MaxPool2d(self.max_pool_kernel_size, stride=self.max_pool_stride)
        
         # Batch normalization
        self.c_bn11 = nn.BatchNorm2d(self.n_out_channels1)
        self.c_bn12 = nn.BatchNorm2d(self.n_out_channels1)
        self.c_bn21 = nn.BatchNorm2d(self.n_out_channels2)
        self.c_bn22 = nn.BatchNorm2d(self.n_out_channels2)
        self.c_bn31 = nn.BatchNorm2d(self.n_out_channels3)
        self.c_bn32 = nn.BatchNorm2d(self.n_out_channels3)
        self.c_bn41 = nn.BatchNorm2d(self.n_out_channels4)
        self.c_bn42 = nn.BatchNorm2d(self.n_out_channels4)
        self.c_bn51 = nn.BatchNorm2d(self.n_out_channels5)
        self.c_bn52 = nn.BatchNorm2d(self.n_out_channels5)
        
        ### contractive path layers ###
        # 1th layer, size of input image : (batch, c=3, h=256, w=1600)
        self.cont_layer1 = nn.Sequential(OrderedDict([
           ('c_conv11', nn.Conv2d(in_channels=self.in_channel, out_channels=self.n_out_channels1, kernel_size=self.c_kernel_size)),
           ('c_act_f11', self.c_activation_f),
           ('c_bn_11', self.c_bn11),
           ('c_conv12', nn.Conv2d(in_channels=self.n_out_channels1, out_channels=self.n_out_channels1, kernel_size=self.c_kernel_size)),
           ('c_act_f12', self.c_activation_f),
           ('c_bn_12', self.c_bn12)
        ]))
    
        # 2th layer
        self.cont_layer2 = nn.Sequential(OrderedDict([
            ('c_conv21', nn.Conv2d(in_channels=self.n_out_channels1, out_channels=self.n_out_channels2, kernel_size=self.c_kernel_size)),
            ('c_act_f21', self.c_activation_f),
            ('c_bn_21', self.c_bn21),
            ('c_conv22', nn.Conv2d(in_channels=self.n_out_channels2, out_channels=self.n_out_channels2, kernel_size=self.c_kernel_size)),
            ('c_act_f22', self.c_activation_f),
            ('c_bn_22', self.c_bn22)
        ]))

        # 3th layer
        self.cont_layer3 = nn.Sequential(OrderedDict([
            ('c_conv31', nn.Conv2d(in_channels=self.n_out_channels2, out_channels=self.n_out_channels3, kernel_size=self.c_kernel_size)),
            ('c_act_f31', self.c_activation_f),
            ('c_bn_31', self.c_bn31),
            ('c_conv32', nn.Conv2d(in_channels=self.n_out_channels3, out_channels=self.n_out_channels3, kernel_size=self.c_kernel_size)),
            ('c_act_f32', self.c_activation_f),
            ('c_bn_32', self.c_bn32),
        ]))
        # 4th layer
        self.cont_layer4 = nn.Sequential(OrderedDict([
            ('c_conv41', nn.Conv2d(in_channels=self.n_out_channels3, out_channels=self.n_out_channels4, kernel_size=self.c_kernel_size)),
            ('c_act_f41', self.c_activation_f),
            ('c_bn_41', self.c_bn41),
            ('c_conv42', nn.Conv2d(in_channels=self.n_out_channels4, out_channels=self.n_out_channels4, kernel_size=self.c_kernel_size)),
            ('c_act_f42', self.c_activation_f),
            ('c_bn_42', self.c_bn42)
        ]))
        # 5th layer
        self.cont_layer5 = nn.Sequential(OrderedDict([
            ('c_conv51', nn.Conv2d(in_channels=self.n_out_channels4, out_channels=self.n_out_channels5, kernel_size=self.c_kernel_size)),
            ('c_act_f51', self.c_activation_f),
            ('c_bn_51', self.c_bn51),
            ('c_conv52', nn.Conv2d(in_channels=self.n_out_channels5, out_channels=self.n_out_channels5, kernel_size=self.c_kernel_size)),
            ('c_act_f52', self.c_activation_f),
            ('c_bn_52', self.c_bn52)
        ]))
        
        ### expansive path layers ###
        self.exp_layer5 = nn.ConvTranspose2d(in_channels=self.n_out_channels5, out_channels=self.n_out_channels4, 
                                                 kernel_size=self.up_kernel_size, stride=self.deconv_stride)
        
        # 4th layer
        self.exp_layer4 = nn.Sequential(OrderedDict([
            ('e_conv41', nn.Conv2d(in_channels=self.n_skip_channels4, out_channels=self.n_out_channels4, kernel_size=self.c_kernel_size)),
            ('e_act_f41', self.c_activation_f),
            ('e_conv42', nn.Conv2d(in_channels=self.n_out_channels4, out_channels=self.n_out_channels4, kernel_size=self.c_kernel_size)),
            ('e_act_f42', self.c_activation_f),
            ('e_up_conv4', nn.ConvTranspose2d(in_channels=self.n_out_channels4, out_channels=self.n_out_channels3, 
                                                 kernel_size=self.up_kernel_size, stride=self.deconv_stride))
        ]))
        
        # 3th layer
        self.exp_layer3 = nn.Sequential(OrderedDict([
            ('e_conv31', nn.Conv2d(in_channels=self.n_skip_channels3, out_channels=self.n_out_channels3, kernel_size=self.c_kernel_size)),
            ('e_act_f31', self.c_activation_f),
            ('e_conv32', nn.Conv2d(in_channels=self.n_out_channels3, out_channels=self.n_out_channels3, kernel_size=self.c_kernel_size)),
            ('e_act_f32', self.c_activation_f),
            ('e_up_conv3', nn.ConvTranspose2d(in_channels=self.n_out_channels3, out_channels=self.n_out_channels2,
                                                kernel_size=self.up_kernel_size, stride=self.deconv_stride))
        ]))
        
        # 2th layer
        self.exp_layer2 = nn.Sequential(OrderedDict([
            ('e_conv21', nn.Conv2d(in_channels=self.n_skip_channels2, out_channels=self.n_out_channels2, kernel_size=self.c_kernel_size)),
            ('e_act_f21', self.c_activation_f),
            ('e_conv22', nn.Conv2d(in_channels=self.n_out_channels2, out_channels=self.n_out_channels2, kernel_size=self.c_kernel_size)),
            ('e_act_f22', self.c_activation_f),
            ('e_up_conv2', nn.ConvTranspose2d(in_channels=self.n_out_channels2, out_channels=self.n_out_channels1,
                                                kernel_size=self.up_kernel_size, stride=self.deconv_stride))
        ]))
        
        # 1th layer
        self.exp_layer1 = nn.Sequential(OrderedDict([
            ('e_conv11', nn.Conv2d(in_channels=self.n_skip_channels1, out_channels=self.n_out_channels1, kernel_size=self.c_kernel_size)),
            ('e_act_f11', self.c_activation_f),
            ('e_conv12', nn.Conv2d(in_channels=self.n_out_channels1, out_channels=self.n_out_channels1, kernel_size=self.c_kernel_size)),
            ('e_act_f12', self.c_activation_f),
            ('e_conv_f', nn.Conv2d(in_channels=self.n_out_channels1, out_channels=self.final_out_channels,
                                   kernel_size=self.final_kernel_size))
         ]))
        
        #### without drop-out implemented
        
        # skip operation : skip -> crop and concatenate
        # return concat [n_batch, n_ch, x, h] [n_batch, n_ch, x, h] -> [n_batch, n_ch+n_ch, x, h]
    def skipped_connection(self, cont_maps, exp_maps, height, width):
        cropped_f_maps = self.crop_feature_maps(cont_maps, height, width)
        return torch.cat((cropped_f_maps, exp_maps), 1)
        
    #features = [batchs, n_channels, height , width]
    # h,w = crop후 image size
    def crop_feature_maps(self, features, h, w):
        h_old, w_old = features[0][0].size()
        x = math.ceil((h_old - h) / 2)
        y = math.ceil((w_old - w) / 2)
        return features[:,:, x:(x + h), y:(y + w)]
    
    def contracting_path(self, x):
        self.cont_layer1_out = self.cont_layer1(x)
        self.cont_layer2_in = self.c_max_pooling(self.cont_layer1_out)
        
        self.cont_layer2_out = self.cont_layer2(self.cont_layer2_in)
        self.cont_layer3_in = self.c_max_pooling(self.cont_layer2_out)
        
        self.cont_layer3_out = self.cont_layer3(self.cont_layer3_in)
        self.cont_layer4_in = self.c_max_pooling(self.cont_layer3_out)
        
        self.cont_layer4_out = self.cont_layer4(self.cont_layer4_in)
        self.cont_layer5_in = self.c_max_pooling(self.cont_layer4_out)
        
        self.cont_layer5_out = self.cont_layer5(self.cont_layer5_in)
        
        return self.cont_layer5_out
    
    # x = cont_layer5_out
    def expansive_path(self, x):
        x = self.exp_layer5(x)
        x = self.skipped_connection(self.cont_layer4_out, x, x.size()[2], x.size()[3])
        x = self.exp_layer4(x)
        x = self.skipped_connection(self.cont_layer3_out, x, x.size()[2], x.size()[3])
        x = self.exp_layer3(x)
        x = self.skipped_connection(self.cont_layer2_out, x, x.size()[2], x.size()[3] )
        x = self.exp_layer2(x)
        x = self.skipped_connection(self.cont_layer1_out, x, x.size()[2], x.size()[3] )
        x = self.exp_layer1(x)
        return x
   
    # input_x has to be shape of (n_batches, n_channels, height, width) 
    def forward(self, input_x):
        o_x = self.contracting_path(input_x)
        o_x = self.expansive_path(o_x)
        return o_x
        


# In[6]:





# In[3]:


# !jupyter nbconvert --to script model.ipynb

