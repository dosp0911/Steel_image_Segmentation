#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import util


# In[24]:


def Dice(pred, target, threshhold=0.5, dims=(2,3)):
  smooth = 1e-4
  pred = nn.Sigmoid(pred)
  intersection = (pred * target).sum(dim=dims) 
  union = pred.sum(dim=dims) + target.sum(dim=dims) 

  dice = torch.mean((2 * intersection + smooth) / (union + smooth))
  return dice



def Iou(p, g):
  p_ = p.contiguous().view(-1)
  g_ = g.contiguous().view(-1)
  intersection = (p_ * g_).sum()
  union = p_.sum() + g_.sum()
  smooth = 1e-4

  return (intersection + smooth) / (union - intersection + smooth)  

def mAP(p, r):
  '''
    p : precision
    r : recall
  '''
  return 1

def Recall(p, g):
  p_ = p.contiguous().view(-1)
  g_ = g.contiguous().view(-1)
  intersection = (p_ * g_).sum()
  smooth = 1
  p_sum = p_.sum()
  g_sum = g_.sum()
  
  return (intersection + smooth) / g_sum + smooth

def Precision(p, g):
  p_ = p.contiguous().view(-1)
  g_ = g.contiguous().view(-1)
  intersection = (p_ * g_).sum()
  smooth = 1
  p_sum = p_.sum()
  g_sum = g_.sum()
  
  return (intersection + smooth) / p_sum + smooth

def F_score(p, r):
  return 2 * (p + r) / p * r 


# In[17]:



class GeneralizedDICELoss(nn.Module):
  '''
    weigths and pred classes must have same size
  '''
  def __init__(self, weight=None, reduction='mean'):
    super(GeneralizedDICELoss, self).__init__()
    self.weigth = torch.tensor(weight)
    
  def forward(self, pred, target, dims=(2,3), threshhold=0.5, reduction='mean'):
    """
      args:
        pred : (N,C,H,W)->dim=(2,3), (N, H, W)->dim=(1,2), (H , W)->dim=None 
        target : (N, C, H, W), (C, H, W) one_hot_eocoded 
        theshhold : to be True
        reduction : 'mean', 'sum'
        default dim=(2,3), threshhold =0.5, reduction='mean'
      return :
        1 - dice . mean reduction
        
      target must be one_hot_encoded and has same number of channles as pred
    """
    
    assert 1 < len(pred.size()) < 5 and 1 < len(target.size()) < 5
    
    
    smooth = 1e-4
    pred = nn.Sigmoid(pred)
      
    intersection = (pred * target).sum(dim=dims) 
    union = pred.sum(dim=dims) + target.sum(dim=dims) 
    
    dice = (2 * intersection + smooth) / (union + smooth)
    
    if reduction.lower() == 'sum':
      dice = torch.sum(self.weigth * dice)
    else:
      dice = torch.mean(self.weigth * dice)
  
    return 1 - dice


# In[14]:


class DICELoss(nn.Module):
  def __init__(self):
    super(DICELoss, self).__init__()
    
  def forward(self, pred, target, dims=(2,3), threshhold=0.5, reduction='mean'):
    """
      args:
        pred : (N,C,H,W)->dim=(2,3), (N, H, W)->dim=(1,2), (H , W)->dim=None 
        target : (N, C, H, W), (C, H, W) one_hot_eocoded 
        theshhold : to be True
        reduction : 'mean', 'sum'
        default dim=(2,3), threshhold =0.5, reduction='mean'
      return :
        1 - dice . mean reduction
        
      target must be one_hot_encoded and has same number of channles as pred
    """
    
    assert 1 < len(pred.size()) < 5 and 1 < len(target.size()) < 5
    
#     pred.requires_grad_(True)
#     target.requires_grad_(True)
    
    smooth = 1e-4
#     pred = (pred >= threshhold).type(torch.float32) # try changing into sigmoid function
    pred = nn.Sigmoid(pred)  
    intersection = (pred * target).sum(dim=dims) 
    union = pred.sum(dim=dims) + target.sum(dim=dims) 
    
    if reduction.lower() == 'sum':
      dice = torch.sum((2 * intersection + smooth) / (union + smooth))
    else:
      dice = torch.mean((2 * intersection + smooth) / (union + smooth))
  
    return 1 - dice
  


# In[23]:


if __name__ == '__main__':
  get_ipython().system('jupyter nbconvert --to script Metrics.ipynb')

