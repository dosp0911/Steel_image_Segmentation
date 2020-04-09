#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import shutil

import pathlib
from tqdm import tqdm

import torch
import torch.nn as nn


# In[2]:


def csv_file_load(f_p, index_col=False ):
    if f_p.exists():
        return pd.read_csv(f_p, index_col=index_col)
    else:
        raise FileExistsError(f'{f_p} no exist!')


# In[16]:


def move_files_to_class_folders(f_names, classes, root_f):
    root_path = pathlib.Path(str(root_f))
    if not(root_path.exists()):
        raise FileExistsError(f'{root_f} does not exist')
        
    class_dirs = pd.unique(classes).astype('str')
    for d in class_dirs:
        class_dir = root_path / d
        if not class_dir.exists():
            class_dir.mkdir()
        
    for file, c_ in tqdm(zip(f_names, classes.astype('str'))):
        shutil.copy(str(root_path / file), str(root_path / c_))
        
    print('Done.')


# In[ ]:


def print_model_memory_size(model):
    total_ = 0
    for k, v in model.state_dict().items():
        print(f'name:{k} size:{v.size()} dtype:{v.dtype}')
        total_ += v.numel()
    print(f'Model size : {total_*4} byte -> {total_*4/1024**2} MiB')


# In[49]:


def get_pixel_value_frequencies(img_arr, dtype=int):
  '''
    img_arr = (N,H,W) or (H,W) 
    dtype = pixel values 
    counts unique pixel values of images
  '''
  arr = np.reshape(img_arr, -1)
  uvals = np.unique(arr).astype(dtype)
  uvals_dic = {}
  
  for u in list(uvals):
    uvals_dic[u] = np.sum(arr==u)
  return uvals_dic


# In[46]:


def get_weights_ratio_over_frequnecies(freq):
  '''
    # [2,3,4,5] -> [1/2, 1/3, 1/4, 1/5]
  '''
  return list(map(lambda x: 1/x, freq))


# In[ ]:


def load_model(path, model):
  load_model = torch.load(path, map_location=device)
  model.load_state_dict(load_model['model_state_dict'])
  return model

  u_net = load_model('u_net_1e_114l.pt', u_net)


# In[ ]:


def display_weights_of_model(model):
  l_p = sum(1 for x in model.parameters())
  fg, axes = plt.subplots(l_p//5+1, 5, figsize=(15,15))
  fg.tight_layout()

  for i, p in enumerate(model.parameters()):
    sns.distplot(p.detach().numpy(), ax=axes[i//5,i%5])


# In[28]:


class class2d_to_onehot(nn.Module):
  def __init__(self, classes):
    '''
    args:
      classes: [0,1,2,3... labels] labels must be integer
      It will add channles of the number of labels to target 
    '''
    super(class2d_to_onehot, self).__init__()
    self.classes = torch.tensor(classes).unique()
    
  def forward(self, target):
    '''
      args: 
        target: (N,H,W), (H,W)
      return:
        (N,H,W)->(N,C,H,W)
        (H,W)->(C,H,W)
    ''' 
    ndims = len(target.size())

    assert ndims == 2 or ndims == 3

    if ndims == 2:
      cls_stacks = torch.stack([(target==c).type(torch.float32) for c in self.classes], dim=0)
    elif ndims == 3:
      cls_stacks = torch.stack([(target==c).type(torch.float32) for c in self.classes], dim=1)

    return cls_stacks
  


# In[33]:


if __name__ == '__main__':
  get_ipython().system('jupyter nbconvert --to script util.ipynb')


# In[30]:


torch.tensor([6,3,4,5,1]).unique()


# In[32]:


if __name__ == '__main__':

  a = torch.zeros(3,7,8)
  a[0,3,:] = 1
  a[0,4,:] = 5
  a[1,2,:] = 7
  a[1,:,7] = 1
  a[2,1:3,5:] = 2
  a[2,3:5,2:4] = 5

  print(a)
  b = class2d_to_onehot([1,2,5])(a)
  # 0, 1, 2, 5, 7
  print(b.size())
  print(b)
  # torch.stack([a,b,c], dim=1)

