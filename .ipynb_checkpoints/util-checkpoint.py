#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import shutil

import pathlib
from tqdm import tqdm 


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



# In[1]:


if __name__ is '__main__':
  get_ipython().system('jupyter nbconvert --to script util.ipynb')

