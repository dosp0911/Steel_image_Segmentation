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




