#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2


# In[2]:


NUM_PATCHES = 3000
NUM_PATCHES_str = str(NUM_PATCHES)

DATASET = 'test'

csv_data = '/home/niccolo/ExamodePipeline/WSI_patches/'+NUM_PATCHES_str+'/'+DATASET+'_'+NUM_PATCHES_str+'/csv_'+DATASET+'_'+NUM_PATCHES_str+'.csv'
#csv_data = '/home/niccolo/ExamodePipeline/panda_challenge/patches/csv_panda_densely.csv'

dataset = pd.read_csv(csv_data,header=None).values
data_test_paths = dataset[:,0]

data_dir = '/home/niccolo/ExamodePipeline/WSI_patches/'+NUM_PATCHES_str+'/'+DATASET+'_'+NUM_PATCHES_str+'/'
#data_dir = '/home/niccolo/ExamodePipeline/panda_challenge/patches/'
THRESHOLD = 1000


# In[3]:


def create_dict(filename,br):
    br_dict={'f':filename,'br':br}
    return br_dict


# In[4]:


def BR(img):
    np_img = np.asarray(img)
    np_img.shape
    
    r = np_img[:,:,0].astype('uint16')
    g = np_img[:,:,1].astype('uint16')
    b = np_img[:,:,2].astype('uint16')
    
    br_mat = 100.*b/(1+g+r)*(256./(1+g+r+b))
    
    #equalisation and median
    br_mat = br_mat.astype('uint16')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    br_val = clahe.apply(br_mat)
    
    br = np.median(br_val)
    
    return br


# In[5]:


def BR_sorted(csv_local):
    array_dict = []
    for f in csv_local:
        img = Image.open(f) 
        br = BR(img)
        array_dict.append(create_dict(f,br))
    return array_dict


# In[ ]:


n = 0
for p in dataset:
    p = p[0]
    print(n)
    print(p)
    d = os.path.split(p)[1]
    directory = data_dir+d
    csv_local_filename = directory+'/'+d+'_'+NUM_PATCHES_str+'.csv'
    #csv_local_filename = directory+'/'+d+'_paths_densely.csv'
    csv_local = pd.read_csv(csv_local_filename,header=None).values[:,0]
    local_sorted = BR_sorted(csv_local)
    array_dict = sorted(local_sorted, key=lambda k: k['br'],reverse=True) 
    new_csv_file = directory+'/'+d+'_'+NUM_PATCHES_str+'_sorted_br_patches.csv'
    #new_csv_file = directory+'/'+d+'_sorted_br_patches.csv'
    
    filenames_i = []
    labels_i = []
    
    #print(array_dict)

    for i in range(len(array_dict)):

        filenames_i.append(array_dict[i]['f'])
    
    File = {'filename':filenames_i}
    df = pd.DataFrame(File,columns=['filename'])
    np.savetxt(new_csv_file, df.values, fmt='%s',delimiter=',')
    
    n = n+1


# In[ ]:




