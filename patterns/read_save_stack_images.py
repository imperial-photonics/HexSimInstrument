import os
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tifffile as tif
import numpy as np
from PIL import Image

file_num = 7

img = np.zeros((7,1000,1000))

for i in range (file_num):
    name_tmp = str(i).zfill(3)
    file_name = name_tmp+'.bmp'
    img_tmp = Image.open(file_name)
    img[i,:,:] = np.sum(np.array(img_tmp),axis=2)/3
    
    #img[i,:,:] = np.array(img_tmp)

#y = x[:,:,1]