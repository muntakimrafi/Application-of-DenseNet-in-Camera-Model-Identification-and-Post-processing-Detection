import numpy as np
import math
import time
from PIL import Image
from imageio import imwrite
import glob
import os
from pyemd.EMD2d import EMD2D
import time
from scipy.misc import imread
from scipy.stats import mode
import random
import numpy as np
from sklearn.utils import shuffle
from skimage.restoration import denoise_wavelet
from skimage import img_as_float,img_as_uint
import matplotlib.pyplot as plt
import pickle
from imageio import imwrite
from PIL import Image
imreadpath = ''
imwritepath = ''

model = ['htc', 'iphone4', 'iphone6', 'lg', 'motodroid', 'motonex', 'motox', 'samsung_galaxynote', 'samsung_galaxys', 'sony'] 
#list of all models 

num_row = 256
num_col = 256

        
for m in range(len(model)):
    start = time.time()
    images = glob.glob(imreadpath + model[m] + '/*')

    os.makedirs(imwritepath+model[m])
    img_no = 0
    for img_name in images:
        img_no += 1 
        
        im = Image.open(img_name)
        single_image = np.array(im)
#        single_image = np.load(img_name)
        red = single_image[:,:,0]
        green = single_image[:,:,1]
        blue = single_image[:,:,2]
        
        img_emd_parts = np.zeros((num_row,num_col,3))    
        
        emd2d = EMD2D()
        try:
            IMFred = emd2d.emd(red, max_imf = -1)
        except:
            continue
        try:
            IMFgreen = emd2d.emd(green, max_imf = -1)
        except:
            continue    
        try:
            IMFblue = emd2d.emd(blue, max_imf = -1)
        except: 
            continue
        try:
            img_emd_parts[:,:,0] = IMFred[0]
        except: 
            continue
        try:    
            img_emd_parts[:,:,1] = IMFgreen[0]
        except: 
            continue
        try:    
            img_emd_parts[:,:,2] = IMFblue[0]
        except: 
            continue 
    
        sub_1st_imf = img_emd_parts.astype('float32')
        
        sub_1st_imf2 = sub_1st_imf - np.min(sub_1st_imf)*np.ones(sub_1st_imf.shape)
        sub_1st_imf3 = sub_1st_imf2/np.max(sub_1st_imf2)
        sub_1st_imf4 = sub_1st_imf3*255
        sub_1st_imf5 = np.around(sub_1st_imf4).astype('uint8')
        image_wrt_dir = imwritepath+ model[m]  +'\\kaggle_emd_{}.png'.format(img_name.split(os.sep)[-1].split('.')[0])
#        sub_1st_imf.dump(image_wrt_dir)
        imwrite(image_wrt_dir,sub_1st_imf5)
        print(img_no)
        print(time.time()-start)





        

 
    