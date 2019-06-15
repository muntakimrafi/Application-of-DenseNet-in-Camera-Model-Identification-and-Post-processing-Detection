import numpy as np
import pandas as pd
import os
names = {1:'HTC-1-M7', 
         2:'iPhone-4s', 
         3:'iPhone-6', 
         4:'LG-Nexus-5x', 
         5:'Motorola-Droid-Maxx',
         6:'Motorola-Nexus-6', 
         7:'Motorola-X', 
         8:'Samsung-Galaxy-Note3', 
         9:'Samsung-Galaxy-S4', 
         10:'Sony-NEX-7'}
predicted_rakib =[]
for i in range(len(y_gen2)):
#    predicted_rakib.append(names[y_gen2[i][0].astype(int)])
    predicted_rakib.append(names[y_gen2[i]])

rafi = test_imdir
main_rafi =[]

'''
for windows
'''
#for k in range(len(rafi)):
#     path = os.path.normpath(rafi[k])
#     rafi2 = path.split(os.sep)
#     main_rafi.append(rafi2[3])
'''
for linux
'''     
for k in range(len(rafi)):
     path = os.path.normpath(rafi[k])
     rafi2 = path.split(os.sep)
     main_rafi.append(rafi2[3])
df = pd.DataFrame(columns=['fname', 'camera'])
df['fname'] = main_rafi
df['camera'] = predicted_rakib
df.to_csv("64.csv", index=False)