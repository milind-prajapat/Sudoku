#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.simplefilter("ignore")

import os
import random
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[ ]:


Training_Data = []

for File in os.listdir("Fonts"):
    df = pd.read_csv(os.path.join("Fonts",File))
    df = df[df["m_label"].isin(list(range(ord('0'),ord('9')+1)))]
    df.reset_index(inplace=True)
    for row in df.index:
        img = list(df.loc[row,"r0c0":"r19c19"])
        Image = []
        for i in range(0,39,2):
            Image.append(img[i*10:(i+2)*10])
        img = np.array(Image,np.uint8)
        Training_Data.append([img,df["m_label"][row]-ord('0')])

random.shuffle(Training_Data)


# In[ ]:


for img,target in Training_Data[:50]:
    plt.imshow(img,cmap='gray',interpolation = 'nearest')
    plt.title("Target: "+str(target))
    plt.show()


# In[ ]:


x_train = []
y_train = []

random.shuffle(Training_Data)
for x,y in Training_Data:
    x_train.append(x)
    y_train.append(y)


# In[ ]:


with open('x_train.pickle', 'wb') as p:
    pickle.dump(x_train, p)
with open('y_train.pickle', 'wb') as p:
    pickle.dump(y_train, p)

