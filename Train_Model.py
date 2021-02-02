#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.simplefilter("ignore")

import pickle
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard


# In[ ]:


with open('x_train.pickle', 'rb') as p:
    x_train = pickle.load(p)
with open('y_train.pickle', 'rb') as p:
    y_train = pickle.load(p)


# In[ ]:


for img,target in list(zip(x_train,y_train))[:50]:
    plt.imshow(img,cmap='gray',interpolation = 'nearest')
    plt.title("Target: "+str(target))
    plt.show()


# In[ ]:


x_train = np.array(x_train).reshape(-1,20,20,1) / 255.0

y_train = to_categorical(y_train)


# In[ ]:


tensorboard = TensorBoard(log_dir = r"logs/3-Conv-100-Nodes-2-Dense-Printed-Digits")

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

opt = SGD(lr=0.01, momentum=0.9)


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(x_train,y_train, epochs = 10,callbacks = [tensorboard])


# In[ ]:


model.save("3-Conv-100-Nodes-2-Dense-CNN-Printed-Digits")


# In[ ]:


model = load_model("3-Conv-100-Nodes-2-Dense-CNN-Printed-Digits")


# In[ ]:


model.evaluate(x_train,y_train)

