import warnings
warnings.simplefilter("ignore")

import pickle
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD

with open('x_train.pickle', 'rb') as p:
    x_train = pickle.load(p)
with open('y_train.pickle', 'rb') as p:
    y_train = pickle.load(p)

x_train = np.array(x_train).reshape(-1, 20, 20, 1) / 255.0
y_train = to_categorical(y_train)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', input_shape = x_train.shape[1:]))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(100, activation = 'relu', kernel_initializer = 'he_uniform'))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = SGD(lr = 0.01, momentum = 0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 10)

model.save("Printed-Digits-CNN-Model")