import os
import pickle
import numpy as np

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

pickle_in = open(os.path.join('Dataset', 'x_train.pickle'), 'rb')
x_train = pickle.load(pickle_in)
pickle_in.close()
 
pickle_in = open(os.path.join('Dataset', 'y_train.pickle'), 'rb')
y_train = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(os.path.join('Dataset', 'x_validation.pickle'), 'rb')
x_validation = pickle.load(pickle_in)
pickle_in.close()
 
pickle_in = open(os.path.join('Dataset', 'y_validation.pickle'), 'rb')
y_validation = pickle.load(pickle_in)
pickle_in.close()

x_train = np.array(x_train).reshape(-1, 20, 20, 1) / 255.0
x_validation = np.array(x_validation).reshape(-1, 20, 20, 1) / 255.0

model = Sequential()

model.add(Conv2D(32, (3, 3), strides = 1, activation = 'relu', kernel_initializer = 'he_uniform', input_shape = x_train.shape[1:]))
model.add(MaxPooling2D((2, 2), strides = (2, 2), padding = 'same'))

model.add(Conv2D(64, (3, 3), strides = 1, activation = 'relu', kernel_initializer = 'he_uniform'))
model.add(MaxPooling2D((2, 2), strides = (2, 2), padding = 'same'))

model.add(Flatten())
model.add(Dense(100, activation = 'relu', kernel_initializer = 'he_uniform'))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = SGD(lr = 0.01, momentum = 0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])

if not os.path.isdir('Model'):
    os.mkdir('Model')

callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                              patience = 7, min_lr = 1e-5),
             EarlyStopping(patience = 9, # Patience should be larger than the one in ReduceLROnPlateau
                          min_delta = 1e-5),
             CSVLogger(os.path.join('Model', 'training.log'), append = True),
             ModelCheckpoint(os.path.join('Model', 'backup_last_model.hdf5')),
             ModelCheckpoint(os.path.join('Model', 'best_val_acc.hdf5'), monitor = 'val_accuracy', mode = 'max', save_best_only = True),
             ModelCheckpoint(os.path.join('Model', 'best_val_loss.hdf5'), monitor = 'val_loss', mode = 'min', save_best_only = True)]

model.fit(x_train, y_train, epochs = 50, validation_data = (x_validation, y_validation), callbacks = callbacks)

model = load_model(os.path.join('Model', 'best_val_loss.hdf5'))
loss, acc = model.evaluate(x_validation, y_validation)

print('Loss on Validation Data : ', loss)
print('Accuracy on Validation Data :', '{:.4%}'.format(acc))