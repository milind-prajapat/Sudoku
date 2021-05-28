import os

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

trainDataGen = ImageDataGenerator(rotation_range = 5,
                                  width_shift_range = 0.1,
                                  height_shift_range = 0.1,
                                  rescale = 1.0 / 255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = False,
                                  fill_mode = 'nearest')

testDataGen = ImageDataGenerator(rescale = 1.0 / 255)

trainGenerator = trainDataGen.flow_from_directory(os.path.join('Split Dataset', 'Train'),
                                                  target_size = (20, 20),
                                                  batch_size = 32,
                                                  color_mode = 'grayscale',
                                                  classes = [str(Class) for Class in range(10)],
                                                  class_mode = 'categorical')

validationGenerator = testDataGen.flow_from_directory(os.path.join('Split Dataset', 'Validation'),
                                                      target_size = (20, 20),
                                                      batch_size = 32,
                                                      color_mode = 'grayscale',
                                                      classes = [str(Class) for Class in range(10)],
                                                      class_mode = 'categorical')

model = Sequential()

model.add(Conv2D(32, (3, 3), strides = 1, activation = 'relu', kernel_initializer = 'he_uniform', input_shape = (20, 20, 1)))
model.add(MaxPooling2D((2, 2), strides = (2, 2), padding = 'same'))

model.add(Conv2D(64, (3, 3), strides = 1, activation = 'relu', kernel_initializer = 'he_uniform'))
model.add(MaxPooling2D((2, 2), strides = (2, 2), padding = 'same'))

model.add(Flatten())

model.add(Dense(100, activation = 'relu', kernel_initializer = 'he_uniform'))

model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = SGD(learning_rate = 0.01, momentum = 0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])

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

model.fit(trainGenerator, epochs = 50, validation_data = validationGenerator, callbacks = callbacks)

model = load_model(os.path.join('Model', 'best_val_loss.hdf5'))
loss, acc = model.evaluate(validationGenerator)

print('Loss on Validation Data : ', loss)
print('Accuracy on Validation Data :', '{:.4%}'.format(acc))