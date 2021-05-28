import os
import cv2
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = load_model(os.path.join('Model', 'best_val_loss.hdf5'))

x_validation = []
y_validation = []

for Class in range(10):
    for Image_Name in os.listdir(os.path.join('Split Dataset', 'Validation', str(Class))):
        x_validation.append(cv2.imread(os.path.join('Split Dataset', 'Validation', str(Class), Image_Name), 0))
        y_validation.append(Class)

x_validation = np.array(x_validation).reshape(-1, 20, 20, 1) / 255.0
Prediction = np.argmax(model.predict(x_validation), axis = 1)

Dict = {}
Dict['Model'] = [accuracy_score(y_validation, Prediction),
                    precision_score(y_validation, Prediction, average = 'weighted', zero_division = 0), 
                    recall_score(y_validation, Prediction, average = 'weighted', zero_division = 0), 
                    f1_score(y_validation, Prediction, average = 'weighted', zero_division = 0)]

Validation_Report = pd.DataFrame.from_dict(Dict, orient = 'index', columns = ['accuracy_score', 'precision_score', 'recall_score', 'f1_score']).round(4)
print(Validation_Report)