import os
import random
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

Files = os.listdir('Fonts')

Files_To_Remove = ['GOTHICE.csv',
                  'JOKERMAN.csv',
                  'GILL.csv',
                  'OCRB.csv',
                  'CURLZ.csv',
                  'SNAP.csv',
                  'VIN.csv',
                  'BROADWAY.csv',
                  'E13B.csv',
                  'GIGI.csv',
                  'CHILLER.csv',
                  'BLACKADDER.csv',
                  'COUNTRYBLUEPRINT.csv',
                  'RAGE.csv',
                  'NUMERICS.csv',
                  'CREDITCARD.csv',
                  'OCRA.csv']

for File in Files_To_Remove:
    Files.remove(File)

Data = [[] for _ in range(10)]

for File in tqdm(Files, unit_scale = True, miniters = 1, desc = 'Loading Dataset '):
    df = pd.read_csv(os.path.join('Fonts', File))
    df = df[df['m_label'].isin(list(range(ord('0'), ord('9') + 1)))]
    df.reset_index(inplace = True)

    for row in df.index:
        List = list(df.loc[row, 'r0c0':'r19c19'])

        Image = []
        for i in range(0, 39, 2):
            Image.append(List[i * 10:(i + 2) * 10])
        Image = np.array(Image, np.uint8)

        Data[df['m_label'][row] - ord('0')].append(Image)

Training_Data = [[Image, index] for index, Feature in enumerate(Data) for Image in Feature[200:]]
random.shuffle(Training_Data)

Validation_Data = [[Sample, index] for index, Feature in enumerate(Data) for Sample in Feature[:200]]
random.shuffle(Validation_Data)

x_train, y_train = np.array([Sample[0] for Sample in Training_Data], dtype = np.float64), to_categorical(np.array([Sample[1] for Sample in Training_Data], dtype = np.int64))
x_validation, y_validation = np.array([Sample[0] for Sample in Validation_Data], dtype = np.float64), to_categorical(np.array([Sample[1] for Sample in Validation_Data], dtype = np.int64))

if not os.path.isdir('Dataset'):
    os.mkdir('Dataset')

pickle_out = open(os.path.join('Dataset', 'x_train.pickle'), 'wb')
pickle.dump(x_train, pickle_out)
pickle_out.close()
 
pickle_out = open(os.path.join('Dataset', 'y_train.pickle'), 'wb')
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join('Dataset', 'x_validation.pickle'), 'wb')
pickle.dump(x_validation, pickle_out)
pickle_out.close()
 
pickle_out = open(os.path.join('Dataset', 'y_validation.pickle'), 'wb')
pickle.dump(y_validation, pickle_out)
pickle_out.close()