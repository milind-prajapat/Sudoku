import os
import cv2
import shutil
import random
import numpy as np
import pandas as pd

from tqdm import tqdm

random.seed(0)

Files = os.listdir('Dataset')

for File in ['GOTHICE.csv', 'JOKERMAN.csv', 'GILL.csv', 'OCRB.csv', 'CURLZ.csv', 'SNAP.csv', 'VIN.csv', 'BROADWAY.csv', 'E13B.csv', 
             'GIGI.csv', 'CHILLER.csv', 'BLACKADDER.csv', 'COUNTRYBLUEPRINT.csv', 'RAGE.csv', 'NUMERICS.csv', 'CREDITCARD.csv', 'OCRA.csv']:
    Files.remove(File)

if os.path.isdir('Split Dataset'):
    shutil.rmtree('Split Dataset')

os.mkdir('Split Dataset')
os.mkdir(os.path.join('Split Dataset', 'Train'))
os.mkdir(os.path.join('Split Dataset', 'Validation'))

for Class in range(10):
    os.mkdir(os.path.join('Split Dataset', 'Train', str(Class)))
    os.mkdir(os.path.join('Split Dataset', 'Validation', str(Class)))

Data = [[] for _ in range(10)]

for File in tqdm(Files, unit_scale = True, miniters = 1, desc = 'Loading Dataset '):
    df = pd.read_csv(os.path.join('Dataset', File))
    df = df[df['m_label'].isin(list(range(ord('0'), ord('9') + 1)))]
    df.reset_index(inplace = True)

    for row in df.index:
        List = list(df.loc[row, 'r0c0':'r19c19'])

        Image = []
        for i in range(0, 39, 2):
            Image.append(List[i * 10:(i + 2) * 10])
        
        Image = np.array(Image, np.uint8)
        Label = df['m_label'][row] - ord('0')
        Data[Label].append(Image)

Count = [0 for _ in range(10)]

for Class, Feature in enumerate(tqdm(Data, unit_scale = True, miniters = 1, desc = 'Splitting Dataset ')):
    random.shuffle(Feature)

    for Image in Feature[:200]:
        cv2.imwrite(os.path.join('Split Dataset', 'Validation', str(Class), f'{Count[Class]}.png'), Image)
        Count[Class] += 1

    Count[Class] = 0

    for Image in Feature[200:]:
        cv2.imwrite(os.path.join('Split Dataset', 'Train', str(Class), f'{Count[Class]}.png'), Image)
        Count[Class] += 1