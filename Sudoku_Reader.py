import os
import cv2
import numpy as np

from tensorflow.keras.models import load_model

model = load_model(os.path.join('Model', 'best_val_loss.hdf5'))

def Read(Image_List):
    Grid = [[0] * 9 for _ in range(9)]

    for i, img in enumerate(Image_List):
        blur = cv2.medianBlur(img, 5)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 27, 6)
        thresh = cv2.bitwise_not(thresh, mask = None)  

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours.sort(key = cv2.contourArea, reverse = True)

        if len(contours):
            x, y, w, h = cv2.boundingRect(contours[0])

            if w * h >= 9 and w >= 3 and h >= 3:
                thresh = thresh[y:y + h, x:x + w]
                thresh = cv2.resize(thresh, (20, 20), interpolation = cv2.INTER_AREA)

                x = thresh.reshape(-1, 20, 20, 1) / 255.0
                y = model.predict(x)[0]
                
                digit = np.argmax(y)
                Grid[int(i / 9)][i % 9] = digit

    return Grid

