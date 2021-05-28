import os
import cv2
import numpy as np

from tensorflow.keras.models import load_model

model = load_model(os.path.join('Model', 'best_val_loss.hdf5'))

def Read(Image_List):
    Grid = [[0] * 9 for _ in range(9)]

    for index, Image in enumerate(Image_List):
        gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)

        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 27, 6)
        thresh = cv2.bitwise_not(thresh, mask = None)  

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours.sort(key = cv2.contourArea, reverse = True)

        if len(contours):
            x, y, w, h = cv2.boundingRect(contours[0])

            if w * h >= 9 and w >= 3 and h >= 3:
                size = max(w, h)

                square_fit = np.zeros((size, size, 3), np.uint8)
                square_fit.fill(255)

                square_fit[int((size - h) / 2):int((size + h) / 2), int((size - w) / 2):int((size + w) / 2)] = thresh[y:y + h, x:x + w]
                square_fit = cv2.resize(square_fit, (20, 20), interpolation = cv2.INTER_AREA)

                x = square_fit.reshape(-1, 20, 20, 1) / 255.0
                y = model.predict(x)[0]
                
                digit = np.argmax(y)
                Grid[int(index / 9)][index % 9] = digit

    return Grid