import cv2
import math
import numpy as np

def Grab(Image):
    gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_ERODE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    for i in range(morph.shape[0]):
        for j in range(morph.shape[1]):
            if not morph[i][j]:
                morph[i][j] = 1
    
    div = gray / morph
    gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 150, 150, apertureSize = 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not len(contours):
        return None, None

    contours.sort(key = cv2.contourArea, reverse = True)

    mask = np.zeros(gray.shape[:2], np.uint8)
    cv2.drawContours(mask, [contours[0]], 0, 255, -1)
    
    sudoku = cv2.bitwise_and(gray, gray, mask = mask)

    sobelx = cv2.Sobel(sudoku, cv2.CV_16S, 1, 0)
    sobely = cv2.Sobel(sudoku, cv2.CV_16S, 0, 1)

    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)

    _, threshx = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, threshy = cv2.threshold(sobely, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
    kernely = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))

    threshx = cv2.morphologyEx(threshx, cv2.MORPH_DILATE, kernelx, iterations = 1)
    threshy = cv2.morphologyEx(threshy, cv2.MORPH_DILATE, kernely, iterations = 1)

    contours, _ = cv2.findContours(threshx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)

        if h / w > 5:
            cv2.drawContours(threshx, [contour], 0, 255, -1)
        else:
            cv2.drawContours(threshx, [contour], 0, 0, -1)

    contours, _ = cv2.findContours(threshy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        
        if w / h > 5:
            cv2.drawContours(threshy, [contour], 0, 255, -1)
        else:
            cv2.drawContours(threshy, [contour], 0, 0, -1)

    points = cv2.bitwise_and(threshx, threshy)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    points = cv2.morphologyEx(points, cv2.MORPH_DILATE, kernel, iterations = 1)

    contours, _ = cv2.findContours(points, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    Length = len(contours)
    
    if Length < 100:
        return None, None

    contours.sort(key = cv2.contourArea, reverse = True)
    contours = contours[:100]

    temp = []
    Length = 100

    for contour in contours:
        m = cv2.moments(contour)
        x = int(m['m10'] / m['m00'])
        y = int(m['m01'] / m['m00'])

        temp.append([x, y])

    temp.sort(key = lambda x : x[1])

    Points = []

    cols = int(math.sqrt(Length))

    for i in range(0, Length, cols):
        List = temp[i:i + 10][:]
        List.sort(key = lambda x : x[0])

        Points.extend(List)

    Points = np.float32(Points)
    Rows = np.array(Points, np.float32).reshape(cols, cols, 2)

    size = (45 * cols, 45 * cols)
    pixels = int(45 * cols / (cols - 1))
    output = np.zeros((45 * cols, 45 * cols, 3), np.uint8)

    Centre = []
    Image_List = []

    for i in range(Length):
        x = int(i / 10)
        y = i % 10

        if all([x != 9, y != 9]):
            src = Rows[x:x + 2, y:y + 2, :].reshape(4, 2)
            dst = np.array([[y * pixels, x * pixels], [(y + 1) * pixels - 1, x * pixels],[y * pixels, (x + 1) * 50 - 1],[(y + 1) * pixels - 1, (x + 1) * pixels - 1]], np.float32)

            M = cv2.getPerspectiveTransform(src, dst)
            warp = cv2.warpPerspective(Image, M, size)[x * pixels:(x + 1) * pixels - 1, y * pixels:(y + 1) * pixels - 1]

            cX, cY = 0, 0
            for i in range(4):
                cX += src[i][0]
                cY += src[i][1]
                
            Centre.append([int(cX / 4), int(cY / 4)])
            Image_List.append(warp.copy()[5:-5, 5:-5, :])
            output[x * pixels: (x + 1) * pixels - 1, y * pixels:(y + 1) * pixels - 1] = warp.copy()

    return Image_List, Centre
