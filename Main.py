import cv2
import numpy as np

import Sudoku_Grabber
import Sudoku_Reader
import Sudoku_Solver

from urllib import request

URL = 'http://192.168.43.1:8080/shot.jpg'

Solved = False
while True:
    img_arr = np.array(bytearray(request.urlopen(URL).read()), dtype = np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = img[200:1000, 200:1000] # For (1920, 1080) Resolution
            
    Image_List, Centres = Sudoku_Grabber.Grab(img)

    if not Image_List:
        Empty_Cells = []
        Solved = False
    elif not Solved:
        Grid = Sudoku_Reader.Read(Image_List)
        Solution, Empty_Cells = Sudoku_Solver.Solve(Grid)

        if Solution:
            Solved = True
        else:
            Empty_Cells = []

    for i, j in Empty_Cells:
        origin = (Centres[i * 9 + j][0] - 15, Centres[i * 9 + j][1] + 15)
        img = cv2.putText(img, str(Solution[i][j]), origin, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Live Video Feed', img)
    
    if cv2.waitKey(1) == 27:
        break
        
cv2.destroyAllWindows()