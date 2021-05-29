import cv2
import numpy as np

import Sudoku_Grabber
import Sudoku_Reader
import Sudoku_Solver

from urllib import request

Solved = False
URL = r'http://192.168.43.1:8080/shot.jpg'

while True:
    Image = np.array(bytearray(request.urlopen(URL).read()), dtype = np.uint8)
    Image = cv2.imdecode(Image, -1)
    Image = Image[200:1000, 200:1000] # For (1920, 1080) Resolution
    
    Image_List, Centres = Sudoku_Grabber.Grab(Image)
    
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
        Image = cv2.putText(Image, str(Solution[i][j]), origin, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 100, 0), 3, cv2.LINE_AA)

    cv2.imshow('Live Video Feed', Image)
    
    if cv2.waitKey(1) == 27:
        break
        
cv2.destroyAllWindows()