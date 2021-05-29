import cv2
import numpy as np

import Sudoku_Grabber
import Sudoku_Reader
import Sudoku_Solver

Path = 'Printed_Sudoku'
Images = sorted(os.listdir(Path), key = lambda x: int(os.path.splitext(x)[0]))
    
for Image_Name in Images:
    Image = cv2.imread(os.path.join(Path, Image_Name)
    Image_List, Centres = Sudoku_Grabber.Grab(Image)
    
    if Image_List:
        Grid = Sudoku_Reader.Read(Image_List)
        Solution, Empty_Cells = Sudoku_Solver.Solve(Grid)

        if Solution:
            for i, j in Empty_Cells:
                origin = (Centres[i * 9 + j][0] - 15, Centres[i * 9 + j][1] + 15)
                Image = cv2.putText(Image, str(Solution[i][j]), origin, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 100, 0), 3, cv2.LINE_AA)

            cv2.imshow('Solution', Image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f'Sudoku Invalid Or Recognized Incorrectly In {Image_Name}!')
    else:
        print(f'Sodoku Recognition Failed In {Image_Name}!')