import os
import cv2
import numpy as np

import Extract_Digits
import Predict_Digits
import Solve_Sudoku

Path = 'Sudoku'
Images = sorted(os.listdir(Path), key = lambda x: int(os.path.splitext(x)[0]))

if not os.path.isdir('Solution'):
    os.mkdir('Solution')

for Image_Name in Images:
    Image = cv2.imread(os.path.join(Path, Image_Name))
    Image_List, Centres = Extract_Digits.Extract(Image)
    
    if Image_List:
        Grid = Predict_Digits.Predict(Image_List)
        Solution, Empty_Cells = Solve_Sudoku.Solve(Grid)

        if Solution:
            for i, j in Empty_Cells:
                origin = (Centres[i * 9 + j][0] - 15, Centres[i * 9 + j][1] + 15)
                Image = cv2.putText(Image, str(Solution[i][j]), origin, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 100, 0), 3, cv2.LINE_AA)

            cv2.imwrite(os.path.join('Solution', Image_Name), Image)
        else:
            print(f'Sudoku Invalid Or Recognized Incorrectly In {Image_Name}!')
    else:
        print(f'Sodoku Recognition Failed In {Image_Name}!')

cv2.destroyAllWindows()