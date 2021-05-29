# Sudoku
Sudoku Recognition And Its Solution Using Image Processing And Constraint Programming And Backtracking

This work allows optical recognition and solution of the sudoku. Image processing techniques enable its detection and extraction of the digits. The extracted digits are then recognized using a convolutional neural network combined with a deep neural network. Constraint programming combined with backtracking enables the faster solution of the sudoku, no matter how hard it is.

Sample images used for recognition and solution can be found in [Sudoku](https://github.com/milind-prajapat/Sudoku/tree/main/Sudoku) directory of the repository.

## Instructions To Use
To perform recognition and solution of the sudoku, accumulate the images in a directory and then provide the complete path to the directory in [Main.py](https://github.com/milind-prajapat/Sudoku/blob/main/Main.py), images with solved sudoku will be saved in the [Solution](https://github.com/milind-prajapat/Sudoku/tree/main/Solution) directory. You can also use live video feed through the webcam using [Webcam.py](https://github.com/milind-prajapat/Sudoku/blob/main/Webcam.py). Currently, the code is configured to use the phone's camera for better resolution using [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) android application.

You can then either run the code directly on visual studio using [Sudoku.sln](https://github.com/milind-prajapat/Sudoku/blob/main/Sudoku.sln) or can run individual python files.

## Structure

## Digit Extraction And Solution

## Model Performance

**Table.** Classification Report on Validation Data 

|  | accuracy_score | precision_score | recall_score | f1_score|
| --- | :---: | :---: | :---: | ---: |
| Model    |      0.993     |      0.9931     |   0.993  |  0.9936 |

## Features

## Limitations

## References
1. [Printed English Characters](https://drive.google.com/file/d/1UYUyG0Z_33_IiMjOhy48w_ek38j-68dx/view?usp=sharing)
2. [Webcam Demonstration](https://drive.google.com/file/d/1NDWFiYbbc5GfrwAFLoWxFhMlrgRaR47R/view?usp=sharing)
