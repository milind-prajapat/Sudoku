# Sudoku
Sudoku Recognition And Its Solution Using Image Processing And Constraint Programming And Backtracking

This work allows optical recognition and solution of the sudoku. Image processing techniques enable its detection and extraction of digits. The extracted digits are then recognized using a convolutional neural network combined with a deep neural network. Constraint programming combined with backtracking enables the faster solution of the sudoku, no matter how hard it is.

Sample images used for recognition and solution can be found in [Sudoku](https://github.com/milind-prajapat/Sudoku/tree/main/Sudoku) directory of the repository.

## Instructions To Use
To perform recognition and solution of the sudoku, accumulate the images in a directory and then provide the complete path to the directory in [Main.py](https://github.com/milind-prajapat/Sudoku/blob/main/Main.py), images with solved sudoku will be saved in the [Solution](https://github.com/milind-prajapat/Sudoku/tree/main/Solution) directory. You can also use live video feed through the webcam using [Webcam.py](https://github.com/milind-prajapat/Sudoku/blob/main/Webcam.py). Currently, the code is configured to use the phone's camera for better resolution using [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) android application.

You can then either run the code directly on visual studio using [Sudoku.sln](https://github.com/milind-prajapat/Sudoku/blob/main/Sudoku.sln) or can run individual python files.

## Structure
* [Extract_Digits.py](https://github.com/milind-prajapat/Sudoku/blob/main/Extract_Digits.py) is used for recognition of sudoku and extraction of digits from it
* [Predict_Digits.py](https://github.com/milind-prajapat/Sudoku/blob/main/Predict_Digits.py) is used for taking the prediction of the extracted digits
* [Solve_Sudoku.py](https://github.com/milind-prajapat/Sudoku/blob/main/Solve_Sudoku.py) is used for solving the recognized sudoku
* [Split_Dataset.py](https://github.com/milind-prajapat/Sudoku/blob/main/Split_Dataset.py) is used for splitting the dataset into training and validation sets
* [Model.py](https://github.com/milind-prajapat/Sudoku/blob/main/Model.py) is used for training convolution neural network
* [Evaluate_Model.py](https://github.com/milind-prajapat/Sudoku/blob/main/Evaluate_Model.py) is used for evaluating the convolution neural network

## Digit Extraction And Solution

## Model Performance

**Table.** Classification Report on Validation Data 

|  | accuracy_score | precision_score | recall_score | f1_score|
| --- | :---: | :---: | :---: | ---: |
| Model    |      0.993     |      0.9931     |   0.993  |  0.9936 |

## Features
1. **Image processing** enables the recognition of the sudoku
2. **Data augmentation** using image data generator class, rotated, shifted, sheared and zoomed
3. **Convolution neural network** combined with a deep neural network
4. **Constraint programming** combined with backtracking enables faster solutions to the sudoku

## Limitations
1. Sudoku with excess noise and highly slanted sudoku might fail to recognize
2. Sudoku in images with the lesser quality might also fail to recognize
3. Incorrect prediction of the extracted digits might cause the formation of incorrect sudoku or sudoku having no solution

## References
1. [Printed English Characters](https://drive.google.com/file/d/1UYUyG0Z_33_IiMjOhy48w_ek38j-68dx/view?usp=sharing)
2. [Webcam Demonstration](https://drive.google.com/file/d/1NDWFiYbbc5GfrwAFLoWxFhMlrgRaR47R/view?usp=sharing)
