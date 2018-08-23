'''
    UTK data preprocessor
    convert images to mat, extract age, gender and ethnicity information
    save the training data to .mat file
'''

import glob
import os, random
import pandas as pd
import scipy.io
import numpy as np
import cv2
from tqdm import tqdm

def create_trainning_set(dir_path):
    X_train, y_each_age, y_gender, y_race = [], [], [], []

    img_list = os.listdir(dir_path)
    random.shuffle(img_list)
    sz = 64

    for filename in tqdm(img_list):
        # read image as a matrix with open CV3
        img = cv2.imread(dir_path + '/' + filename)
        # print(img.shape)
        X_train.append(cv2.resize(src=img, dsize=(sz, sz)))

        # split file name to age[0-116], gender[m=0, f=1], race[0-4] and timestamp - see remarks [1]
        img_info = [int(i) for i in filename.split(".")[0].split("_")]
        y_each_age.append(img_info[0])
        y_gender.append(img_info[1])
        y_race.append(img_info[2])
    bins = [0, 4, 8, 15, 25, 38, 48, 60, 150]
    y_age = pd.cut(x=y_each_age, bins=bins, labels=False, right=False)

    unique, counts = np.unique(y_gender, return_counts=True)
    print( dict(zip(unique, counts)) )

    # Save a dictionary of names and arrays into a MATLAB-style .mat file.
    scipy.io.savemat("phase_00/UTK_train/X_Y_64",
                     {'X': X_train,
                      'y_age': y_age,
                      'y_gender': y_gender,
                      'y_race': y_race})


def main():
    UTK_path = "data/UTKFace"
    create_trainning_set(UTK_path)


if __name__ == '__main__':
    main()

'''
Remarks:
[1] 
    The labels of each face image is embedded in the file name, 
    formated like [age]_[gender]_[race]_[date&time].jpg

    [age] is an integer from 0 to 116, indicating the age
    [gender] is either 0 (male) or 1 (female)
    [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
    [date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace
'''