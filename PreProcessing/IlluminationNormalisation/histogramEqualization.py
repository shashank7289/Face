# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 10:01:19 2016

@author: uid38420
"""
import os
import numpy as np
import cv2
from glob import glob

# load the image
database = 'D:/Codes/Git/AI_Bio_Face/11_OpenCV/Python/demo_with_gui_facial_expression/database/baslerfaces_demo/yOliver/*.jpg'
img_names = glob(database)
for fn in img_names:
    img1 = cv2.imread(fn, 0)
    
    # create a HE object
    equ = cv2.equalizeHist(img1)
    res = np.hstack((img1,equ)) #stacking images side-by-side
    
    name = fn.rpartition('\\')
    print('processing...' + name[2])
    
    resultPath = os.getcwd() + "/results/he/"
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    
    resultName = resultPath + name[2]
    cv2.imwrite(resultName, res)