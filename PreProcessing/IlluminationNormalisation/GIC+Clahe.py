'''
Created on Sep 17, 2016

@author: uid38420
'''
import os
import cv2
import numpy as np
from glob import glob
import algorithm as algo

database = "D:/Codes/Git/For everything else/AI_Bio_Face_db_YaleB/"
gamma = 1.5

dir = os.listdir(database) 
for item in dir :
    if ".git" not in item:
        folderName = database + item  + "/*.jpg"
    
        for fn in glob(folderName):
            img1 = cv2.imread(fn, 0)
        
            cl1 = algo.clahe(img1)
            adjusted = algo.gammaCorrection(cl1, gamma=gamma)
            cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            
            name = fn.rpartition('\\')
            print('processing...' + name[2])
            fileName = name[2].rpartition('.')
            resultPath = os.getcwd() + "\\" + "results" + "\\" + item + "\\"
            if not os.path.exists(resultPath):
                os.makedirs(resultPath)
            resultName = resultPath + name[2]
#             cv2.imwrite(resultName, np.hstack([cl1, adjusted]))
            cv2.imwrite(resultName, adjusted)
