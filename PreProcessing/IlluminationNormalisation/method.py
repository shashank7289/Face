'''
Created on Aug 29, 2016

@author: uid38420
'''

import os
import cv2
import numpy as np
import algorithm as algo
from PIL import Image
from util import from_pil, to_pil
from glob import glob

def greyStretch(database, case):
    imgNames = glob(database)
    for fn in imgNames:
        img1 = Image.open(fn)
#         img1 = Image.open("D:/Codes/Python/ColorCorrect/zlatan.jpg")
    
        name = fn.rpartition('\\')
        print('processing...' + name[2])
        result = to_pil(algo.greyWorldStretch(from_pil(img1)))
        
        fileName = name[2].rpartition('.')
        resultPath = os.getcwd() + "\\" + "results" + "\\" + case + "\\"
#         resultPath = os.getcwd() + "\\" + "results" + "\\" + fileName[0] + "\\" + case + "\\"
        if not os.path.exists(resultPath):
            os.makedirs(resultPath)
        resultName = resultPath + name[2]
        result.save(resultName)
        
def stretchGrey(database, case):
    imgNames = glob(database)
    for fn in imgNames:
        img1 = Image.open(fn)
    
        name = fn.rpartition('\\')
        print('processing...' + name[2])
        result = to_pil(algo.stretchGreyWorld(from_pil(img1)))
        
        fileName = name[2].rpartition('.')
        resultPath = os.getcwd() + "\\" + "results" + "\\" + case + "\\"
#         resultPath = os.getcwd() + "\\" + "results" + "\\" + fileName[0] + "\\" + case + "\\"
        if not os.path.exists(resultPath):
            os.makedirs(resultPath)
        resultName = resultPath + name[2]
        result.save(resultName)
        
def retinexWithAdjust(database, case):
    imgNames = glob(database)
    for fn in imgNames:
        img1 = Image.open(fn)
    
        name = fn.rpartition('\\')
        print('processing...' + name[2])
        result = to_pil(algo.retinexWithAdjust(from_pil(img1)))
        
        fileName = name[2].rpartition('.')
        resultPath = os.getcwd() + "\\" + "results" + "\\" + case + "\\"
#         resultPath = os.getcwd() + "\\" + "results" + "\\" + fileName[0] + "\\" + case + "\\"
        if not os.path.exists(resultPath):
            os.makedirs(resultPath)
        resultName = resultPath + name[2]
        result.save(resultName)

def histogramEqualization(database, case):
    imgNames = glob(database)
    for fn in imgNames:
        img1 = cv2.imread(fn, 0)
        
        equ = algo.histogramEqualization(img1)
        res = np.hstack((img1,equ)) #stacking images side-by-side
        
        name = fn.rpartition('\\')
        print('processing...' + name[2])
        fileName = name[2].rpartition('.')
        resultPath = os.getcwd() + "\\" + "results" + "\\" + case + "\\"
#         resultPath = os.getcwd() + "\\" + "results" + "\\" + fileName[0] + "\\" + case + "\\"
        if not os.path.exists(resultPath):
            os.makedirs(resultPath)
        resultName = resultPath + name[2]
        cv2.imwrite(resultName, res)
            
def clahe(database, case):
    imgNames = glob(database)
    for fn in imgNames:
        img1 = cv2.imread(fn, 0)
        
        cl1 = algo.clahe(img1)
        res = np.hstack((img1,cl1)) #stacking images side-by-side
        
        name = fn.rpartition('\\')
        print('processing...' + name[2])
        fileName = name[2].rpartition('.')
        resultPath = os.getcwd() + "\\" + "results" + "\\" + case + "\\"
#         resultPath = os.getcwd() + "\\" + "results" + "\\" + fileName[0] + "\\" + case + "\\"
        if not os.path.exists(resultPath):
            os.makedirs(resultPath)
        resultName = resultPath + name[2]
        cv2.imwrite(resultName, res)
      
def gammaCorrection(database, case, gamma):
    imgNames = glob(database)
    for fn in imgNames:
        img1 = cv2.imread(fn, 0)
        
        adjusted = algo.gammaCorrection(img1, gamma=gamma)
        cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        name = fn.rpartition('\\')
        print('processing...' + name[2])
        fileName = name[2].rpartition('.')
        resultPath = os.getcwd() + "\\" + "results" + "\\" + case + "\\"
#         resultPath = os.getcwd() + "\\" + "results" + "\\" + fileName[0] + "\\" + case + "\\"
        if not os.path.exists(resultPath):
            os.makedirs(resultPath)
        resultName = resultPath + name[2]
        cv2.imwrite(resultName, np.hstack([img1, adjusted]))
        
def gammaCorrectionClahe(database, case, gamma):
    imgNames = glob(database)
    for fn in imgNames:
        img1 = cv2.imread(fn, 0)
        
        cl1 = algo.clahe(img1)
        adjusted = algo.gammaCorrection(cl1, gamma=gamma)
        cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        name = fn.rpartition('\\')
        print('processing...' + name[2])
        fileName = name[2].rpartition('.')
        resultPath = os.getcwd() + "\\" + "results" + "\\" + case + "\\"
#         resultPath = os.getcwd() + "\\" + "results" + "\\" + fileName[0] + "\\" + case + "\\"
        if not os.path.exists(resultPath):
            os.makedirs(resultPath)
        resultName = resultPath + name[2]
        cv2.imwrite(resultName, np.hstack([cl1, adjusted]))
        
        
        
        
    