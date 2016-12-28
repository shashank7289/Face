'''
Created on Nov 25, 2016

@author: uid38420
'''
import numpy as np

def faceOnly(img,y,h,x,w):
    faceImg = img[y:h,x:w]
    blankImage = np.ones((img.shape[0], img.shape[1], 3), np.uint8)
    blankImage[:,:] = 255
    blankImage[y:h,x:w] = faceImg
    return blankImage