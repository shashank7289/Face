'''
Created on Sep 2, 2016

@author: uid38420
'''
import os
import cv2
import numpy as np
from glob import glob

database = "D:/Codes/Python/PreProcessing/IlluminationNormalisation/results/YaleB/"

def detectBlur(image):
    threshold = 100
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # compute the Laplacian of the image and return the focus measure (variance of Laplacian)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    text = "Not Blurry"

    if var < threshold:
        text = "Blurry"
    cv2.putText(image, "{}: {:.2f}".format(text, var), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return image

# generating kernels
#Sharpening
kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#Excessive Sharpening
kernel_sharpen_2 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
#Edge Enhancement
kernel_sharpen_3 = np.array([[-1, -1, -1, -1, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, 2, 8, 2, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, -1, -1, -1, -1]]) / 8.0

dir = os.listdir(database) 
for item in dir :
    if ".git" not in item:
        fileName = database + item  + "/*.jpg"
        
        for fn in glob(fileName):
            img1 = cv2.imread(fn)
            img2 = cv2.imread(fn)
            
            #detect blur
            blurriness = detectBlur(img2)
            
            name = fn.rpartition('\\')
            print('processing...' + name[2])
            
            # applying different kernels to the input image
            output_1 = cv2.filter2D(img1, -1, kernel_sharpen_1)
            output_2 = cv2.filter2D(img1, -1, kernel_sharpen_2)
            output_3 = cv2.filter2D(img1, -1, kernel_sharpen_3)
            output_4 = cv2.filter2D(output_1, -1, kernel_sharpen_3) #sharpen+edge
            
            #see all results simultaneously
#             result = np.hstack((blurriness, output_1,output_2, output_3, output_4))
#             cv2.imshow("res", result)
#             cv2.waitKey(0)
#             fileName = name[2].rpartition('.')
#             resultPath = os.getcwd() + "\\" + "results" + "\\" + item + "\\"
#             if not os.path.exists(resultPath):
#                 os.makedirs(resultPath)
#             resultName = resultPath + name[2]
#             cv2.imwrite(resultName, result)
            
            #save results
            #result - Sharpening
            resultPath = os.getcwd() + "\\" + "results" + "\\" + item + "\\Sharpening\\"
            if not os.path.exists(resultPath):
                os.makedirs(resultPath)
            resultName = resultPath + name[2]
            cv2.imwrite(resultName, output_1)
            
            #result - Edge Enhancement
            resultPath = os.getcwd() + "\\" + "results" + "\\" + item + "\\Edge Enhancement\\"
            if not os.path.exists(resultPath):
                os.makedirs(resultPath)
            resultName = resultPath + name[2]
            cv2.imwrite(resultName, output_3)
            
            #result - Sharpen+Edge
            resultPath = os.getcwd() + "\\" + "results" + "\\" + item + "\\Sharpen+Edge\\"
            if not os.path.exists(resultPath):
                os.makedirs(resultPath)
            resultName = resultPath + name[2]
            cv2.imwrite(resultName, output_4)