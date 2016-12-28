'''
Created on Oct 5, 2016

@author: uid38420
'''
import cv2
import math
import os
from glob import glob

import numpy as np

class face:
    def __init__(self):
        self.folderName = imgName = [];

# Read landmark points from text files in directory
def readPoints(databaseLmarks):
    # Create an array of points.
    pointsArray = [];

    #List all files in the directory and read points from text files one by one
    for item in os.listdir(databaseLmarks):
        fileName = databaseLmarks + item + "/*.txt"
        for fn in glob(fileName):
            
            #Create an array of points.
            points = [];            
            
            # Read points from filePath
            with open(os.path.join(databaseLmarks, fn)) as file:
                for i, line in enumerate(file):
                    if i<=67:
                        x, y = line.split()
                        points.append((int(x), int(y)))
            
            # Store array of points
            pointsArray.append(points)
            
    return pointsArray;

# Read all jpg images in folder.
def readImages(databaseImages):
    
    #Create array of array of images.
    imagesArray = [];
    folderName = [];
    imgName = [];
    
    #List all files in the directory and read points from text files one by one
    for item in os.listdir(databaseImages):
        fileName = databaseImages + item + "/*.jpg"
        for fn in glob(fileName):
            folderName.append(item)
            name = fn.rpartition('\\')
            imgName.append(name[2])
    
            # Read image found.
            img = cv2.imread(os.path.join(databaseImages,fn));
    
            # Convert to floating point
            img = np.float32(img)/255.0;
    
            # Add to array of images
            imagesArray.append(img);
            
    face.folderName=folderName
    face.imgName=imgName
    return imagesArray;

# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.
def similarityTransform(inPoints, outPoints):
    s60 = math.sin(60*math.pi/180);
    c60 = math.cos(60*math.pi/180);  
  
    inPts = np.copy(inPoints).tolist();
    outPts = np.copy(outPoints).tolist();
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0];
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1];
    
    inPts.append([np.int(xin), np.int(yin)]);
    
    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0];
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1];
    
    outPts.append([np.int(xout), np.int(yout)]);
    
    tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False);
    
    return tform;

if __name__ == '__main__':
    
    databaseLmarks = "D:/Codes/TestData/Alignment/Yalelandmarks/"
    databaseImages = 'D:/Codes/TestData/IlluminationNormalisation/YaleB_GIC+Clahe/yale3/'
    
    # Dimensions of output image
    w = 128;
    h = 128;

    # Read points for all images
    allPoints = readPoints(databaseLmarks);
    
    # Read all images
    images = readImages(databaseImages);
    
    # Eye corners
    eyecornerDst = [(np.int(0.3*w ), np.int(h/3)), (np.int(0.7*w), np.int(h/3))];

    numImages = len(images)
    
    for i in xrange(0, numImages):

        points1 = allPoints[i];

        # Corners of the eye in input image
        eyecornerSrc  = [allPoints[i][36], allPoints[i][45]]
        
        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst)
        
        # Apply similarity transformation
        img = cv2.warpAffine(images[i], tform, (w,h));
        morphed = img * 255
        
        resultPath = "D:/Codes/TestData/Alignment/YaleResults/" + face.folderName[i] + '/'
        if not os.path.exists(resultPath):
            os.makedirs(resultPath)
        
        resultName = resultPath + face.imgName[i]
        cv2.imwrite(resultName, morphed)
        
#         cv2.imshow("img",img)
#         cv2.waitKey(0)
