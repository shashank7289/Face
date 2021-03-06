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
        self.folderName = [];

# Read landmark points from text files in directory
def readPoints(databaseLmarks):
    # Create an array of points.
    pointsArray = [];

    #List all files in the directory and read points from text files one by one
    dir = os.listdir(databaseLmarks)
    for item in dir :
#         if '11' in item[2:-4] or '12' in item[2:-4] or '13' in item[2:-4]:
            fileName = databaseLmarks + item
                
            #Create an array of points.
            points = [];            
            
            # Read points from filePath
            with open(os.path.join(databaseLmarks, fileName)) as file:
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
    name = [];
    
    #List all files in the directory and read points from text files one by one
    dir = os.listdir(databaseImages)
    for item in dir :
#         if '11' in item[2:-4] or '12' in item[2:-4] or '13' in item[2:-4]:
        if ".git" not in item:
            fileName = databaseImages + item
            name.append(item)

            # Read image found.
            img = cv2.imread(os.path.join(databaseImages,fileName));

            # Convert to floating point
            img = np.float32(img)/255.0;

            # Add to array of images
            imagesArray.append(img);
            
    face.folderName=name
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
    
    database = 'D:/Codes/TestData/Cropping/Frontalized_image_FEI_database/'
    databaseImages = database + '/images/'
    databaseLmarks = database + '/lmarks/'
    
    # Dimensions of output image
    w = 128;
    h = 128;

    # Read points for all images
    allPoints = readPoints(databaseLmarks);
    
    # Read all images
    images = readImages(databaseImages);
    
    # Eye corners
    eyeFactor = 0.2
    eyecornerDst = [(np.int(eyeFactor*w ), np.int(h/3)), (np.int((1-eyeFactor)*w), np.int(h/3))];

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
        
        databaseLmarks = database + '/results/'
        if not os.path.exists(databaseLmarks):
            os.makedirs(databaseLmarks)
         
        resultName = databaseLmarks + face.folderName[i]
        cv2.imwrite(resultName, morphed)
        
#         cv2.imshow("img",img)
#         cv2.waitKey(0)
