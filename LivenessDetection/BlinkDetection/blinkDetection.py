'''
Created on Dec 5, 2016

@author: uid38420
'''
import cv2
import dlib
import winsound         # for sound  
import heapq

import numpy as np

class face:
    def __init__(self):
        self.leftEye = self.rightEye =  []
        self.x = 5
        self.leftArr = [0] * self.x
        self.rightArr = [0] * self.x
        self.counter = 0

    def fxn(self,val):
        self.leftArr[(self.counter%self.x)] = val
        self.rightArr[(self.counter%self.x)] = val
        self.counter = self.counter + 1
    
def faceParts(faceObj, lMarks):
    faceObj.leftEye = np.array(lMarks[36:42])
    faceObj.rightEye = np.array(lMarks[42:48])
    
def findLandmarks(img,faces):
    lMarks = np.zeros((68, 2))
    #find Landmarks
    for (x,y,w,h) in faces:
        d = dlib.rectangle(x,y,x+w,y+h)
        i = 0
        for p in predictor(img, d).parts():
            cv2.circle(img,(p.x,p.y),1,(0,255,0), 1)
            lMarks = np.insert(lMarks, i, [p.x, p.y], 0)
            i = i+1
    return lMarks
    
def distance(faceObj):
    #distance between upper and lower points of left eye
    leftEyeUpper = faceObj.leftEye[1]
    leftEyeLower = faceObj.leftEye[5]
    yDiffLeftEye = leftEyeLower[1] - leftEyeUpper[1]
    
    #distance between upper and lower points of right eye
    rightEyeUpper = faceObj.rightEye[1]
    rightEyeLower = faceObj.rightEye[5]
    yDiffRightEye = rightEyeLower[1] - rightEyeUpper[1]
    
    faceObj.fxn(yDiffLeftEye)
    faceObj.fxn(yDiffRightEye)
    speedFactor = 1.45 #detection performance decreases with increasing speed factor
    avgFactor = 1.5 #detection performance increases with reducing speed factor
    
    #condition to reject someone moving with speed
    if ((np.amax(faceObj.leftArr, 0) - np.amin(faceObj.leftArr, 0)) > (np.amax(faceObj.leftArr, 0)/speedFactor)) or (np.amax(faceObj.rightArr, 0) - np.amin(faceObj.rightArr, 0) > (np.amax(faceObj.rightArr, 0)/speedFactor)):
        leftAvg = np.average(heapq.nlargest(2,faceObj.leftArr))
        rightAvg = np.average(heapq.nlargest(2,faceObj.rightArr))
        
#         print ((np.amax(faceObj.leftArr, 0) - np.amin(faceObj.leftArr, 0)), '.........', (np.amax(faceObj.leftArr, 0)/speedFactor))
        print yDiffLeftEye, ".........", leftAvg, ".........", (leftAvg/avgFactor)
        
        #condition to include gradual changes in user position
        if yDiffLeftEye < (leftAvg/avgFactor) or yDiffRightEye < (rightAvg/avgFactor):
            winsound.Beep(1000, 250) # frequency, duration

if __name__ == '__main__' :
    faceCascade = cv2.CascadeClassifier('D:/Codes/Libraries/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    predictor_path = "D:/Codes/Libraries/dlib-19.1.0/predictor/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    fObj = face()
            
    cap = cv2.VideoCapture(0)
    
    while True:
        ret,img = cap.read()
        faces = faceCascade.detectMultiScale(img, 1.3, 5)
        
        if np.size(faces, 0) > 1:
            for (x,y,w,h) in faces:
                if h < 100:
                    pass
        else:
            for (x,y,w,h) in faces:
                if h > 100:
                    #determine face landmark points
                    lMarks = findLandmarks(img, faces)
                    
                    #determine points of eyes
                    faceParts(fObj,lMarks)
                    
                    #distance between upper and lower points of eyes
                    distance(fObj)
                    
        cv2.imshow('output',img)
        cv2.waitKey(1)