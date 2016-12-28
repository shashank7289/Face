'''
Created on Nov 18, 2016

@author: uid38420
'''
import os
import cv2
import dlib
import numpy as np

from glob import glob
from skimage.transform import rotate

class face:
    def __init__(self):
        self.leftEye = self.rightEye = 0

def videoCapture():
    cap = cv2.VideoCapture(0)
    ret,img = cap.read()
    return ret,img
        
def detectFace(img):
    faceCascade = cv2.CascadeClassifier('D:/Codes/Libraries/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(img, 1.3, 5)
    return faces
        
def findLandmarks(img,faces):
    predictor_path = "D:/Codes/Libraries/dlib-19.1.0/predictor/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    lMarks = np.zeros((68, 2))
    #find Landmarks
    for (x,y,w,h) in faces:
        d = dlib.rectangle(x,y,x+w,y+h)
        i = 0
        for p in predictor(img, d).parts():
#             cv2.circle(img,(p.x,p.y),1,(0,255,0), 2)
            lMarks = np.insert(lMarks, i, [p.x, p.y], 0)
            i = i+1
    return lMarks

def faceParts(lMarks):
    face.leftEye = lMarks[36]
    face.rightEye = lMarks[42]
    
def faceOnly(img,faces):
    faceImg = 0
    for (x,y,w,h) in faces:
#         faceImg = img[y-50:y+h+50,x-50:x+w+50]
        multiplier = 0.35
        faceImg = img[y-multiplier*h:y+h+multiplier*h,x-multiplier*h:x+w+multiplier*h]
    return faceImg

def rollAngle(p1, p2):
    if (p1[1]> p2[1]):
        p1[1] = p1[1]-p2[1]
        p2[1] = p2[1]-p2[1]
    elif (p2[1]> p1[1]):
        p2[1] = p2[1]-p1[1]
        p1[1] = p1[1]-p1[1]
        
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    angle = np.arctan2(yDiff,xDiff) * (180 / np.pi)

    if angle > 180:
        angle = 360-angle
        angle = -angle
    return angle

if __name__ == '__main__':
    
    # load the original image
    database = "D:/Codes/TestData/Rotation/wholeImage/"
    resultPath = database + "/results/"
    if not os.path.exists(resultPath):
            os.makedirs(resultPath)
    img_names = glob(database + '*.jpg')
    for fn in img_names:
        img = cv2.imread(fn)
        name = fn.rpartition('\\')
        resultName = resultPath + name[2]
        
        #detect face
        faces = detectFace(img)
        
        #determine face landmark points
        lMarks = findLandmarks(img, faces)
        
        #determine corner of eyes
        faceParts(lMarks)
            
        #determine roll angle
        angle = rollAngle(face.leftEye, face.rightEye)
        
        #rotate image
        faceImg = faceOnly(img,faces)
        rotatedImg = rotate(faceImg, angle)
            
        cv2.putText(img, np.str(round(angle,2)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        cv2.imshow('imput',img)
        cv2.imshow('output',rotatedImg)
#         cv2.imwrite(resultName,rotatedImg*255)
        cv2.waitKey(0)