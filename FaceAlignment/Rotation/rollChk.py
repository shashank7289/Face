'''
Created on Nov 18, 2016

@author: uid38420
'''
import cv2
import dlib
import numpy as np

class face:
    def __init__(self):
        self.leftEye = self.rightEye = 0

def faceParts(lMarks):
    face.leftEye = lMarks[36]
    face.rightEye = lMarks[42]
    
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

if __name__ == '__main__' :
    faceCascade = cv2.CascadeClassifier('D:/Codes/Libraries/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    predictor_path = "D:/Codes/Libraries/dlib-19.1.0/predictor/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret,img = cap.read()
    
        #detecting face
        faces = faceCascade.detectMultiScale(img, 1.3, 5)
        
        #determining face landmark points
        for (x,y,w,h) in faces:
            
            #find landmarks
            d = dlib.rectangle(x,y,x+w,y+h)
            lMarks = np.zeros((68, 2))
            i = 0
            for p in predictor(img, d).parts():
#                 cv2.circle(img,(p.x,p.y),1,(0,255,0), 2)
                lMarks = np.insert(lMarks, i, [p.x, p.y], 0)
                i = i+1
            faceParts(lMarks)
            
            #Rotate image
            angle = rollAngle(face.leftEye, face.rightEye)
            
            cv2.putText(img, np.str(round(angle,2)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
            cv2.imshow('original',img)
            
        cv2.waitKey(1)