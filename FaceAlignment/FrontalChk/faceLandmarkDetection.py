'''
Created on Sep 21, 2016

@author: uid38420
'''
import cv2
import dlib
import numpy as np

faceCascade = cv2.CascadeClassifier('D:/Codes/Libraries/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
predictor_path = "D:/Codes/Libraries/dlib-19.1.0/predictor/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
        
cap = cv2.VideoCapture(0)

while True:
    ret,img = cap.read()
    faces = faceCascade.detectMultiScale(img, 1.3, 5)
    
    for (x,y,w,h) in faces:
        d = dlib.rectangle(x,y,x+w,y+h)
            
        lMarks = np.zeros((68, 2))
        i = 0
        for p in predictor(img, d).parts():
            cv2.circle(img,(p.x,p.y),1,(0,255,0), 2)
            lMarks = np.insert(lMarks, i, [p.x, p.y], 0)
            i = i+1
            print i
            print p.x,p.y
            print "-------------"
            cv2.imshow('output',img)
            cv2.waitKey(0)