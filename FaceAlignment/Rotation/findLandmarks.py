'''
Created on Nov 25, 2016

@author: uid38420
'''
import cv2
import dlib
import numpy as np

def lMarks(img,faces):
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