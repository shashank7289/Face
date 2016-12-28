'''
Created on Sep 28, 2016

@author: uid38420
'''
import cv2
import dlib
import os
import numpy as np
from glob import glob

faceCascade = cv2.CascadeClassifier('D:/Codes/Libraries/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
predictor_path = "D:/Codes/Libraries/dlib-19.1.0/predictor/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

database = 'D:/Codes/TestData/Cropping/Frontalized_image_FEI_database_OpenCV_only_chin_to_chin/11/'
fileName = database  + "*.jpg"

for fn in glob(fileName):
#     if '11' in fn[2:-4] or '12' in fn[2:-4] or '13' in fn[2:-4]:

        img = cv2.imread(fn)
        faces = faceCascade.detectMultiScale(img, 1.3, 5)
        
        txtName = fn.rpartition('\\')
        txtName = str(txtName[2])
        if txtName.endswith('.jpg'):
            txtName = txtName[:-4]
        if np.size(faces, 0) > 0:
            for (x,y,w,h) in faces:
                d = dlib.rectangle(x,y,x+w,y+h)
                lMark = np.zeros((68, 2))
                i = 0
                for p in predictor(img, d).parts():
                    cv2.circle(img,(p.x,p.y),1,(0,255,0), 2)
                    lMark = np.insert(lMark, i, [p.x, p.y], 0)
                    i = i+1
            
            lMarks = str(lMark)
            lMarks = lMarks.replace("[", "")
            lMarks = lMarks.replace("]", "")
            lMarks = lMarks.replace(".", "")
            lMarks = lMarks.replace("  ", " ")
            lMarks = lMarks.replace("0  0", "")
            
            textFile = open(database + "LMarks/" + txtName + ".txt", "w")
            textFile.write(lMarks)
            textFile.close()