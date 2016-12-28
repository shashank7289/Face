'''
Created on Sep 23, 2016

@author: uid38420
'''
import cv2
import dlib

import numpy as np


class face:
    def __init__(self):
        self.outline = self.eyeBrows = self.nose = self.leftEye = self.rightEye = self.mouth = []
    
def faceParts(lMarks):
    face.outline = np.array(lMarks[:17])
    face.eyeBrows = np.array(lMarks[17:27])
    face.nose = np.array(lMarks[27:36])
    face.leftEye = np.array(lMarks[36:42])
    face.rightEye = np.array(lMarks[42:48])
    face.mouth = np.array(lMarks[48:])
        
def frontalChk():
    ratio = score = 0.0
    
    #Condition 1 distance between outer eye point and edge face outline
    #difference of face outline and outermost point of left eye
    minPos = np.argmin(face.leftEye,axis=0)
    minX = minPos[0]
    minX = np.split(face.leftEye[minX],[1])
    outline = np.split(face.outline[0],[1])
    diff1 = np.subtract(outline,minX)
    diff1 = np.sqrt(np.square(diff1[0]) + np.square(diff1[1]))
       
    #difference of face outline and outermost point of right eye
    maxPos = np.argmax(face.rightEye,axis=0)
    max = maxPos[0]
    max = np.split(face.rightEye[max],[1])
    outline = np.split(face.outline[16],[1])
    diff2 = np.subtract(outline,max)
    diff2 = np.sqrt(np.square(diff2[0]) + np.square(diff2[1]))
       
    if (diff1 >= diff2):
        ratio = diff2/diff1
    else:
        ratio = diff1/diff2
    if (ratio >= 0.5):
        score = score  + 0.3;
      
#     print "Condition1: distance between outer eye point and edge face outline"
#     print 'diff1: ',diff1
#     print 'diff2: ',diff2
#     print 'ratio: ',ratio

    #Condition 2 distance between bottom nose point and corresponding point on the chin
    min = 0; max = 1000
    point = [0,0]
     
    noseCentre = np.split(face.nose[3],[1])
    outlineX = np.split(face.outline[:,0],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
    outlineY = np.split(face.outline[:,-1],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
     
    for i in xrange(17):
        #top point
        if((outlineY[i] <= (noseCentre[1]+2)) and (outlineY[i] >= min)):            
            min = outlineY[i]
            point[0] = i
        #bottom point
        elif((outlineY[i] >= (noseCentre[1]-2)) and (outlineY[i] <= max)):
            max = outlineY[i]
            point[1] = i
    
    diffX = noseCentre[0] - outlineX[point[0]]
    diffY = noseCentre[1] - outlineY[point[0]]
    diff1 = np.sqrt(np.square(diffX) + np.square(diffY))
     
    diffX = noseCentre[0] - outlineX[point[1]]
    diffY = noseCentre[1] - outlineY[point[1]]
    diff2 = np.sqrt(np.square(diffX) + np.square(diffY))
     
    if (diff1 >= diff2):
        ratio = diff2/diff1
    else:
        ratio = diff1/diff2
    if (ratio >= 0.5):
        score = score  + 0.3;
     
#     print "Condition2: distance between bottom nose point and corresponding point on the chin"
#     print 'diff1: ',diff1
#     print 'diff2: ',diff2
#     print 'ratio: ',ratio
    
    #Condition 3 distance between outer eyebrow point and extreme end chin points
    diff1 = np.subtract(face.outline[0],face.eyeBrows[0])
    diff1 = np.sqrt(np.square(diff1[0]) + np.square(diff1[1]))
     
    diff2 = np.subtract(face.outline[16],face.eyeBrows[9])
    diff2 = np.sqrt(np.square(diff2[0]) + np.square(diff2[1]))
     
    if (diff1 >= diff2):
        ratio = diff2/diff1
    else:
        ratio = diff1/diff2
    if (ratio >= 0.5):
        score = score  + 0.3;
           
#     print "Condition3: distance between outer eyebrow point and extreme end chin points"
#     print 'diff1: ',diff1
#     print 'diff2: ',diff2
#     print 'ratio: ',ratio
    
    #decision making
    decision = ""
    if (score> 0.5):
        decision = "frontal"
    else:
        decision = "non-frontal"
    return decision

if __name__ == '__main__' :
    faceCascade = cv2.CascadeClassifier('D:/Codes/Libraries/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    predictor_path = "D:/Codes/Libraries/dlib-19.1.0/predictor/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    cap = cv2.VideoCapture(0)
    
    # database = 'D:/Codes/TestData/IlluminationNormalisation/YaleB_GIC+Clahe/yaleB12/0.jpg'
    
    while True:
        ret,img = cap.read()
    #     img = cv2.imread(database)
        
        #defining ROI for faster processing
        shape = np.shape(img)
        startCol = 0.2*(shape[1])
        startCol = int(startCol)
        endCol = shape[1] - startCol
        endCol = int(endCol)
        img = img[0:endCol,startCol:endCol]
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        
        #detecting face
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        
        #determining face landmark points
        for (x,y,w,h) in faces:
            d = dlib.rectangle(x,y,x+w,y+h)
            
            lMarks = np.zeros((68, 2))
            i = 0
            for p in predictor(gray, d).parts():
                cv2.circle(img,(p.x,p.y),1,(0,255,0), 2)
                lMarks = np.insert(lMarks, i, [p.x, p.y], 0)
                i = i+1
                            
            faceParts(lMarks)
            decision = frontalChk()
            cv2.putText(img, decision, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        cv2.imshow('output',img)
        cv2.waitKey(1)