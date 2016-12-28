'''
Created on Nov 25, 2016

@author: uid38420
'''

def face(img,faces):
    faceImg = 0
    for (x,y,w,h) in faces:
        multiplier = 0.35
        faceImg = img[y-multiplier*h:y+h+multiplier*h,x-multiplier*h:x+w+multiplier*h]
    return faceImg