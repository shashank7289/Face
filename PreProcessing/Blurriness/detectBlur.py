'''
Created on Aug 31, 2016

@author: uid38420
'''
import cv2
from glob import glob

# database = "D:/Codes/Python/PreProcessing/Blurriness/images/*.png"
database = 'D:/Codes/Git/AI_Bio_Face/11_OpenCV/Python/demo_with_gui_facial_expression/database/baslerfaces_demo/yOliver/*.jpg'

threshold = 100
    
imgNames = glob(database)
for fn in imgNames:
    
    image = cv2.imread(fn)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    text = "Not Blurry"
    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if var < threshold:
        text = "Blurry"
 
    cv2.putText(image, "{}: {:.2f}".format(text, var), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    key = cv2.waitKey(0)