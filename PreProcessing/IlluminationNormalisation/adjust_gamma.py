# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 10:53:01 2016

@author: uid38420
"""

import os
import numpy as np
import cv2
from glob import glob

image = "testFace.jpg"
gamma = 1.5

def adjust_gamma(image, gamma):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

# load the original image
# database = 'D:/Codes/Git/AI_Bio_Face/11_OpenCV/Python/demo_with_gui_facial_expression/database/baslerfaces_demo/yOliver/*.jpg'
database = "C:/Users/uid38420/Downloads/VChithra/Trust/*.jpg"

img_names = glob(database)
for fn in img_names:
	original = cv2.imread(fn, 0)
	
	name = fn.rpartition('\\')
	print('processing...' + name[2])
	
	# apply gamma correction and show the images
	adjusted = adjust_gamma(original, gamma=gamma)
# 	cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
# 			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	
	resultPath = os.getcwd() + "/results/gicOnClahe/"
	if not os.path.exists(resultPath):
			os.makedirs(resultPath)
	resultName = resultPath + name[2]
# 	cv2.imwrite(resultName, np.hstack([original, adjusted]))
	cv2.imwrite(resultName,adjusted)