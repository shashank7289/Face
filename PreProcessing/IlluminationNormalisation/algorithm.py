'''
Created on Aug 26, 2016

@author: uid38420
'''
import numpy as np
import cv2

def preStretch(nimg):
    """
    from 'Applicability Of White-Balancing Algorithms to Restoring Faded Color Slides: An Empirical Evaluation'
    """
    nimg = nimg.transpose(2, 0, 1)
    nimg[0] = np.maximum(nimg[0] - nimg[0].min(), 0)
    nimg[1] = np.maximum(nimg[1] - nimg[1].min(), 0)
    nimg[2] = np.maximum(nimg[2] - nimg[2].min(), 0)
    return nimg.transpose(1, 2, 0)

def greyWorld(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0] * (mu_g / np.average(nimg[0])), 255)
    nimg[2] = np.minimum(nimg[2] * (mu_g / np.average(nimg[2])), 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def maxWhite(nimg):
    if nimg.dtype == np.uint8:
        brightest = float(2 ** 8)
    elif nimg.dtype == np.uint16:
        brightest = float(2 ** 16)
    elif nimg.dtype == np.uint32:
        brightest = float(2 ** 32)
    else:
        brightest == float(2 ** 8)
    nimg = nimg.transpose(2, 0, 1)
    nimg = nimg.astype(np.int32)
    nimg[0] = np.minimum(nimg[0] * (brightest / float(nimg[0].max())), 255)
    nimg[1] = np.minimum(nimg[1] * (brightest / float(nimg[1].max())), 255)
    nimg[2] = np.minimum(nimg[2] * (brightest / float(nimg[2].max())), 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def stretch(nimg):
    return maxWhite(preStretch(nimg))

def greyWorldStretch(nimg):
    return  greyWorld(stretch(nimg))

def stretchGreyWorld(nimg):
    return  stretch(greyWorld(nimg))

def retinex(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = nimg[1].max()
    nimg[0] = np.minimum(nimg[0] * (mu_g / float(nimg[0].max())), 255)
    nimg[2] = np.minimum(nimg[2] * (mu_g / float(nimg[2].max())), 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def retinexAdjust(nimg):
    """
    from 'Combining Gray World and Retinex Theory for Automatic White Balance in Digital Photography'
    """
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    sum_r = np.sum(nimg[0])
    sum_r2 = np.sum(nimg[0] ** 2)
    max_r = nimg[0].max()
    max_r2 = max_r ** 2
    sum_g = np.sum(nimg[1])
    max_g = nimg[1].max()
    coefficient = np.linalg.solve(np.array([[sum_r2, sum_r], [max_r2, max_r]]),
                                  np.array([sum_g, max_g]))
    nimg[0] = np.minimum((nimg[0] ** 2) * coefficient[0] + nimg[0] * coefficient[1], 255)
    sum_b = np.sum(nimg[2])
    sum_b2 = np.sum(nimg[2] ** 2)
    max_b = nimg[2].max()
    max_b2 = max_b ** 2
    coefficient = np.minimum(np.linalg.solve(np.array([[sum_b2, sum_b], [max_b2, max_b]]),
                                             np.array([sum_g, max_g])), 255)
    nimg[2] = (nimg[2] ** 2) * coefficient[0] + nimg[2] * coefficient[1]
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def retinexWithAdjust(nimg):
    return retinexAdjust(retinex(nimg))

def histogramEqualization(img1):
    equ = cv2.equalizeHist(img1)
    return equ
 
def clahe(img1):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img1)
    return cl1
 
def gammaCorrection(img1, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    
    # apply gamma correction using the lookup table
    adjusted = cv2.LUT(img1,table)
    return adjusted