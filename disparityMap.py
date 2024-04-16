import cv2
import numpy as np

import utils
import constant as c

def distanceZframe(disparity):
    x,y = disparity.shape
    disparity = disparity/16.0

    #Estimate a main disparity
    disp0 = disparity[disparity>0]
    d = np.mean(disp0) 
    
    #Estimate a most frequent disparity
    #d = max(set(disp0), key = disp0.count)

    #Estimate a median disparity
    #d = np.median(disp0) 

    #Determine the distance in mm
    z=(c.B*c.F)/(d + 1e-5) #mm
    #convert in m
    z_m = z/1000 #m
    
    return z_m, d

def computeDisparityMap(frameL, frameR, numDisp=128, blockSize=33, show=False):
    #compute central frames
    center = frameL.shape
    centerY = int(center[0]/2)
    centerX = int(center[1]/2)
    interval = 50

    #Get image center box
    imgL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    #initialize stereo disparity
    stereoMatcher = cv2.StereoBM_create(numDisparities=numDisp, blockSize=blockSize)
    
    #compute disparity map: gray or colour
    disparity = stereoMatcher.compute(imgL, imgR)
    disparity = disparity[centerY-interval:centerY+interval, centerX-interval:centerX+interval]

    #print("Disparity range:{ ", disparity.min()," , ",disparity.max()," }")

    #show disparity frame 
    if show:
        utils.imshow("Disparity","Disparity map", disparity)

    return disparity