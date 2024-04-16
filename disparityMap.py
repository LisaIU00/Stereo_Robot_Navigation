import cv2
import numpy as np

import utils
import constant as c

def distanceZframe(disparity):
    disp0 = (disparity[disparity>=0]/16).mean()
    d=disp0
    #Estimate a main disparity
    #d = disp0 #np.mean(disp0) 
    
    #Estimate a most frequent disparity
    #d = max(set(disp0), key = disp0.count)

    #Estimate a median disparity
    #d = np.median(disp0) 

    #Determine the distance in mm
    z=(c.B*c.F)/d #mm
    
    return z, d

def computeDisparityMap(imgL, imgR, dmain=0, numDisp=128, blockSize=33, interval=50, showDisparity=False):

    #initialize stereo disparity
    stereoMatcher = cv2.StereoBM_create()
    if dmain>0:
        sigma = round((dmain-32)/16)*16
        print("SIGMA: ",sigma)
        stereoMatcher.setNumDisparities(64 + sigma)
        #stereoMatcher. setMinDisparity(sigma)
        stereoMatcher.setBlockSize(blockSize)
    else:
        stereoMatcher.setNumDisparities(64)
        stereoMatcher.setBlockSize(blockSize)
    
    #compute disparity map: gray or colour
    disparity = stereoMatcher.compute(imgL, imgR)

    #take only central frames
    center = imgL.shape
    centerY = int(center[0]/2)
    centerX = int(center[1]/2)

    disparity = disparity[centerY-interval:centerY+interval, centerX-interval:centerX+interval]

    #print("Disparity range:{ ", disparity.min()," , ",disparity.max()," }")

    #show disparity frame 
    if showDisparity:
        utils.imshow("Disparity","Disparity map", disparity)

    return disparity