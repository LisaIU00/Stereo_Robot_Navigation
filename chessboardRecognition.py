import cv2
import numpy as np
import constant as c


def chessboard(img):
    w=0
    h=0

    #find chessboard
    ret,corners = cv2.findChessboardCorners(img ,c.pattern_size)

    
    if ret == True:
        #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
        #corners = cv2.cornerSubPix(img,corners,(15,15),(3,3),criteria)

        #display the pixel coordinates of the internal corners of the chessboard
        cv2.drawChessboardCorners(img, c.pattern_size, corners, ret)

        #show frame with corners chessboard
        #imshow("Chessboard", "Chessboard", img)
    
        #compute pixel dimention of chessboard
        w = abs(corners[c.NUM_H*(c.NUM_W-1)][0][0]-corners[0][0][0])
        h = abs(corners[c.NUM_H-1][0][1]-corners[0][0][1])

    return ret, w,h