import cv2
import numpy as np
import constant as c
import utils as u


def chessboard(img, showChess=False):
    w=0
    h=0

    #find chessboard 
    ret,corners = cv2.findChessboardCorners(img ,c.pattern_size)

    
    if ret == True:

        #display the pixel coordinates of the internal corners of the chessboard
        cv2.drawChessboardCorners(img, c.pattern_size, corners, ret)

        #show frame with corners chessboard
        if showChess:
            u.imshow("Chessboard", "Chessboard", img)
    
        #compute pixel dimention of chessboard
        w = abs(corners[c.NUM_H-1][0][0]-corners[corners.shape[0]-1][0][0])
        h = abs(corners[c.NUM_H-1][0][1]-corners[0][0][1])


    return ret, w,h