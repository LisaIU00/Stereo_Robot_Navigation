import cv2
import numpy as np
import pandas as pd

import chessboardRecognition as chess
import disparityMap as disp
import constant as c
import utils as u


def main(videoL, videoR, show = False):

    try:

        num_frame = 0 #count number of frame
        diffW=[] #difference from real W dimention to computed in millimeters for all frame
        diffH=[] #difference from real H dimention to computed in millimeters for all frame
        dist = [] #distance z in metres for all frame in meters 
        alarm_frame = [] #for all frame save 1 if z is minus of alarm distance, otherwise save 0
        all_dmain = [] #save main disparity for all frame in meters
        dh = 0
        dw = 0

        while(videoL.isOpened() and videoR.isOpened()):
            #extract frames
            retL, frameL = videoL.read()
            retR, frameR = videoR.read()
            
            #check frames
            if not retL or frameL is None or  not retR or frameR is None:
                videoL.release()
                videoR.release()
                break

            frameL = frameL.astype(np.uint8)
            frameR = frameR.astype(np.uint8)
            
            imgL= cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
            imgR= cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
            
            #compute disparity map in a central area of the reference frame (100x100 pixels)
            disparity = disp.computeDisparityMap(frameL, frameR)

            #estimate main disparity of the frame
            zFrame, dmain = disp.distanceZframe(disparity) #m

            #print distance and show the frame
            if show:
                title = "frame "+str(num_frame)+":\ndistance="+str(zFrame)+" ; dmain="+str(dmain)
                u.imshow("VideoL",title, imgL)
            
            dist.append(zFrame)
            all_dmain.append(dmain)
            
            #verify if the distance zFrame[m] is below 0.8m
            if zFrame <= c.MIN_DIST:
                #save number of frame
                print("The distance from camera to the obstable is minus then", c.MIN_DIST,"m :   ",zFrame)
                alarm = 1
            else:
                alarm = 0
                
            #Find chessboard & Compute dimension of the chessboard
            ret, wL,hL = chess.chessboard(imgR)
            
            #if chessboard was found
            if(ret == True):
                
                #Compare the size of the chessboard computed with the real ones
                z_mm = zFrame*1000
                W_mm = (z_mm*wL)/c.F
                H_mm = (z_mm*hL)/c.F
                
                dw = abs(c.REAL_W-W_mm)
                dh = abs(c.REAL_H-H_mm)

            #save calculated difference
            diffW.append(dw)
            diffH.append(dh)

            num_frame+=1

    except KeyboardInterrupt:
        # If we press stop (jupyter GUI) release the video
        videoL.release()
        videoR.release()
        print("Released Video Resource")

    #plot graph with difference between real W and calculated W for each frame
    u.plotgraph('distance Z meters',np.arange(num_frame-1), dist[1:],  'frame', 'z [m]', 'plot/z_plot.png')
    u.plotgraph('distance alarm',np.arange(len(alarm_frame)), alarm_frame,  'frame', 'alarm flag', 'plot/alarm_frame_plot.png')
    u.plotgraph('main disparity',np.arange(len(all_dmain)), all_dmain,  'frame', 'main disparity', 'plot/main_disparity_plot.png')
    u.plotgraph('difference weighted from real',np.arange(len(diffW)-1), diffW[1:],  'frame', 'difference W [mm]', 'plot/Wdiff_plot.png')
    u.plotgraph('difference heigh from real',np.arange(len(diffH)-1), diffH[1:],  'frame', 'difference H [mm]', 'plot/Hdiff_plot.png')
 

    
if __name__ == "__main__":
    #args = u.getParams()

    videoL = cv2.VideoCapture(c.video_pathL)
    videoR = cv2.VideoCapture(c.video_pathR)

    main(videoL, videoR)