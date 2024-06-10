import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import chessboardRecognition as chess
import disparityMap as disp
import constant as c
import utils as u


def main(videoL, videoR,showFrame = False, showDisparity = False, showChess = True):

    try:
        
        #initialization of variables useful later
        df = pd.DataFrame(columns = ['dMain','z [mm]','alarm', 'H*W difference'])

        num_frame = 0
        dh = 0
        dw = 0

        #run the body of the while loop until have finished examining each frame of the video
        while(videoL.isOpened() and videoR.isOpened()):

            #extract frames from video
            retL, frameL = videoL.read()
            retR, frameR = videoR.read()
            
            #check frames
            if not retL or frameL is None or  not retR or frameR is None:
                videoL.release()
                videoR.release()
                break
            
            #convert the frame using cv2.COLOR_BGR2GRAY color space
            imgL= cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
            imgR= cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
            
            #compute disparity map in a central area of the reference frame (default 80x80 pixels)
            disparity = disp.computeDisparityMap(imgL, imgR, showDisparity)

            #estimate the distance z from the obstacle (zFrame) and the mean disparity (dmain) of the frame from disparity map
            zFrame, dmain = disp.distanceZframe(disparity) #mm

            #print distance and show the frame if showFrame=True (default = False)
            if showFrame:
                title = "frame "+str(num_frame)+":\ndistance="+str(zFrame)+" ; dmain="+str(dmain)
                u.imshow("VideoL",title, imgL)

            #verify if the distance zFrame[mm] is below 800mm
            if zFrame <= c.MIN_DIST:
                print("The distance from camera to the obstable is minus then", c.MIN_DIST,"m :   ",zFrame)
                #if the condition is verified then we are in an alarm state, set the alarm variable to 1
                alarm = 1
            else:
                alarm = 0           
                
            #Find chessboard & Compute dimension of the chessboard
            ret, wL,hL = chess.chessboard(imgL, showChess)
            #ret = True if the chessboard was found;
            #wL and hL indicate the estimated size of the chessboard in pixels.
            
            #if chessboard was found:
            if(ret == True):
                #compute estimated dimension of the chessboard from pixel to mm
                z_mm = zFrame
                W_mm = (z_mm*wL)/c.F
                H_mm = (z_mm*hL)/c.F
                dim_mm = W_mm*H_mm

                #Compare the size of the chessboard computed with the real ones
                '''dw = abs(c.REAL_W-W_mm)
                dh = abs(c.REAL_H-H_mm)'''
                dd = abs(c.REAL_DIM-dim_mm)

                #create output plot
                output = {
                            'dMain': dmain,
                            'z [mm]': zFrame,
                            'alarm': alarm,
                            'H*W difference':dd
                            }
                df.loc[len(df)] = output

                num_frame += 1 
                
    except KeyboardInterrupt:
        # If we press stop (jupyter GUI) release the video
        videoL.release()
        videoR.release()
        print("Released Video Resource")



    # plotting dataframe
    df.plot(subplots=True,grid=True)
    plt.tight_layout()
    plt.savefig("result_plot.png")

    plt.show()

    
if __name__ == "__main__":
    #capturing from video files
    videoL = cv2.VideoCapture(c.video_pathL)
    videoR = cv2.VideoCapture(c.video_pathR)

    main(videoL, videoR)