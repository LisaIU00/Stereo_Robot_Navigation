import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import chessboardRecognition as chess
import disparityMap as disp
import constant as c
import utils as u


def main(videoL, videoR, showFrame = False):

    try:
        
        df = pd.DataFrame(columns = ['dMain','z [mm]','Z [m]','alarm','Hdiff','Wdiff'])

        num_frame = 0
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
            
            imgL= cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
            imgR= cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
            
            #compute disparity map in a central area of the reference frame (100x100 pixels)
            disparity = disp.computeDisparityMap(imgL, imgR)

            #estimate main disparity of the frame
            zFrame, dmain = disp.distanceZframe(disparity) #mm

            #print distance and show the frame
            if showFrame:
                title = "frame "+str(num_frame)+":\ndistance="+str(zFrame)+" ; dmain="+str(dmain)
                u.imshow("VideoL",title, imgL)
            
            
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
                z_mm = zFrame
                W_mm = (z_mm*wL)/c.F
                H_mm = (z_mm*hL)/c.F
                
                dw = abs(c.REAL_W-W_mm)
                dh = abs(c.REAL_H-H_mm)


                output = {
                            'dMain': dmain,
                            'z [mm]': zFrame,
                            'Z [m]': zFrame/1000,
                            'alarm': alarm,
                            'Hdiff': dh,
                            'Wdiff': dw
                            }
                df.loc[len(df)] = output

            num_frame += 1

    except KeyboardInterrupt:
        # If we press stop (jupyter GUI) release the video
        videoL.release()
        videoR.release()
        print("Released Video Resource")

    # plotting dataframe
    df.plot(subplots=True, grid=True)
    plt.tight_layout()
    plt.savefig("result_plot.png")
    plt.show()

    
if __name__ == "__main__":
    #args = u.getParams()

    videoL = cv2.VideoCapture(c.video_pathL)
    videoR = cv2.VideoCapture(c.video_pathR)

    main(videoL, videoR)