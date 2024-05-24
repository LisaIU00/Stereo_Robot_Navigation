import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import chessboardRecognition as chess
import disparityMap as disp
import constant as c
import utils as u


def main(videoL, videoR,showFrame = False, showDisparity = False, showChess = True):
    att_first = True
    first = False
    att_sec = False
    second = False
    numF = 0
    numS = 0
    sumF = 0
    sumS = 0
    try:
        
        df = pd.DataFrame(columns = ['dMain','z [mm]','alarm','H difference','W difference'])

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
            disparity = disp.computeDisparityMap(imgL, imgR, showDisparity)

            #estimate main disparity of the frame
            zFrame, dmain = disp.distanceZframe(disparity) #mm

            #print distance and show the frame
            if showFrame:
                title = "frame "+str(num_frame)+":\ndistance="+str(zFrame)+" ; dmain="+str(dmain)
                u.imshow("VideoL",title, imgL)
            
            
            #verify if the distance zFrame[mm] is below 800mm
            if zFrame <= c.MIN_DIST:
                #save number of frame
                print("The distance from camera to the obstable is minus then", c.MIN_DIST,"m :   ",zFrame)
                alarm = 1
                if att_first:
                    att_first=False
                    first=True
                if att_sec:
                    att_sec=False
                    second=True
                if first:
                    numF+=1
                if second:
                    numS += 1
            else:
                if first:
                    first=False
                    att_sec = True
                if second:
                    second=False
                alarm = 0
                
            #Find chessboard & Compute dimension of the chessboard
            ret, wL,hL = chess.chessboard(imgL, showChess)
            
            #if chessboard was found
            if(ret == True):
                
                #Compare the size of the chessboard computed with the real ones
                z_mm = zFrame
                W_mm = (z_mm*wL)/c.F
                H_mm = (z_mm*hL)/c.F
                
                dw = abs(c.REAL_W-W_mm)
                dh = abs(c.REAL_H-H_mm)

                if first:
                    sumF += (dh+dw)/2
                else:
                    if second:
                        sumS += (dh+dw)/2

                output = {
                            'dMain': dmain,
                            'z [mm]': zFrame,
                            'alarm': alarm,
                            'H difference': dh,
                            'W difference': dw
                            }
                df.loc[len(df)] = output

            num_frame += 1

    except KeyboardInterrupt:
        # If we press stop (jupyter GUI) release the video
        videoL.release()
        videoR.release()
        print("Released Video Resource")

    print("PRIMO: ",sumF/numF)
    print("SECONDO: ", sumS/numS)

    # plotting dataframe
    df.plot(subplots=True,grid=True)
    plt.tight_layout()
    plt.savefig("result_plot.png")

    plt.show()

    
if __name__ == "__main__":
    #args = u.getParams()

    videoL = cv2.VideoCapture(c.video_pathL)
    videoR = cv2.VideoCapture(c.video_pathR)

    main(videoL, videoR)