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
        
        df = pd.DataFrame(columns = ['dMain','z [mm]','alarm','H difference','W difference', 'Average difference'])

        num_frame = 0
        dh = 0
        dw = 0
        
        alarm_range = []
        start_alarm = 0
        prec_alarm = 0

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
            else:
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

                output = {
                            'dMain': dmain,
                            'z [mm]': zFrame,
                            'alarm': alarm,
                            'H difference': dh,
                            'W difference': dw,
                            'Average difference': (dh+dw)/2
                            }
                df.loc[len(df)] = output

                if alarm == 1:
                    if prec_alarm == 0:
                        start_alarm = num_frame
                    prec_alarm = 1
                else:
                    if prec_alarm == 1:
                        alarm_range.append([start_alarm, num_frame-1])
                    prec_alarm=0
                num_frame += 1 
                
    except KeyboardInterrupt:
        # If we press stop (jupyter GUI) release the video
        videoL.release()
        videoR.release()
        print("Released Video Resource")

    general_differenceH = []
    general_differenceW = []
    general_difference = []

    for i in range(len(alarm_range)):
        sum = 0
        sumH = 0
        sumW = 0
        num = 0
        for n in (alarm_range[i][0], alarm_range[i][1], 1):
            sum += df['Average difference'][n]
            sumH += df['H difference'][n]
            sumW += df['W difference'][n]
            num += 1
        general_difference.append(sum/num)
        general_differenceH.append(sumH/num)
        general_differenceW.append(sumW/num)
    print(alarm_range)
    print("difference between computed and real: ", general_difference)
    print("H difference between computed and real: ", general_differenceH)
    print("W difference between computed and real: ", general_differenceW)


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