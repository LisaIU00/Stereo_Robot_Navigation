import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
from scipy import stats as st


plt.rcParams["figure.figsize"] = (15,6)

f = 567.2
b = 92.226
min_dist = 0.8
num_h = 8
num_w = 6
pattern_size = (num_h, num_w)
real_h = 178
real_w = 125


def getParams():
    parser = argparse.ArgumentParser(prog='CVproject',description='Stereo Robot Navigation')
    parser.add_argument('-d','--numDisparities',default='128',help='numDisparities parameter for disparity map algorithm',type=int)
    parser.add_argument('-b','--blockSize',default='33',help='blocksize parameter for disparity map algorithm', type=int)
    return parser.parse_args()

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
    z=(b*f)/(d + 1e-5) #mm
    #convert in m
    z_m = z/1000 #m
    
    return z_m, d

# display image
def imshow(wname,title, img):
    plt.figure(wname); 
    plt.clf()
    plt.imshow(img)
    plt.title(title)
    plt.pause(0.0001)

def plotgraph(title, x, y, xlabel, ylabel, nomeimg,color='m'):
    plt.figure(title)
    plt.plot(x, y,color, linewidth=3)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(nomeimg)
    plt.show(block=False)
    

def computeDisparityMap(frameL, frameR, numDisp=128, blockSize=15):
    #compute central frames
    center = frameL.shape
    centerY = int(center[0]/2)
    centerX = int(center[1]/2)
    interval = 50

    #Get image center box
    imgL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)#[centerY-interval:centerY+interval, centerX-interval:centerX+interval]
    imgR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)#[centerY-interval:centerY+interval, centerX-interval:centerX+interval]

    #initialize stereo disparity
    stereoMatcher = cv2.StereoBM_create(numDisparities=numDisp, blockSize=blockSize)
    
    #compute disparity map: gray or colour
    disparity = stereoMatcher.compute(imgL, imgR)
    disparity = disparity[centerY-interval:centerY+interval, centerX-interval:centerX+interval]

    #disparity = cv2.normalize(disparity, None, alpha=0, beta=128, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    #print("Disparity range:{ ", disparity.min()," , ",disparity.max()," }")

    #show disparity frame 
    #imshow("Disparity","Disparity map", disparity)

    return disparity

def chessboard(img):
    w=0
    h=0

    #find chessboard
    ret,corners = cv2.findChessboardCorners(img ,pattern_size)

    
    if ret == True:
        #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
        #corners = cv2.cornerSubPix(img,corners,(15,15),(3,3),criteria)

        #display the pixel coordinates of the internal corners of the chessboard
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)

        #show frame with corners chessboard
        #imshow("Chessboard", "Chessboard", img)
    
        #compute pixel dimention of chessboard
        w = abs(corners[num_h*(num_w-1)][0][0]-corners[0][0][0])
        h = abs(corners[num_h-1][0][1]-corners[0][0][1])

    return ret, w,h

#show video frames
def playVideoS(video):
    i=0
    #comute all frames of video
    while(video.isOpened()):
        #extract frame
        ret, frame = video.read()

        #check frame
        if not ret:
            video.release()
            break
            
        #compute central frames
        center = frame.shape
        centerY = int(center[0]/2)
        centerX = int(center[1]/2)
        interval = 50
            
        #Get image center box
        centerGray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[centerY-interval:centerY+interval, centerX-interval:centerX+interval]
          
        #get total image gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #imshow("Vv", "Video", centerGray)


def main(blockSize,numDisp=128):

    #Left_Stereo_Map,Right_Stereo_Map = rc.rectify()

    video_pathL = 'robot-navigation-video/robotL.avi'
    videoL = cv2.VideoCapture(video_pathL)
    video_pathR = 'robot-navigation-video/robotR.avi'
    videoR = cv2.VideoCapture(video_pathR)

    diffW=[]
    diffH=[]
    dist = []

    dh = 0
    dw = 0

    try:
        num_frame = 0
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
            
            #compute disparity map in a central area of the reference frame (100x100 pixels)
            disparity = computeDisparityMap(frameL, frameR, numDisp, blockSize)

            #estimate main disparity of the frame
            zFrame, dmain = distanceZframe(disparity) #m

            imgL= cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
            imgR= cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

            #print distance and show the frame
            title = "frame "+str(num_frame)+":\ndistance="+str(zFrame)+" ; dmain="+str(dmain)
            imshow("VideoL",title, imgL)
            
            dist.append(zFrame)
            
            #verify if the distance zFrame[m] is below 0.8m
            if zFrame <= min_dist:
                #save number of frame
                print("The distance from camera to the obstable is minus then", min_dist,"m :   ",zFrame)
                

            #Find chessboard & Compute dimension of the chessboard
            ret, wL,hL = chessboard(imgR)
            
            #if chessboard was found
            if(ret == True):
                
                #Compare the size of the chessboard computed with the real ones
                z_mm = zFrame*1000
                W_mm = (z_mm*wL)/f
                H_mm = (z_mm*hL)/f


                dw = abs(real_w-W_mm)
                dh = abs(real_h-H_mm)

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
    plotgraph('distance Z',np.arange(num_frame-1), dist[1:],  'frame', 'z [m]', 'z_plot.png')
    plotgraph('difference weighted from real',np.arange(len(diffW)-1), diffW[1:],  'frame', 'difference W [mm]', 'Wdiff_plot.png')
    plotgraph('difference heigh from real',np.arange(len(diffH)-1), diffH[1:],  'frame', 'difference H [mm]', 'diffH_plot.png')
    

    
    
if __name__ == "__main__":
    args =getParams()

    main(args.blockSize,args.numDisparities)