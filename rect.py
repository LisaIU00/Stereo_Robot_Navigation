    
import cv2
import numpy as np
from matplotlib import pyplot as plt

def imshow(wname,title, img):
    plt.figure(wname); 
    plt.clf()
    plt.imshow(img)
    plt.title(title)
    plt.pause(0.00001)


def rectify():
    ############# open video and put all frame on c1_images for left and c2_images for right ##########
    video_pathL = 'robot-navigation-video/robotL.avi'
    videoL = cv2.VideoCapture(video_pathL)
    video_pathR = 'robot-navigation-video/robotR.avi'
    videoR = cv2.VideoCapture(video_pathR)

    c1_images = []
    c2_images = []
    try:
        numFrame=0
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

            c1_images.append(imgL)
            c2_images.append(imgR)

    except KeyboardInterrupt:
        # If we press stop (jupyter GUI) release the video
        videoL.release()
        videoR.release()
        print("Released Video Resource")

    videoL.release()
    videoR.release()
    print("define c1 e c2")

    ############### define element necessary to compute calibration ################

    #change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    #num columns and rows of chessboard
    columns = 8
    rows = 6
    pattern_size = (columns, rows)

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = objp*20
    
    # Arrays to store object points and image points from all the images.    
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
    
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    for i in range(115,155,1):
        frame1 = c1_images[i]
        frame2 = c2_images[i]

        c_ret1, corners1 = cv2.findChessboardCorners(frame1, pattern_size, None)
        c_ret2, corners2 = cv2.findChessboardCorners(frame2, pattern_size, None)
    
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv2.cornerSubPix(frame1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(frame2, corners2, (11, 11), (-1, -1), criteria)

            cv2.drawChessboardCorners(frame1, pattern_size, corners1, c_ret1)
            cv2.drawChessboardCorners(frame2, pattern_size, corners2, c_ret2)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

            #imshow("Chessboard Left", "Chessboard Left", frame1)

    print("find and draw chessboard")

    #element that described the dimention of frame (left and right is equal)
    size1 = c1_images[0].shape[::-1]
    frameSize = (c1_images[0].shape[1], c1_images[0].shape[0])
    #height, width = c1_images[0].shape

    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, frameSize, None, None)
    print("calibrate L")
    
    newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, frameSize, 1, frameSize)
    print("opt new camera matrix L")
    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, frameSize, None, None)
    print("calibrate R")
    newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, frameSize, 1, frameSize)
    print("opt new camera matrix R")

    ######### Stereo Vision Calibration #############################################
    ## stereoCalibrate Output: retStereo is RSME, newCameraMatrixL and newCameraMatrixR are the intrinsic matrices for both
                    ## cameras, distL and distR are the distortion coeffecients for both cameras, rot is the rotation matrix,
                    ## trans is the translation matrix, and essentialMatrix and fundamentalMatrix are self descriptive
    flags = 0
    flags = cv2.CALIB_FIX_INTRINSIC

    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, newCameraMatrixL, distL, newCameraMatrixR, distR, size1, criteria_stereo, flags)
    print(retS)
    print('stereo vision calibrate done')

    rectify_scale= 1
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR, size1, Rot, Trns, rectify_scale,(0,0))
    print("rectify done")

    Left_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,size1, cv2.CV_16SC2)
    Right_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,size1, cv2.CV_16SC2)
    print("undistorted map done: ")
    
    print("Left_Stereo_Map_x",Left_Stereo_Map[0])
    print("Left_Stereo_Map_y",Left_Stereo_Map[1])
    print("Right_Stereo_Map_x",Right_Stereo_Map[0])
    print("Right_Stereo_Map_y",Right_Stereo_Map[1])
    return Left_Stereo_Map,Right_Stereo_Map



'''Left_Stereo_Map,Right_Stereo_Map = rectify()
###### rewrite all frame of video left and right to remap using the result of rectifcation #############
video_pathL = 'robot-navigation-video/robotL.avi'
videoL = cv2.VideoCapture(video_pathL)
video_pathR = 'robot-navigation-video/robotR.avi'
videoR = cv2.VideoCapture(video_pathR)

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

        imgL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)#[centerY-interval:centerY+interval, centerX-interval:centerX+interval]
        imgR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)#[centerY-interval:centerY+interval, centerX-interval:centerX+interval]

        left_rectified = cv2.remap(imgL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT, 0)
        right_rectified = cv2.remap(imgR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT, 0)

        #imshow("Left Rectified", "Left Rectified", left_rectified)
        #imshow("Right Rectified", "Right Rectified", right_rectified)

        num_frame+=1


except KeyboardInterrupt:
    # If we press stop (jupyter GUI) release the video
    videoL.release()
    videoR.release()
    print("Released Video Resource")'''
