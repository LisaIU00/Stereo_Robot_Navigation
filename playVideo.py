import cv2

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
