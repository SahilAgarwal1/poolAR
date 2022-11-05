import numpy as np
import cv2
from matplotlib import pyplot as plt

clip = cv2.VideoCapture('clips/2019_PoolChamp_Clip2.mp4')

if not clip.isOpened():
    print('Cannot Open Camera or File')
    exit()
while True:
    ret, frame = clip.read()

    if not ret:
        print('Cant Recieve Frame')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # HSV Transform
    cv2.imshow('gray',gray)
    cv2.waitKey(0)
    plt.hist(gray.ravel(), 256, [0,256])
    plt.show()
    break
    '''
    mask = cv2.inRange(hsv, np.array([100,50,50]), np.array([105, 255, 255])) # create mask
    conts,hier = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # find contours
    largestCountour = max(conts, key=cv2.contourArea) # find largest contour
    x,y,w,h = cv2.boundingRect(largestCountour) # create bounding rectangle around largest contour
    #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
    #cv2.drawContours(frame,conts,-1,color=1, thickness=cv2.FILLED )
    hsv2 = hsv[y + 20:y+h - 20, x + 20:x+ w - 20] #crop
    edges = cv2.Canny(hsv2, 25, 100)
    #mask2 = cv2.inRange(hsv2, np.array([95, 50, 50]), np.array([100, 255, 255]))

    cv2.imshow('frame', edges)
    '''
    cv2.waitKey(0)
    if cv2.waitKey(1) == ord('q'):
        break
clip.release()
cv2.destroyAllWindows()

