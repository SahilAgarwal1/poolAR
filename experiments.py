import numpy as np
import cv2
from cv2 import convertScaleAbs
import math
import numpy as np
import matplotlib.pyplot as plt
from poolAR_utils import get_table_boundries
import poolAR_utils





clip = cv2.VideoCapture('clips/2019_PoolChamp_Clip4.mp4')

if not clip.isOpened():
    print('Cannot Open Camera or File')
    exit()
while True:
    ret, frame = clip.read()

    if not ret:
        print('Cant Recieve Frame')
        break



    # Get Frame and crop to play area
    x, y , w, h = poolAR_utils.get_play_area_coords(frame)
    frame = frame[y + 20:y + h - 20, x + 20:x + w - 20]

    frame_cp = frame

    poolAR_utils.get_table_balls(frame_cp)

    #Get Boundries and plot
    #table_edges = poolAR_utils.get_table_boundries(frame)
    #for line in table_edges:

    #    cv2.line(frame_cp, (0, int(line[1])), (frame_cp.shape[1], int((frame_cp.shape[1] * line[0]) + line[1])), (0,0,255))



    #cv2.imshow('Image', frame_cp)
    #cv2.waitKey(0)






'''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # HSV Transform

    mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([105, 255, 255]))  # create mask

    conts, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours

    largestCountour = max(conts, key=cv2.contourArea)  # find largest

    x, y, w, h = cv2.boundingRect(largestCountour)  # create bounding rectangle around largest contour

    # Change to Gray Scale

    frame = frame[y + 20:y + h - 20, x + 20:x + w - 20]

    # frame = cv2.GaussianBlur(frame, [0,0], sigmaX=1, sigmaY= 1)

    frame_cp = frame

    table_bounds = get_table_boundries(frame)



    for line in table_bounds:

        print(line)
        cv2.line(frame_cp, (0, int(line[1])), (frame_cp.shape[1], int((frame_cp.shape[1] * line[0]) + line[1])), (0,0,255))

    cv2.imshow('Image', frame_cp)
    cv2.waitKey(0)





    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    upperEdge = frame[0: int(frame.shape[0] * 0.1), 0: frame.shape[1]]  # crop
    lowerEdge = frame[int(frame.shape[0] * 0.9):frame.shape[0], 0: frame.shape[1]]
    leftEdge = frame[0: frame.shape[0], 0: int(frame.shape[1] * 0.05)]
    rightEdge = frame[0:frame.shape[0], int(frame.shape[1] * 0.95): frame.shape[1]]

    # SOBEL OPERATOR LIKES DARK TO LIGHT EDGES DUE TO KERNEL!!!

    sobel_lowerEdge = cv2.Sobel(lowerEdge, ddepth=-1, dx=0, dy=1, ksize=3)
    sobel_upperEdge = cv2.rotate(cv2.Sobel(cv2.rotate(upperEdge, cv2.ROTATE_180), ddepth=-1, dx=0, dy=1, ksize=3)
                                 , cv2.ROTATE_180)

    sobel_rightEdge = cv2.Sobel(rightEdge, ddepth=-1, dx=1, dy=0, ksize=3)
    sobel_leftEdge = cv2.rotate(cv2.Sobel(cv2.rotate(leftEdge, cv2.ROTATE_180), ddepth=-1, dx=1, dy=0, ksize=3)
                                , cv2.ROTATE_180)


    sobel_Edge = sobel_rightEdge
    _, sobel_Edge = cv2.threshold(sobel_Edge, 0, 255, cv2.THRESH_TOZERO+ cv2.THRESH_OTSU)
    linesP = cv2.HoughLinesP(sobel_Edge, 1, np.pi / 180, 50, None, 100, 100)

    blank_test = np.zeros(sobel_Edge.shape, dtype=np.uint8)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(blank_test, (l[0], l[1]), (l[2], l[3]), (100), 1, cv2.LINE_4)


    points = np.column_stack(np.where(blank_test.transpose() != 0))

    X = points[:, 0]
    y = points[:, 1]
    m, b = np.polyfit(y, X, 1)
    b = (-b/m)
    m = 1/m


    x_lin = np.linspace(0, blank_test.shape[1], blank_test.shape[1])

    y_lin = (m * x_lin) + b



    plt.scatter(X,y, c = 'b')
    plt.plot(x_lin,y_lin, c = 'g')
    plt.xlim((0,blank_test.shape[1]))
    plt.show()

    print(m, b)


    cv2.line(blank_test, (0, int(b)), (blank_test.shape[1], int((blank_test.shape[1] * m) + b)), (255))

    cv2.imshow('Image', blank_test)
    cv2.waitKey(0)

    edges = np.zeros(frame.shape, dtype=int)

    edges[0: int(frame.shape[0] * 0.1), 0: frame.shape[1]] += sobel_upperEdge
    edges[int(frame.shape[0] * 0.9):frame.shape[0], 0: frame.shape[1]] += sobel_lowerEdge
    edges[0: frame.shape[0], 0: int(frame.shape[1] * 0.05)] += sobel_leftEdge
    edges[0:frame.shape[0], int(frame.shape[1] * 0.95): frame.shape[1]] += sobel_rightEdge

    edges = np.uint8(edges)

    thr, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)

    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 100, 100)

    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(edges, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv2.LINE_4)

# cv2.imshow('Image', edges)
# cv2.waitKey(0)
'''