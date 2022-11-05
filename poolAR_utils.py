import cv2
import numpy as np

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

# Function to get the play area of the pool table
# Best to use an additional crop buffer as the function often gets the wooden rim of table as well
# currently looks for blue table, need to make more robust scheme for any table cloth color
# input is orignal frame, output is x y w h of the crop
def get_play_area_coords(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # HSV Transform


    # get mode near center of image, cut out 75 percent of the image

    mode = get_mode(hsv[int(hsv.shape[1] * 0.33) : int(hsv.shape[1] * 0.66), int(hsv.shape[1] * 0.33) : int(hsv.shape[1] * 0.66) ])

    mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([105, 255, 255]))  # create mask




    conts, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours

    largestCountour = max(conts, key=cv2.contourArea)  # find largest

    x, y, w, h = cv2.boundingRect(largestCountour)  # create bounding rectangle around largest contour

    return (x,y,w,h)



# function to get crops of the table rails
# input is play area cropped frame
# returns 10 percent of the top and bottom of image and 5 percent of the sides
# of the image by default

def get_table_rail_crops(frame, top_crop_ratio = 0.1, side_crop_ratio = 0.05):

    upperEdge = frame[0: int(frame.shape[0] * top_crop_ratio), 0: frame.shape[1]]
    lowerEdge = frame[int(frame.shape[0] * (1 - top_crop_ratio)):frame.shape[0], 0: frame.shape[1]]
    leftEdge = frame[0: frame.shape[0], 0: int(frame.shape[1] * side_crop_ratio)]
    rightEdge = frame[0:frame.shape[0], int(frame.shape[1] * (1 - side_crop_ratio)): frame.shape[1]]

    return upperEdge, lowerEdge, leftEdge, rightEdge



# function to get edge boundry lines from the table
# input is the frame itself, the cropping boundries for the rail crops


def get_table_boundries(frame, top_crop_ratio = 0.1, side_crop_ratio = 0.05 ):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray scale
    frame = cv2.GaussianBlur(frame, (11,11), 0) # add gaussian blur

    #crop the edges
    upperEdge, lowerEdge,leftEdge, rightEdge = get_table_rail_crops(frame, top_crop_ratio, side_crop_ratio)

    # SOBEL OPERATOR LIKES DARK TO LIGHT EDGES DUE TO KERNEL!!!

    # perform sobel with proper rotations so that edges are detected from dark to light

    sobel_lowerEdge = cv2.Sobel(lowerEdge, ddepth=-1, dx=0, dy=1, ksize=3)
    sobel_upperEdge = cv2.rotate(cv2.Sobel(cv2.rotate(upperEdge, cv2.ROTATE_180), ddepth=-1, dx=0, dy=1, ksize=3)
                                 , cv2.ROTATE_180)

    sobel_rightEdge = cv2.Sobel(rightEdge, ddepth=-1, dx=1, dy=0, ksize=3)
    sobel_leftEdge = cv2.rotate(cv2.Sobel(cv2.rotate(leftEdge, cv2.ROTATE_180), ddepth=-1, dx=1, dy=0, ksize=3)
                                , cv2.ROTATE_180)

    #perform thresholding to get rid of noise

    _, sobel_upperEdge = cv2.threshold(sobel_upperEdge, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    _, sobel_lowerEdge = cv2.threshold(sobel_lowerEdge, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    _, sobel_leftEdge = cv2.threshold(sobel_leftEdge, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    _, sobel_rightEdge = cv2.threshold(sobel_rightEdge, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)


    list_of_lines = [] # list of lines in order upper lower left right

    for edge in [sobel_upperEdge,sobel_lowerEdge,sobel_leftEdge,sobel_rightEdge]:

        linesP = cv2.HoughLinesP(edge, 1, np.pi / 180, 50, None, 100, 100)

        blank = np.zeros(edge.shape, dtype=np.uint8)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(blank, (l[0], l[1]), (l[2], l[3]), (50), 1, cv2.LINE_4)

        points = np.column_stack(np.where(blank.transpose() != 0))
        X = points[:, 0]
        y = points[:, 1]


        # Calculate lines of best fit for the table
        if np.array_equal(edge,sobel_upperEdge) or np.array_equal(edge, sobel_lowerEdge):
            m,b =  np.polyfit(X, y, 1)
        else:
            m, b = np.polyfit(y, X, 1)
            b = (-b / m)
            m = 1 / m

        cv2.line(blank, (0, int(b)), (blank.shape[1], int((blank.shape[1] * m) + b)), (255))

        list_of_lines.append([m,b])

    #adjusting the bottom line y intercept due to the coord change
    list_of_lines[1][1] += int(frame.shape[0] * (1 - top_crop_ratio)) # shift bottom line down b + shift
    list_of_lines[3][1] -= int(list_of_lines[3][0] * frame.shape[1] * (1-side_crop_ratio)) # shift right line right [b - m * shift]



    return list_of_lines # returns lines in order top, bottom, left, right



def get_mode(frame):
    mode = []
    for dim in range(frame.shape[2]):
        array = frame[:,:,dim]
        vals, counts = np.unique(array, return_counts=True)
        index = np.argmax(counts)
        mode.append(vals[index])


    return mode


def get_table_balls(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # blur the HSV
    hsv = cv2.medianBlur(hsv, 11)
    hsv = cv2.GaussianBlur(hsv, (11,11), 0)

    # get the mode from the frame
    mode = get_mode(hsv)

    mode = [100,218,235]

    print(mode)


    #mask from hsv
    mask = cv2.inRange(hsv, np.subtract(np.array(mode), np.array( [4, 75, 50]))  , np.add(np.array(mode), np.array([4, 50 , 100])))  # create mask


    # subtract mask from gray

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    sub = cv2.subtract(frame, mask)

    cv2.imshow('Image', sub)
    cv2.waitKey(0)















