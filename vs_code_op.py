import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

  
# Computer Camera
# cap = cv.VideoCapture(0)

# Waste/Ant video
# 1
#cap = cv.VideoCapture('/Users/valesanchez/Documents/VS_Code/Waste_Ant/truck-enter-1.avi')
# 2
cap = cv.VideoCapture('/Users/valesanchez/Documents/VS_Code/Waste_Ant/mergedVideo.mp4')

# Highway with cars video
# cap = cv.VideoCapture('/Users/valesanchez/Documents/VS_Code/Waste_Ant/Cars On The Road.mp4')



def vel_hist(input, label):
    H = plt.hist(input.ravel(), bins=20, range=[-50,75]) 

    plt.xlabel('Velocity') 
    plt.ylabel('Number of Pixels')

    plt.title(label)
  
    #plt.show()

    return H



def arrows_opt_flow(img, flow, count, step=16):

    h, w = img.shape[:2]  # exctract rows and columns
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)  # gets the coordinates from the grid
    fx, fy = flow[y,x].T  # vel 

    #print("Min velocity: ",np.amin(flow))
    #print("Max velocity: ",np.amax(flow))

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)  
    lines = np.int32(lines)

    img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    # empty data frame
    #df2 = pd.DataFrame(columns=['No. Pixels','X Velocities','Y Velocities'])
    df2 = pd.DataFrame({'No Pixels': [],
                            'X Velocities': [],
                            'Y Velocities':[]})

    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(img_bgr, (x1, y1), 1, (0, 0, 255), -1)
        cv.arrowedLine(img_bgr,(x1, y1), (_x2,_y2),(0,250,0), 1, tipLength=0.5)
        # image, start point, end point, color, thickness
        # tip length: the length of the arrow tip in relation to the arrow lenght
    
    if count == 10:

        # plot velocities of x and y
        x = vel_hist(fx, 'X Velocities')
        y = vel_hist(fy, 'Y Velocities')
        # dictionary, each key represents the column of the data frame
        df2 = pd.DataFrame({'No Pixels': [x[0]],
                            'X Velocities': [x[1]],
                            'Y Velocities':[y[1]]})

        # df = pd.concat([df, df2], ignore_index=True)
    
    #cv.waitKey()

    return img_bgr,df2

################### MAIN ###################


ret, img = cap.read()

# counter of frames
count = 0

# resize image
first_frame = cv.resize(img, (480, 434))

prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

mask = np.zeros_like(first_frame)
  
# Sets image saturation to maximum
mask[..., 1] = 255

# empty data frame
# df = pd.DataFrame(columns=['No. Pixels','X Velocities','Y Velocities'])
df = pd.DataFrame({'No Pixels': [],
                        'X Velocities': [],
                        'Y Velocities':[]})

  
# video loop
while(cap.isOpened()):
      
    ret, img_two = cap.read()

    # resize image
    frame = cv.resize(img_two, (480, 434))


    cv.imshow("input", frame)
        
    # Converts each frame to grayscale - we previously 
    # only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculates dense optical flow by Farneback method
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, 
                                       None, 
                                       0.5,  # image scale for the pyramid
                                       3,  # levels or num of pyramid layers
                                       15,  # window size
                                       3,  # iterations
                                       5,  # size of pixel neighbourhood
                                       1.2,  # standard deviation of Gaussian
                                       0)  # flags

    rgb, df2  = arrows_opt_flow(gray, flow, count)
 
    # Opens a new window and displays the output frame
    cv.imshow("dense optical flow", rgb)
    # Updates previous frame
    prev_gray = gray

    # increase count for number of frames
    if count != 10:
        count += 1
    else:
        df = pd.concat([df, df2], ignore_index=True)
        print(df)
        count = 0

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

df.to_csv("velocities.csv")

cap.release()
cv.destroyAllWindows()
