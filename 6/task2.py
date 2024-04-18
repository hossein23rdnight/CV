import numpy as np
import cv2

I1 = cv2.imread('/Users/hossein/Desktop/CV/6/scene1.jpg')
I2 = cv2.imread('/Users/hossein/Desktop/CV/6/scene2.jpg')

cv2.imshow('Image 1 (background)', I1)
cv2.waitKey(0)

cv2.imshow('Image 2', I2)
cv2.waitKey(0)

K = np.abs(np.int16(I2)-np.int16(I1)) # take the (signed int) differnce
K = K.max(axis=2) # choose the maximum value over color channels
K = np.uint8(K)
cv2.imshow('The difference image', K)
cv2.waitKey(0)

threshold = 39 
ret, T = cv2.threshold(K,threshold,255,cv2.THRESH_BINARY)
cv2.imshow('Thresholded', T)
cv2.waitKey(0)

## opening 
kernel = np.ones((3,3),np.uint8)
T = cv2.morphologyEx(T, cv2.MORPH_OPEN, kernel)
cv2.imshow('After Openning', T)
cv2.waitKey(0)

## closing 
kernel = np.ones((17,17),np.uint8)
T = cv2.morphologyEx(T, cv2.MORPH_CLOSE, kernel)
cv2.imshow('After Closing', T)
cv2.waitKey(0)

n,C = cv2.connectedComponents(T);

J = I2.copy()
J[T != 0] = [255,255,255]
font = cv2.FONT_HERSHEY_SIMPLEX 
cv2.putText(J,'There are %d toys!'%(n-1),(20,40), font, 1,(0,0,255),2)
cv2.imshow('Number', J)
cv2.waitKey()
  
n,C,stats, centroids = cv2.connectedComponentsWithStats(T);

largest_area = 0
largest_component_index = -1

for i in range(1, n):  
    #area =  stats[i][4]
    area = stats[i, cv2.CC_STAT_AREA]
    if area > largest_area:
        largest_area = area
        largest_component_index = i

J = I2.copy()
J[C == largest_component_index] = [0, 0, 255]

cv2.imshow('Largest Toy in red', J)
cv2.waitKey()









    

    
