import cv2
import numpy as np
I = cv2.imread('2/cv-lab2/damavand.jpg', cv2.IMREAD_UNCHANGED)
B = I[:,:,0]
G = I[:,:,1]
R = I[:,:,2]
#print(B.shape)

cv2.imshow('win1',I)
while 1:
    k = cv2.waitKey()
    if k == ord('o'):
        cv2.imshow('win1',I)
    elif k == ord('b'):
        cv2.imshow('win1',B)
    elif k == ord('g'):
      cv2.imshow('win1',G)
    elif k == ord('r'):
      cv2.imshow('win1',R)
    elif k == ord('q'):
        cv2.destroyAllWindows()
        break
        
