import numpy as np
import cv2

cap = cv2.VideoCapture('/Users/hossein/Desktop/CV/eggs-reverse.avi')

while True:
    ret, I = cap.read()
    
    if not ret:
        break

    cv2.imshow('win1', I)
    
    key = cv2.waitKey(33) 

    if key & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
