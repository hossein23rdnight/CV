import numpy as np
import cv2

cap = cv2.VideoCapture('/Users/hossein/Desktop/CV/3/cv-lab3/eggs.avi')

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

fourcc = cv2.VideoWriter_fourcc(*'XVID')


out = cv2.VideoWriter('eggs-reverse.avi',fourcc, 30.0, (w,h))


buffer = []
while True:
    ret, I = cap.read()

    if ret == False: # end of video (or error)
        break
    
    buffer.append(I)

buffer.reverse()
for frame in buffer:
    out.write(frame)

cap.release()
out.release()
