import cv2
import numpy as np

I = cv2.imread('/Users/hossein/Desktop/CV/8/cv-lab8/karimi.jpg',0)

c = np.array([[I.shape[1]/2.0], [I.shape[0]/2.0]])

for theta in range(0,360):
    th = theta * np.pi / 180 

    R = np.array([[np.cos(th),-np.sin(th)],
                  [np.sin(th), np.cos(th)]])

    #t = np.zeros((2,1)) # you need to change this!
    t = c - R @ c


    # concatenate R and t to create the 2x3 transformation matrix
    M = np.hstack([R,t])

    J = cv2.warpAffine(I,M, (I.shape[1], I.shape[0]) )

    cv2.imshow('J',J)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

