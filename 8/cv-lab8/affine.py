import cv2
import numpy as np

I = cv2.imread('/Users/hossein/Desktop/CV/8/cv-lab8/karimi.jpg')

t = np.array([[30],
              [160]], dtype=np.float32)

A = np.array([[1, 0],
              [0, 1]], dtype=np.float32)




M = np.hstack([A,t])

output_size = (I.shape[1], I.shape[0])
J = cv2.warpAffine(I,M,  output_size)

cv2.imshow('I',I)
cv2.waitKey(0)
cv2.imshow('J',J)
cv2.waitKey(0)
