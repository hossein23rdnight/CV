import cv2
import numpy as np
from matplotlib import pyplot as plt

I = cv2.imread('/Users/hossein/Desktop/CV/2/cv-lab2/masoleh.jpg')

B = np.zeros_like(I)
B[:,:,0] = I[:,:,0]

G = np.zeros_like(I)
G[:,:,1] = I[:,:,1]

R = np.zeros_like(I)
R[:,:,2] = I[:,:,2]

cv2.imshow('win1', I)

while True:
    k = cv2.waitKey()

    if k == ord('o'):
        cv2.imshow('win1', I)
    elif k == ord('b'):
        cv2.imshow('win1', B)
    elif k == ord('g'):
        cv2.imshow('win1', G)
    elif k == ord('r'):
        cv2.imshow('win1', R)
    elif k == ord('q'):
        cv2.destroyAllWindows()
        break
