import cv2
import numpy as np


I, J = cv2.imread('2/cv-lab2/damavand.jpg'), cv2.imread('2/cv-lab2/eram.jpg')


alpha_1 = np.linspace(0, 1, 300)

for i in range(300):
    alpha_2 = i / 300
    #result = cv2.addWeighted(I, alpha_2, J, 1 - alpha_2, 0)

    #result = cv2.addWeighted(I, alpha_1[i], J, 1 - alpha_1[i], 0)
    cv2.imshow('result', result)
    cv2.waitKey(10)


cv2.waitKey()

cv2.destroyAllWindows()
