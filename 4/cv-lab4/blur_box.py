import numpy as np
import cv2

I = cv2.imread('/Users/hossein/Desktop/CV/4/cv-lab4/isfahan.jpg').astype(np.float64) / 255

cv2.imshow('original', I)
cv2.waitKey()

m = 15

F = np.ones((m, m), np.float64) / (m * m)
print(F)

#J = cv2.filter2D(I,-1, F)
#J = cv2.blur(I, (m, m))
J = cv2.boxFilter(I, -1, (m, m))


cv2.imshow('blurred', J)
cv2.waitKey()

cv2.destroyAllWindows()
