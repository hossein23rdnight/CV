import cv2
import numpy as np
I, J = cv2.imread('2/cv-lab2/damavand.jpg'), cv2.imread('2/cv-lab2/eram.jpg')

print(I.shape)
print(J.shape)
K = I.copy()

#K[::2,::2,:] = J[::2,::2,:]
#K = I//2+J//3

K=cv2.addWeighted(I,alpha,J,beta,gamma)
cv2.imshow("Image 1", I)
cv2.imshow("Image 2", J)
cv2.imshow("Blending", K)
cv2.waitKey(3000)
cv2.destroyAllWindows()