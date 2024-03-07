import cv2
import numpy as np

I_gray = cv2.imread('2/cv-lab2/damavand.jpg', cv2.IMREAD_GRAYSCALE)
I_color = cv2.imread('2/cv-lab2/damavand.jpg')

alpha_values = np.linspace(0, 1, 300)

for alpha in alpha_values:
    I_bgr = cv2.cvtColor(I_gray, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(I_bgr, 1 - alpha, I_color, alpha, 0)

    cv2.imshow('result', result)
    cv2.waitKey(10)

cv2.waitKey()
cv2.destroyAllWindows()
