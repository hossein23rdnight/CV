import numpy as np
import cv2

# Read the image
I = cv2.imread('/Users/hossein/Desktop/CV/4/cv-lab4/isfahan.jpg').astype(np.float64) / 255

# Display the original image
cv2.imshow('original', I)
cv2.waitKey()

# Choose filter size
m = 7

# Create an m by m box filter
F = np.ones((m, m), np.float64) / (m * m)
print(F)

# Now, filter the image using cv2.blur
J = cv2.blur(I, (m, m))
J = cv2.boxFilter(I, -1, (m, m))


# Display the blurred image
cv2.imshow('blurred', J)
cv2.waitKey()

cv2.destroyAllWindows()
