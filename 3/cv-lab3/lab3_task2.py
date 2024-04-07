import cv2
import numpy as np
from matplotlib import pyplot as plt

fname = '/Users/hossein/Desktop/CV/3/cv-lab3/office.jpg'

I = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

hist, bins = np.histogram(I.flatten(), 256, [0, 256])

# a = np.min(I) 
# b = np.max(I)
#---------------------
a = np.percentile(I, 5)
b = np.percentile(I, 99.5)
J = np.clip((I - a) * 255.0 / (b - a), 0, 255).astype(np.uint8)
#---------------------



K = cv2.equalizeHist(I)
#---------------------
clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
c = clahe.apply(I)
#---------------------



f, axes = plt.subplots(2, 4)

axes[0, 0].imshow(I, 'gray', vmin=0, vmax=255)
axes[0, 0].axis('off')
axes[0, 0].set_title('Original Image')

axes[1, 0].hist(I.ravel(), 256, [0, 256])
axes[1, 0].set_title('Original Histogram')

axes[0, 1].imshow(J, 'gray', vmin=0, vmax=255)
axes[0, 1].axis('off')
axes[0, 1].set_title('Expanded Image')

axes[1, 1].hist(J.ravel(), 256, [0, 256])
axes[1, 1].set_title('Expanded Histogram')

axes[0, 2].imshow(K, 'gray', vmin=0, vmax=255)
axes[0, 2].axis('off')
axes[0, 2].set_title('Equalized Image')

axes[1, 2].hist(K.ravel(), 256, [0, 256])
axes[1, 2].set_title('Equalized Histogram')

axes[0, 3].imshow(c, 'gray', vmin=0, vmax=255)
axes[0, 3].axis('off')
axes[0, 3].set_title('clahe Image')

axes[1, 3].hist(c.ravel(), 256, [0, 256])
axes[1, 3].set_title('clahe Histogram')

plt.show()
