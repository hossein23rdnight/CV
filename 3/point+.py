import cv2
import numpy as np
from matplotlib import pyplot as plt

def linear_histogram_expansion(image):
    # Calculate histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    
    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()
    
    # Normalize CDF
    cdf_normalized = cdf / float(cdf.max())
    
    # Find a and b using percentile values
    a = np.percentile(image, 10)  # Adjust percentile value as needed
    b = np.percentile(image, 80)  # Adjust percentile value as needed
    
    # Linearly expand histogram
    J = (image - a) * 255.0 / (b - a)
    
    # Clip values to ensure they are within the valid range [0, 255]
    J[J < 0] = 0
    J[J > 255] = 255
    
    # Convert J to uint8
    J = J.astype(np.uint8)
    
    return J

# Load image
fname = '/Users/hossein/Desktop/CV/3/cv-lab3/crayfish.jpg'
I = cv2.imread(fname)

# Apply linear histogram expansion
J = linear_histogram_expansion(I)

# Plot original and expanded images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(I, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(J, cmap='gray')
plt.title('Expanded Image')
plt.axis('off')

plt.show()
