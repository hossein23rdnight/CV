import cv2
import numpy as np
import matplotlib.pyplot as plt

I = cv2.imread("/Users/hossein/Desktop/CV/3/cv-lab3/pasargadae.jpg", cv2.IMREAD_GRAYSCALE)

levels = 256

# calculating histogram
def calc_hist(I, levels):
    hist = np.zeros(levels)
    rows, cols = I.shape
    for i in range(rows):
        for j in range(cols):
            pixel_value = I[i, j]
            hist[pixel_value] += 1
    return hist


# calculating CDF
def calc_cdf(hist, levels):
    cdf = np.zeros_like(hist)
    cdf[0] = hist[0]
    for i in range(1, levels):
        cdf[i] = cdf[i-1] + hist[i]
    return cdf

# normalize CDF
def normalize_cdf(cdf, total_pixels):
    normalized_cdf = cdf / total_pixels
    return normalized_cdf

# mapping
def create_mapping(normalized_cdf):
    mapping = np.zeros_like(normalized_cdf)
    for i in range(len(normalized_cdf)):
        mapping[i] = np.round(normalized_cdf[i] * 255)
    return mapping
  
  
  
# replace intensity
def replace_intensity(I, mapping):
    rows, cols = I.shape
    equalized_image = np.zeros_like(I)
    for i in range(rows):
        for j in range(cols):
            equalized_image[i, j] = mapping[I[i, j]]
    return equalized_image



hist = calc_hist(I, levels)
cdf = calc_cdf(hist, levels)


total_pixels = np.sum(hist)
normalized_cdf = normalize_cdf(cdf, total_pixels)
mapping = create_mapping(normalized_cdf)
equalized_image = replace_intensity(I, mapping)


equalized_image_hist = calc_hist(equalized_image, levels)
equalized_image_cdf = calc_cdf(equalized_image_hist, levels)

fig = plt.figure(figsize= (16, 8))
fig.add_subplot(2,3,1)
plt.imshow(I, cmap='gray')
plt.title('pasargadae')
plt.axis('off')

fig.add_subplot(2,3,2)
plt.plot(hist)
plt.title('Source histogram')

fig.add_subplot(2,3,3)
plt.plot(cdf)
plt.title('Source CDF')

fig.add_subplot(2,3,4)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized image')
plt.axis('off')

fig.add_subplot(2,3,5)
plt.plot(equalized_image_hist)
plt.title('Equalized histogram')


fig.add_subplot(2,3,6)
plt.plot(equalized_image_cdf)
plt.title('Equalized CDF')

plt.show()