
import numpy as np
import cv2
from extract import extract
from masking import mask
from blurring import blur



image_path = '/Users/hossein/Desktop/CV/PROJECTS/project-2-at-2024-06-23-17-27-4bb16a39/images/0d89ec4f-day_14319.jpg'
label_path = '/Users/hossein/Desktop/CV/PROJECTS/project-2-at-2024-06-23-17-27-4bb16a39/labels/0d89ec4f-day_14319.txt' 
cover_path = '/Users/hossein/Desktop/CV/PROJECTS/kntu.jpg' 

image = cv2.imread(image_path)

with open(label_path, 'r') as file:
    label = file.read().strip().split()
    for_corner = np.array(label[1:], dtype=np.float32).reshape(-1, 2)
    
    
    
    
# cover = cv2.imread(cover_path)
# image = mask(image, for_corner, cover)


#image = extract(image, for_corner)


image = blur(image, for_corner)

#print(for_corner)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('resilt_image.jpg',image)