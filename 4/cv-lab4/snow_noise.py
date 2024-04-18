import numpy as np
import cv2

I = cv2.imread('/Users/hossein/Desktop/CV/4/cv-lab4/isfahan.jpg')
I = I.astype(float) / 255

sigma = 0.04  

while True:
    N = np.random.randn(*I.shape) * sigma
    
    sigma = max(0, sigma)
    
    J = I + N
    
    J = np.clip(J, 0, 1)
    
    cv2.imshow('snow noise', J)
    
    key = cv2.waitKey(33)
    
    if key & 0xFF == ord('u'):  
        sigma += 0.05  
    elif key & 0xFF == ord('d'):  
        sigma -= 0.05  
    elif key & 0xFF == ord('q'):  
        break  

cv2.destroyAllWindows()
