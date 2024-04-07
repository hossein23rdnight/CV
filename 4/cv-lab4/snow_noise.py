import numpy as np
import cv2

I = cv2.imread('/Users/hossein/Desktop/CV/4/cv-lab4/isfahan.jpg')
I = I.astype(float) / 255

sigma = 0.4  

while True:
    N = np.random.randn(*I.shape) * sigma
    
    # Ensure sigma never goes negative
    sigma = max(0, sigma)
    
    # Add noise to the original image
    J = I + N
    
    # Clip values to ensure they are in the valid range [0, 1]
    J = np.clip(J, 0, 1)
    
    # Display the noisy image
    cv2.imshow('snow noise', J)
    
    # Wait for a key press (33 milliseconds)
    key = cv2.waitKey(33)
    
    if key & 0xFF == ord('u'):  # if 'u' is pressed
        sigma += 0.05  # increase noise intensity
    elif key & 0xFF == ord('d'):  # if 'd' is pressed
        sigma -= 0.05  # decrease noise intensity
    elif key & 0xFF == ord('q'):  # if 'q' is pressed then
        break  # quit

cv2.destroyAllWindows()
