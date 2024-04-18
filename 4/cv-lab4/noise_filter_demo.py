import numpy as np
import cv2

I = cv2.imread('/Users/hossein/Desktop/CV/4/cv-lab4/isfahan.jpg').astype(np.float32) / 255

noise_sigma = 0.04  # initial standard deviation of noise

m = 1  # initial filter size,

gm = 3  # gaussian filter size

size = 9  # bilateral filter size

sigmaColor = 0.3
sigmaSpace = 25

filter = 'b'  # box filter

while True:

    # add noise to image
    N = np.random.rand(*I.shape) * noise_sigma
    J = I + N
    J = J.astype(np.float32)


    if filter == 'b':
        # filter with a box filter
        K = cv2.blur(J, (m, m))
    
    elif filter == 'g':
        # filter with a Gaussian filter
        K = cv2.GaussianBlur(J, (gm, gm), 0)
    
    elif filter == 'l':
        # filter with a bilateral filter
        #J_uint8 = (J * 255).astype(np.uint8)
        K = cv2.bilateralFilter(J, size, sigmaColor, sigmaSpace)


    # filtered image
    cv2.imshow('img', K)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('b'):
        filter = 'b'  # box filter
        print('Box filter')

    elif key == ord('g'):
        filter = 'g'  # filter with a Gaussian filter
        print('Gaussian filter')

    elif key == ord('l'):
        filter = 'l'  # filter with a bilateral filter
        print('Bilateral filter')

    elif key == ord('+'):
        # increase m
        m += 2
        print('m=', m)

    elif key == ord('-'):
        # decrease m
        if m >= 3:
            m -= 2
        print('m=', m)

    elif key == ord('u'):
        # increase noise intensity
        noise_sigma += 0.01

    elif key == ord('d'):
        # decrease noise intensity
        if noise_sigma >= 0.01:
            noise_sigma -= 0.01

    elif key == ord('p'):
        # increase sigma_color
        sigmaColor += 0.1

    elif key == ord('n'):
        # decrease sigma_color
        if sigmaColor >= 0.1:
            sigmaColor -= 0.1
            
    elif key == ord('>'):
        # increase size
        sigmaSpace += 1

    elif key == ord('<'):
        # decrease size
        if sigmaSpace >= 1:
            sigmaSpace -= 1

    elif key == ord('q'):
        break  # quit

cv2.destroyAllWindows()
