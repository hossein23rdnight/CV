
import numpy as np
import cv2

I = cv2.imread('/Users/hossein/Desktop/CV/4/cv-lab4/isfahan.jpg').astype(np.float64) / 255

noise_sigma = 0.04  # initial standard deviation of noise

m = 1  # initial filter size,

gm = 3  # gaussian filter size

size = 9  # bilateral filter size
sigmaColor = 0.3
sigmaSpace = 75

# with m = 1 the input image will not change
filter = 'b'  # box filter

while True:

    # add noise to image
    N = np.random.rand(*I.shape) * noise_sigma
    J = I + N
    J = J.astype(np.float32)

    if filter == 'b':
        # filter with a box filter
        blur = cv2.blur(J, (m,m))
    elif filter == 'g':
        # filter with a Gaussian filter
        blur = cv2.GaussianBlur(J, (gm, gm), 0)
        # pass
    elif filter == 'l':
        # filter with a bilateral filter
        # pass
        blur = cv2.bilateralFilter(J,size, sigmaColor, sigmaSpace)

    # filtered image

    cv2.imshow('img', blur)
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
        m = m + 2
        print('m=', m)

    elif key == ord('-'):
        # decrease m
        if m >= 3:
            m = m - 2
        print('m=', m)
    elif key == ord('u'):
        # increase noise
        noise_sigma=noise_sigma+0.1
        if noise_sigma>1 :
            noise_sigma=1
        print('noise_sigma=', noise_sigma)
        # pass
    elif key == ord('d'):
        # decrease noise
        noise_sigma=noise_sigma-0.1
        if noise_sigma<0 :
            noise_sigma=0
        print('noise_sigma=',noise_sigma)
        # pass
    elif key == ord('p'):
        # increase gm
        # sigmaColor+=0.05
        #check to correct answer
        gm=gm+2
        print('gussian_sigma=', gm)
        # pass
    elif key == ord('n'):
        # decrease gm
        gm=gm-2
        if gm<1 :
           gm=1
        print('gussian_sigma=',gm)
        # pass
    elif key == ord('1'):
        sigmaColor+=0.05
        print('sigma color is =',sigmaColor) 

    elif key == ord('2'):
        sigmaColor-=0.05
        print('sigma color is =',sigmaColor) 
    
    elif key == ord('>'):
        # increase size
        size=size+1
        print('size=', size)
        # pass
    elif key == ord('<'):
        # decrease size
        size=size-1
        if size<1 :
            size=1
        print('size=',size)
        # pass
    elif key == ord('q'):
        break  # quit

cv2.destroyAllWindows()
