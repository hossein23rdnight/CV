import cv2
import numpy as np
   
I = cv2.imread('/Users/hossein/Desktop/CV/10/kntu4.jpg')
G = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
G = np.float32(G)

for k in range(1,7):
    win_size = 2**k # 16 ->k = 4 kntu1  ||||  8 ->k = 3 kntu2  ||||   4 ->k = 2 kntu4
    soble_kernel_size  = 3 # kernel size for gradients
    alpha = 0.04
    H = cv2.cornerHarris(G,win_size,soble_kernel_size,alpha)
    H = H / H.max()
    
    C = np.uint8(H > 0.01) * 255
    nc,CC = cv2.connectedComponents(C);

    J = I.copy()
    J[C != 0] = [0,0,255]
    cv2.putText(J,'winsize=%d, corners=%d'%(win_size, nc-1),(20,40), \
                cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
    
    cv2.imshow('corners',J)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

    
    
