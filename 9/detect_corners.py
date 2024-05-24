import cv2
import numpy as np

I = cv2.imread('/Users/hossein/Desktop/CV/9/square.jpg')
G = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)

G = np.float32(G)
window_size = 2
soble_kernel_size  = 3 
alpha = 0.04
H = cv2.cornerHarris(G,window_size,soble_kernel_size,alpha)
print(H)

H = H / H.max()

# C[i,j] == 255 if H[i,j] > 0.01, and C[i,j] == 0 otherwise
C = np.uint8(H > 0.005) * 255

nc,CC = cv2.connectedComponents(C);
nc = nc - 1 

n = np.count_nonzero(C) 

I[CC != 0] = [0,0,255]

cv2.imshow('corners',C)
cv2.waitKey(0) 

font = cv2.FONT_HERSHEY_SIMPLEX 
cv2.putText(I,'There are %d corners!'%nc,(20,40), font, 1,(0,0,255),2)
cv2.imshow('corners',I)
cv2.waitKey(0) 

cv2.destroyAllWindows()

