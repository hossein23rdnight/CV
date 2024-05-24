import cv2
import numpy as np

def Nms(H, threshold=0.01):
    corners = np.zeros_like(H)
    H = H / H.max()

    for i in range(1, H.shape[0] - 1):
        for j in range(1, H.shape[1] - 1):
            if H[i, j] > threshold:
                if H[i, j] > H[i-1, j-1] and H[i, j] > H[i-1, j] and H[i, j] > H[i-1, j+1] and \
                   H[i, j] > H[i, j-1] and H[i, j] > H[i, j+1] and \
                   H[i, j] > H[i+1, j-1] and H[i, j] > H[i+1, j] and H[i, j] > H[i+1, j+1]:
                    corners[i, j] = H[i, j]
    return corners

I = cv2.imread('/Users/hossein/Desktop/CV/9/polygons.jpg')
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

ret, T = cv2.threshold(G, 220, 255, cv2.THRESH_BINARY_INV)

nc1, CC1 = cv2.connectedComponents(T)

for k in range(1, nc1):
    Ck = np.zeros(T.shape, dtype=np.uint8)
    Ck[CC1 == k] = 255

    Ck = cv2.GaussianBlur(Ck, (5, 5), 0)
    
    
    window_size = 6
    sobel_kernel_size = 3
    alpha = 0.04
    H = cv2.cornerHarris(np.float32(Ck), window_size, sobel_kernel_size, alpha)
    
    corners = Nms(H)
    
    
    Ck_color = cv2.cvtColor(Ck, cv2.COLOR_GRAY2BGR)
    
    for X in np.argwhere(corners > 0):
        cv2.circle(Ck_color, (X[1], X[0]), 3, (0, 0, 255), -1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(Ck_color, 'There are %d vertices!' % len(np.argwhere(corners > 0)), (20, 30), font, 1, (0, 0, 255), 1)
    
    cv2.imshow('res', Ck_color)
    cv2.waitKey(0)  
cv2.destroyAllWindows()
