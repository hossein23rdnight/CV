import cv2
import numpy as np

I = cv2.imread('/Users/hossein/Desktop/CV/9/polygons.jpg')
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

ret, T = cv2.threshold(G, 220, 255, cv2.THRESH_BINARY_INV)

nc1, CC1 = cv2.connectedComponents(T)

for k in range(1, nc1):
    Ck = np.zeros(T.shape, dtype=np.float32)
    Ck[CC1 == k] = 1
    Ck = cv2.GaussianBlur(Ck, (5,5), 0)
    Ck = cv2.cvtColor(Ck, cv2.COLOR_GRAY2BGR)
    
    # ---------------------
    Ck_gray = cv2.cvtColor(Ck, cv2.COLOR_BGR2GRAY)
    Ck_gray = np.float32(Ck_gray)

   
    window_size = 5
    sobel_kernel_size = 3
    alpha = 0.04
    H = cv2.cornerHarris(Ck_gray, window_size, sobel_kernel_size, alpha)
    H = H / H.max()
    
    C = np.uint8(H > 0.01) * 255

    nC, CC, stats, centroids = cv2.connectedComponentsWithStats(C)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(Ck_gray, np.float32(centroids), (5,5), (-1,-1), criteria)
    
    for i in range(1, nC):
        cv2.circle(Ck, (int(corners[i, 0]), int(corners[i, 1])), 3, (0, 0, 255))
    
    # --------------
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(Ck, 'There are %d vertices!' % (nC - 1), (20, 30), font, 1, (0, 0, 255), 1)
    
    cv2.imshow('corners', Ck)
    cv2.waitKey(0)  # Press any key

cv2.destroyAllWindows()
