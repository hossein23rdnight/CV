import numpy as np
import cv2
import glob

sift = cv2.SIFT_create()

I2 = cv2.imread('/Users/hossein/Desktop/CV/12/cv-lab12/scene.jpg')
G2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
keypoints2, desc2 = sift.detectAndCompute(G2, None)

fnames = glob.glob('/Users/hossein/Desktop/CV/12/cv-lab12/obj?.jpg')
fnames.sort()

for fname in fnames:
    I1 = cv2.imread(fname)
    G1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    keypoints1, desc1 = sift.detectAndCompute(G1, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    alpha = 0.75
    good_matches = [m1 for m1, m2 in matches if m1.distance < alpha * m2.distance]
    
    points1 = [keypoints1[m.queryIdx].pt for m in good_matches]
    points1 = np.array(points1, dtype=np.float32)
    
    points2 = [keypoints2[m.trainIdx].pt for m in good_matches]
    points2 = np.array(points2, dtype=np.float32)
    
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    
    h, w = I1.shape[:2]
    point = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    
    transform = cv2.perspectiveTransform(point, H).reshape(4, 2)
    
    J = I2.copy()
    cv2.line(J, tuple(map(int, transform[0])), tuple(map(int, transform[1])), (255, 0, 0), 3)
    cv2.line(J, tuple(map(int, transform[1])), tuple(map(int, transform[2])), (255, 0, 0), 3)
    cv2.line(J, tuple(map(int, transform[2])), tuple(map(int, transform[3])), (255, 0, 0), 3)
    cv2.line(J, tuple(map(int, transform[3])), tuple(map(int, transform[0])), (255, 0, 0), 3)
    
    I = cv2.drawMatches(I1, keypoints1, J, keypoints2, good_matches, None)
    
    cv2.imshow('keypoints', I)
    
    if cv2.waitKey() & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
