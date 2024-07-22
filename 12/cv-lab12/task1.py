import numpy as np
import cv2
import glob

sift = cv2.SIFT_create()  

I2 = cv2.imread('/Users/hossein/Desktop/CV/12/cv-lab12/scene.jpg')
G2 = cv2.cvtColor(I2,cv2.COLOR_BGR2GRAY)
keypoints2, desc2 = sift.detectAndCompute(G2, None); 

fnames = glob.glob('/Users/hossein/Desktop/CV/12/cv-lab12/obj?.jpg')
fnames.sort()
for fname in fnames:
    I1 = cv2.imread(fname)
    G1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    keypoints1, desc1 = sift.detectAndCompute(G1, None)  

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    alpha = 0.75
    for m1, m2 in matches:
        if m1.distance < alpha * m2.distance:
            good_matches.append(m1)

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    mask = mask.ravel().tolist()
    good_matches = [m for m, msk in zip(good_matches, mask) if msk == 1]

    J = cv2.warpPerspective(I1, H, (I2.shape[1], I2.shape[0]))

    ind = 0
    imgs = [I2, J]
    while True:
        ind = 1 - ind

        cv2.imshow('Reg', imgs[ind])
        key = cv2.waitKey(800)

        if key & 0xFF == ord('q'):
            exit()
        elif key & 0xFF != 0xFF:
            break