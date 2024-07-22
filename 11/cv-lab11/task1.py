import numpy as np
import cv2
import glob

sift = cv2.SIFT_create() 

I2 = cv2.imread('/Users/hossein/Desktop/CV/11/cv-lab11/scene.jpg')
G2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
keypoints2, desc2 = sift.detectAndCompute(G2, None)

fnames = glob.glob('/Users/hossein/Desktop/CV/11/cv-lab11/obj?.jpg')
fnames.sort()
for fname in fnames:

    I1 = cv2.imread(fname)
    G1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    keypoints1, desc1 = sift.detectAndCompute(G1, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    alpha = 0.75 
    for m, n in matches:
        if m.distance < alpha * n.distance:
            good_matches.append(m)

    I = cv2.drawMatches(I1, keypoints1, I2, keypoints2, good_matches, None)

    no_matches = len(good_matches)
    if no_matches > 30:
        txt = "Object found! (matches = %d)" % no_matches
    else:
        txt = "Object not found! (matches = %d)" % no_matches

    cv2.putText(I, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow('keypoints', I)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
