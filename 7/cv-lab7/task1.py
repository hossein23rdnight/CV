import numpy as np
import cv2

I = cv2.imread('/Users/hossein/Desktop/CV/7/cv-lab7/coins.jpg')

G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

G = cv2.GaussianBlur(G, (5, 5), 0)

canny_high_threshold = 160
min_votes = 30  
min_centre_distance = 40  
resolution = 1  

circles = cv2.HoughCircles(G, cv2.HOUGH_GRADIENT, resolution, min_centre_distance,
                           param1=canny_high_threshold,
                           param2=min_votes, minRadius=20, maxRadius=60)


num_coins = 0


for c in circles[0, :]:
        x, y, r = c
        cv2.circle(I, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.circle(I, (int(x), int(y)), 2, (0, 0, 255), 2) 
        num_coins += 1


font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(I, f'There are {num_coins} coins!', (10, 30), font, 1, (255, 0, 0), 2)
cv2.imshow("Detected Coins", I)
cv2.waitKey(0)
cv2.destroyAllWindows()