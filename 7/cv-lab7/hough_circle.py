import numpy as np
import cv2

I = cv2.imread('/Users/hossein/Desktop/CV/7/cv-lab7/samand.jpg')

G = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY) 
G = cv2.GaussianBlur(G, (3,3), 0);     

canny_high_threshold = 300 
min_votes = 100 #80
min_centre_distance = 40 
resolution = 1 
circles = cv2.HoughCircles(G,cv2.HOUGH_GRADIENT,resolution,min_centre_distance,
                           param1=canny_high_threshold,
                           param2=min_votes,minRadius=0,maxRadius=110)

print(circles)

for c in circles[0,:]:
    x = int(c[0])  # Convert x-coordinate to integer
    y = int(c[1])  # Convert y-coordinate to integer
    r = int(c[2])  # Convert radius to integer
   
    cv2.circle(I,(x,y), r, (0,255,0),2)

    cv2.circle(I,(x,y),2,(0,0,255),2)
    
E = cv2.Canny(G,10,canny_high_threshold)
cv2.imshow("e",E)
cv2.waitKey(0)

cv2.imshow("I",I)
cv2.waitKey(0)
cv2.destroyAllWindows()


