import numpy as np
import cv2

def draw_line(I,rho,theta):
    "draws a line in image 'I' given 'rho' and 'theta'"
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    
    cv2.line(I,(x1,y1),(x2,y2),(0,0,255),1)

    
I = cv2.imread('/Users/hossein/Desktop/CV/7/cv-lab7/highway.jpg')

G = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)

E = cv2.Canny(G,100,200)

min_votes = 200 
distance_resolution = 1 
angle_resolution = np.pi/180
L = cv2.HoughLines(E,distance_resolution,angle_resolution,min_votes)

for [[rho,theta]] in L:
    draw_line(I,rho,theta)

cv2.imshow("E",E)
cv2.imshow("I",I)
cv2.waitKey(0)
cv2.destroyAllWindows()

