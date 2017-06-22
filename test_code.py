from imgproc import *
import cv2

img = cv2.imread('0.jpg')
mat = matread(img)
l = convexhull(img)
cv2.drawContours(img, [l], -1, (0, 0, 0))
cv2.imshow("", img)
