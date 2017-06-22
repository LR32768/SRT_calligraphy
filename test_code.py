from imgproc import *
import cv2

img = imgread('0.jpg')
mat = matread(img)
l = convexhull(img)
cv2.drawContours(img, [l], -1, (0, 0, 0))
imgshow(img)
