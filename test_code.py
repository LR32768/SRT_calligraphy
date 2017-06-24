from imgproc import *
import cv2

img = imgread('0.jpg')
mat = matread(img)
l = convexhull(img)
# print((l))
# cv2.polylines(img, [trans3(l)], False, (0, 0, 0))
print(l)
print(trans3(l))
imgshow(img)