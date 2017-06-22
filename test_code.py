from imgproc import *
import cv2

img = imgread('0.jpg')
mat = matread(img)
l = convexhull(mat)
print(len(l))
drawconvex(img, l)
imgshow(img)
