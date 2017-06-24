from imgproc import *
import cv2

img = imgread('0.jpg')
mat = matread(img)
l = convexhull(mat)
print(l)
print('---')
print(findaxis(mat))
