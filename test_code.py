from imgproc import *
import cv2
import numpy as np

img = imgread('img/4.jpg')
mat = matread(img)

l = wholehull(img)
print(l)
# l = wholehull(img)
# print(np.matrix(l))
axisa, axisb = findaxis(mat)
center = findcenter(mat)
print(mat.shape)
print(center)
print(axisa * center[0] + axisb)
print(center[1])
img = drawpoly(img, l)
img = drawline(img, axisa, axisb)
imgshow(img)
