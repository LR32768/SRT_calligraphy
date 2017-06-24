from imgproc import *
import cv2
import numpy as np

img = imgread('2.jpg')
l = wholehull(img)
print(l)
# l = wholehull(img)
# print(np.matrix(l))
img = drawpoly(img, l)
imgshow(img)
