from imgproc import *
import cv2
import numpy as np
'''
for i in range(1, 100):
    img = imgread('img/%d.jpg' % i)
    mat = matread(img)

    l = wholehull(img)
    X, y = line_to_X_y(l)

    # l = wholehull(img)
    # print(np.matrix(l))
    axisa, axisb = findaxis(mat)
    print(i, ': ', line_cut_convex(X, y, axisa, axisb))
'''
img = imgread('img/5.jpg')
mat = matread(img)

l = wholehull(img)
X, y = line_to_X_y(l)

# l = wholehull(img)
# print(np.matrix(l))
axisa, axisb = findaxis(mat)
n, l = line_cut_convex(X, y, axisa, axisb)

center = findcenter(mat)
print(mat.shape)
print(center)
print(axisa * center[0] + axisb)
print(center[1])
img = drawpoly(img, l)
img = drawline(img, axisa, axisb)
imgshow(img)
