from imgproc import *
import cv2
import numpy as np

mat = matget('2.jpg')
img = imgread('2.jpg')
feature = 20 * [0.0]


def dot(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


def uni(v):
    length = (v[0]**2 + v[1]**2)**0.5
    nx = v[0] / length
    ny = v[1] / length
    return [nx, ny]


px = []
py = []
n = 0

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        if mat[i][j] == 1:
            px.append(i)
            py.append(j)
            n = n + 1

#Now strech feature 1(convexarea / rectanglearea);
xmax = 0
ymax = 0
xmin = mat.shape[0]
ymin = mat.shape[1]

for i in range(n):
    #print("The coordinate",px[i],' ',py[i]);
    if px[i] < xmin:
        xmin = px[i]
    if px[i] > xmax:
        xmax = px[i]
    if py[i] < ymin:
        ymin = py[i]
    if py[i] > ymax:
        ymax = py[i]

height = xmax - xmin
width = ymax - ymin

print(height)
print(width)

area = height * width
print(area)

# To get the area of convex hull
l = wholehull(img)
print(l)
conarea = 0.0

l.append(l[0])
for i in range(len(l)-1):
   conarea = conarea + l[i][0] * l[i + 1][1] - l[i][1] * l[i + 1][0]

conarea = conarea / 2.0

print(conarea)
feature[0] = conarea / area


# Now strech feature 2, 3;(Minimum deviation line slope)
line = findaxis(img)
print(line)

feature[2] = line[0]
feature[3] = line[1] / height

#Now stretch feature 4; (Gravity center)
Gravitycenter = findcenter(mat)
feature[4] = Gravitycenter[0]
feature[5] = Gravitycenter[1]

#Now strtch feature 5; (White ratio)
blackblocks = len(px)
feature[5] = blackblocks / conarea

print(feature)


