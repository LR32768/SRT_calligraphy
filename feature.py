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
blackpixels = len(px)

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
a, b = findaxis(mat)
print(a, b)

feature[1] = a
feature[2] = b / width

#Now stretch feature 4; (Gravity center)
Gravitycenter = findcenter(mat)
feature[3] = (Gravitycenter[0]-xmin) / height
feature[4] = (Gravitycenter[1]-ymin) / width

#Now stretch feature 5; (White ratio)
feature[5] = blackpixels / conarea

#Now stretch feature 6; (Area of Left / Area of convex hull)

#Now stretch feature 7; (quadrants distribution)
quad = 4 * [0]

for i in range(blackpixels):
    if px[i] > Gravitycenter[0]:
        if py[i] > Gravitycenter[1]:
            quad[3] += 1
        else:
            quad[2] += 1
    else:
        if py[i] > Gravitycenter[1]:
            quad[0] += 1
        else:
            quad[1] += 1

for i in range(4):
    feature[7 + i] = quad[i] / blackpixels


print(feature)


