from imgproc import *
import cv2

img = imgread('3.jpg')
mat = matread(img)
mat = matclear(mat)
l = convexhull(mat)
l = trans3(l)
print(l[0][0])