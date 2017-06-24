from imgproc import *
import cv2

img = imgread('3.jpg')
mat = matread(img)
mat = matclear(mat)
img = matwrite(mat)
imgshow(img)