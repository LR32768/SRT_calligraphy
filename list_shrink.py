from imgproc import *
import cv2
import numpy as np

fold = 'img/'
fold_object = 'img/'

num = 0
while (True):
    try:
        img = imgread(fold + '%d.jpg' % num)
        mat = matread(img)
        l = pos_black(mat)
        rd = [0, 0]
        lu = [len(mat), len(mat[0])]
        for point in l:
            if point[0] < lu[0]:
                lu[0] = point[0]
            if point[1] < lu[1]:
                lu[1] = point[1]
            if point[0] > rd[0]:
                rd[0] = point[0]
            if point[1] > rd[1]:
                rd[1] = point[1]
        y = rd[1] - lu[1]
        x = rd[0] - lu[0]
        newmat = np.zeros((x, y))
        for i in range(x):
            for j in range(y):
                newmat[i][j] = mat[lu[0] + i][lu[1] + j]
        newimg = matwrite(newmat)
        cv2.imwrite(fold_object + '%d.jpg' % num, newimg)
        print(fold_object + '%d.jpg' % num)
        num += 1
    except:
        num += 1
        continue
