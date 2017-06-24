from imgproc import *
import cv2


def cut(matl):
    length = 30
    mat = matl[0]
    for i in range(len(mat) - length - 10, len(mat)):
        for j in range(len(mat[0])):
            mat[i][j] = 0
    for i in range(0, length):
        for j in range(len(mat[0])):
            mat[i][j] = 0
    for i in range(len(mat)):
        for j in range(0, length):
            mat[i][j] = 0
    for i in range(len(mat)):
        for j in range(len(mat[0]) - length, len(mat[0])):
            mat[i][j] = 0
    return


fold_source = 'crawler/'  # 'crawler/img/'
fold_object = ''  # 'img/'
try:
    i = 0
    while (True):
        try:
            img = imgread(fold_source + '%d.jpg' % i)
            mat = matread(img)
            mat = matclear(mat)
            mat = matdilate(mat)
            cut([mat])
            img = matwrite(mat)
            cv2.imwrite(fold_object + '%d.jpg' % i, img)
            print(fold_object + "%d.jpg" % i + ' done')
            i += 1
        except:
            i += 1
            print(fold_object + "%d.jpg" % i + ' error')
            continue
except:
    print('End')
