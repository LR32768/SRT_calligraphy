import numpy as np
import cv2
from general import two_threshold


def imgread(addr):
    try:
        img = cv2.imread(addr)  # read image into a mat
        NewImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # transform into gray
        for i in range(NewImage.shape[0]):
            for j in range(NewImage.shape[1]):
                Num = NewImage[i][j]
                if Num < two_threshold:
                    NewImage[i][j] = 0
                else:
                    NewImage[i][j] = 255
        return NewImage
    except Exception as e:
        print('There is an error in imgproc.imgread' + str(e))


def imgshow(img):
    cv2.namedWindow("")
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyWindow("")
    return


def matread(img):
    mat = np.zeros(img.shape)
    mat = img.copy()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            Num = mat[i][j]
            if Num == 0:
                mat[i][j] = 1
            else:
                mat[i][j] = 0
    return mat


def matwrite(mat):
    img = np.zeros(mat.shape)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            Num = mat[i][j]
            if Num == 0:
                img[i][j] = 255
            else:
                img[i][j] = 0
    return img


def matget(addr):
    img = imgread(addr)
    mat = matread(img)
    return mat


def pos_black(mat):
    size = mat.shape
    result = []
    for i in range(size[0]):
        for j in range(size[1]):
            if mat[i][j] == 1:
                result.append([i, j])
    return result


def trans3(mat3):
    l = []
    for i in range(len(mat3)):
        l.append([mat3[i][0][0], mat3[i][0][1]])
    return np.matrix(l)


def convexhull(img):
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST,
                                                  cv2.CHAIN_APPROX_NONE)
    l = []
    for i in range(len(contours)):
        cnt = contours[i]
        l.append(cv2.convexHull(cnt, None, True))
    return l


def wholehull(img):
    l = pos_black(matread(img))
    l = np.matrix(l)
    return cv2.convexHull(l, None, True)


def drawconvex(img, cnt):
    cv2.drawContours(img, cnt, -1, (0, 0, 0))


def findcenter(mat, isimg=False):
    result = [0, 0]
    num = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j] == 1 or (isimg and mat[i][j] == 0):
                result[0] += i
                result[1] += j
                num += 1
    return [result[0] / num, result[1] / num]


def findaxis(mat):
    l = pos_black(mat)
    matrix = np.zeros((len(l), 2))
    result = np.zeros((len(l), 1))
    for i in range(len(l)):
        matrix[i][0] = l[i][0]
        matrix[i][1] = 1
    for i in range(len(l)):
        result[i][0] = l[i][1]
    pinv = np.linalg.pinv(matrix)
    line = np.dot(pinv, result)
    return line[0][0], line[1][0]
