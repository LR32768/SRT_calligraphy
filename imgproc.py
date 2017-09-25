import numpy as np
import cv2
from general import two_threshold
import math
import random
import matplotlib.pyplot as plt


def line_to_X_y(line):
    X = []
    y = []
    for i in range(len(line)):
        X.append(line[i][0])
        y.append(line[i][1])
    return X, y


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


def pos_white(mat):
    size = mat.shape
    result = []
    for i in range(size[0]):
        for j in range(size[1]):
            if mat[i][j] == 0:
                result.append([i, j])
    return result


def trans3(mat3):
    l = []
    for k in range(len(mat3)):
        for i in range(len(mat3[k])):
            l.append([mat3[k][i][0][0], mat3[k][i][0][1]])
    return l


def convexhull(img):
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST,
                                                  cv2.CHAIN_APPROX_NONE)
    l = []
    for i in range(len(contours)):
        cnt = contours[i]
        l.append(cv2.convexHull(cnt, None, True))
    return l


def wholehull(img):
    mat = matread(img)
    feature = 10 * [0.0]

    def xdot(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    def uni(v):
        length = (v[0]**2 + v[1]**2)**0.5
        nx = v[0] / length
        ny = v[1] / length
        return [nx, ny]

    def dot(v1, v2):
        return v1[0] * v2[0] + v1[1] * v2[1]

    px = []
    py = []
    n = 0

    l = convexhull(mat)
    convexlist = trans3(l)

    convx = []
    convy = []

    st = 0
    for i in range(len(convexlist)):
        if (convexlist[i][0] > convexlist[st][0]
                or (convexlist[i][0] == convexlist[st][0]
                    and convexlist[i][1] > convexlist[st][1])):
            st = i
    convx.append(convexlist[st][0])
    convy.append(convexlist[st][1])
    # find the most right down point which must be one of the point of the convex hull

    cur = st
    curdir = [1, 0]
    curdot = -1
    nstep = 0

    for i in range(len(convexlist)):
        if (i != st):
            newdir = [
                convexlist[i][0] - convexlist[st][0],
                convexlist[i][1] - convexlist[st][1]
            ]
            newxdot = xdot(uni(curdir), uni(newdir))
            newdot = dot(uni(curdir), uni(newdir))
            if (newxdot > 0 and newdot > curdot):
                nstep = i
                curdot = newdot
    curdir = [
        convexlist[nstep][0] - convexlist[cur][0],
        convexlist[nstep][1] - convexlist[cur][1]
    ]
    cur = nstep

    while (cur != st):
        convx.append(convexlist[cur][0])
        convy.append(convexlist[cur][1])
        curdot = -1
        nstep = 0
        for i in range(len(convexlist)):
            if (i != cur):
                newdir = [
                    convexlist[i][0] - convexlist[cur][0],
                    convexlist[i][1] - convexlist[cur][1]
                ]
                newxdot = xdot(uni(curdir), uni(newdir))
                newdot = dot(uni(curdir), uni(newdir))
                if (newxdot > 0 and newdot > curdot):
                    nstep = i
                    curdot = newdot
        curdir = [
            convexlist[nstep][0] - convexlist[cur][0],
            convexlist[nstep][1] - convexlist[cur][1]
        ]
        cur = nstep

    result = []
    for i in range(len(convx)):
        result.append([convx[i], convy[i]])
    return result


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


def materode(mat):
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    eroded = cv2.erode(mat, element)
    return eroded


def matdilate(mat):
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    dilated = cv2.dilate(mat, element)
    return dilated


def matclear(mat):
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    opened = cv2.morphologyEx(mat, cv2.MORPH_OPEN, element)
    mat = matdilate(opened)
    return mat


def drawpoly(img, pointlist):
    num = len(pointlist)

    def list2tuple(p):
        point = (int(p[0]), int(p[1]))
        return point

    for i in range(num):
        img = cv2.line(img,
                       list2tuple(pointlist[i]),
                       list2tuple(pointlist[(i + 1) % num]), (50, 0, 0))
    return img


def drawline(img, a, b):
    '''draw line in img by x = ay+b'''

    def list2tuple(p):
        point = (int(p[0]), int(p[1]))
        return point

    maxx = len(img[0])
    img = cv2.line(img, (0, int(-b / a)), (int(maxx), int((maxx - b) / a)),
                   (0, 0, 0))
    return img


def line_cut_convex(X, y, axisa, axisb):
    def get_point_line():
        change_points = []

        def value_cal(i):
            value = X[i] - (y[i] * axisa + axisb)
            return value

        if len(X) != len(y):
            raise Exception('length of X and y error')
        for i in range(1, len(X) + 1):
            value = value_cal(i - 1)
            value_next = value_cal(i % len(X))
            if value_next * value < 0 or value_next**2 + value**2 == 0:
                change_points.append(i - 1)
                change_points.append(i)
        if len(change_points) != 4:
            raise Exception('error in number of the change_points')
        return change_points

    def line_cross(A, B, a, b):
        '''line_cross(A, B, a, b)->[cross_x, cross_y] - x = ay+b cross A-B '''
        result = [[0], [0]]
        num1 = B[1] - A[1]
        num2 = B[0] - A[0]
        nump1 = [[-1, a], [num1, -num2]]
        nump2 = [[-b], [num1 * A[0] - num2 * A[1]]]
        result = np.dot(np.linalg.inv(nump1), nump2)
        return [result[0][0], result[1][0]]

    def index_to_point(i):
        result = [X[i], y[i]]
        return result

    def get_line():
        result = []
        for i in range(len(X)):
            result.append([X[i], y[i]])
        return result

    def area_cal(l):
        conarea = 0.0
        l.append(l[0])
        for i in range(len(l) - 1):
            conarea = conarea + l[i][0] * l[i + 1][1] - l[i][1] * l[i + 1][0]
        conarea = conarea / 2.0
        return conarea

    change_points = get_point_line()
    num1, num2, num3, num4 = change_points[:]
    cross_point1 = line_cross(
        index_to_point(num1), index_to_point(num2), axisa, axisb)
    cross_point2 = line_cross(
        index_to_point(num3), index_to_point(num4), axisa, axisb)
    line1 = [cross_point1]
    line2 = [cross_point2]
    convex = get_line()
    line1 += convex[num2:num3 + 1]
    line1.append(cross_point2)

    line2 += convex[num4:num1:-1]
    line2.append(cross_point1)

    area = area_cal(line1)
    area_whole = area_cal(get_line())
    return area / area_whole


def white_within_convex(img):
    wwc_list = []
    l = wholehull(img)
    polyimg = drawpoly(img, l)
    imgshape = polyimg.shape

    for i in range(0, imgshape[0]):
        f = False
        st = 0

        while not (f and polyimg[i][st] == 255) and st < imgshape[1]:
            if polyimg[i][st] != 0 and polyimg[i][st] != 255:
                f = True
            st += 1
            if st >= len(polyimg[0]):
                break
        f = False
        ed = imgshape[1] - 1
        while not (f and polyimg[i][ed] == 255) and ed >= 0:
            if polyimg[i][ed] != 0 and polyimg[i][ed] != 255:
                f = True
            ed -= 1
        #print(st,' ',ed)
        if ed > st:
            for j in range(st, ed):
                if polyimg[i][j] == 255:
                    wwc_list.append([i, j])

    return wwc_list


def isInChar(pos_x, pos_y, mat):
    size1 = mat.shape[0]
    for i in range(len(mat) - 1):
        x1 = pos_x - mat[i][0]
        y1 = pos_y - mat[i][1]
        x2 = pos_x - mat[(i + 1) % size1][0]
        y2 = pos_y - mat[(i + 1) % size1][1]
        if x1 * y2 - x2 * y1 < 1e-2:
            return False
    return True


def orientation_white_block_length(point, mat, orientation):
    """
    给定一个点和一个数字矩阵和一个方向(非垂直方向),计算在这个方向上的白块长度
    """

    def legal(pos_x, pos_y):
        if (pos_x < 0 or pos_y < 0 or pos_x >= mat.shape[0]
                or pos_y >= mat.shape[1]):
            return False
        #elif not isInChar(pos_x, pos_y, mat):
        #return False
        else:
            return True

    #if orientation % 90 == 0 and orientation != 0:
    #return None
    path_width = 10
    if orientation % 90 != 0 or orientation % 180 == 0:
        path_height = path_width * math.tan(orientation)
    else:
        path_width = 0
        path_height = 10
    n = 0
    left = right = True
    while left or right:
        if left:
            n += 1
            pos_x = (int)(point[0] - path_width * n)
            pos_y = (int)(point[0] - path_height * n)

            if not legal(pos_x, pos_y):
                left = False
            else:
                next_point = mat[pos_x][pos_y]
                if next_point != 0:
                    left = False
        if right:
            n += 1
            pos_x = (int)(point[0] + path_width * n)
            pos_y = (int)(point[0] + path_height * n)

            if not legal(pos_x, pos_y):
                right = False
            else:
                next_point = mat[pos_x][pos_y]
                if next_point != 0:
                    right = False
    return n * math.sqrt(path_width**2 + path_height**2)


def white_block_length(point, mat):
    orientations = (0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 175)
    min_length = mat.shape[0]
    for orientation in orientations:
        length = orientation_white_block_length(point, mat, orientation)
        if length < min_length:
            min_length = length
    return min_length


def white_distribute(img):
    mat = matread(img)
    point_white = white_within_convex(img)
    time = 20
    result = []
    for i in range(0, time):
        pos = random.randint(0, len(point_white) - 1)

        #plt.imshow(mat)
        #plt.plot(point_white[pos][0], point_white[pos][1], 'X')
        #plt.show()
        #print(isInChar(point_white[pos][0], point_white[pos][1], mat))

        #if isInChar(point_white[pos][0], point_white[pos][1], mat):
        #break
        result.append(white_block_length(point_white[pos], mat))
    #print(result)
    avg = 0
    for i in range(len(result)):
        avg += result[i]
    avg /= len(result)
    return np.std(result)
