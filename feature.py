from imgproc import *
mat = matget("0.jpg")
feature = 10 * [0.0]


def dot(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


def uni(v):
    length = (v[0]**2 + v[1]**2)**0.5
    nx = v[0] / length
    ny = v[1] / length
    return [nx, ny]


px = []
py = []
n = 0p

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        if mat[i][j] == 1:
            px.append(i)
            py.append(j)
            n = n + 1

#Now strech feature 1;
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

area = height * width
l = convexhull(mat)
convexlist = trans3(l)

convx = []
convy = []

st = 0
for i in range(len(convexlist)):
    if (convexlist[i][0] > convexlist[st][0] or
        (convexlist[i][0] == convexlist[st][0] and
         convexlist[i][1] > convexlist[st][1])):
        st = i
convx.append(convexlist[st][0])
convy.append(convexlist[st][1])
#find the most right down point which must be one of the point of the convex hull

cur = st
curdir = [1, 0]
curdot = 2
nstep = 0

for i in range(len(convexlist)):
    if (i != st):
        newdir = [
            convexlist[i][0] - convexlist[st][0],
            convexlist[i][1] - convexlist[st][1]
        ]
        newdot = dot(uni(curdir), uni(newdir))
        if (newdot > 0 and newdot < curdot):
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
    curdot = 2
    nstep = 0
    for i in range(len(convexlist)):
        if (i != cur):
            newdir = [
                convexlist[i][0] - convexlist[cur][0],
                convexlist[i][1] - convexlist[cur][1]
            ]
            newdot = dot(uni(curdir), uni(newdir))
            if (newdot > 0 and newdot < curdot):
                nstep = i
                curdot = newdot
    curdir = [
        convexlist[nstep][0] - convexlist[cur][0],
        convexlist[nstep][1] - convexlist[cur][1]
    ]
    cur = nstep

print(convx)
print(convy)

# To get the area of convex hull

# Now stech feature 2;
