import numpy as np
import math
import matplotlib.pyplot as plt

def length(x, y):
    return np.sqrt(x ** 2 + y ** 2)

def sdCircle(x, y, r):
    return length(x,y) - r

def sdBox(x, y, w, h):
    dx = np.abs(x) - w
    dy = np.abs(y) - h
    return length(dx.clip(min=0), dy.clip(min=0)) + np.fmax(dx, dy).clip(max=0)

def sdTriangle(x, y, p0, p1, p2):
    results = []
    e0 = p1 - p0
    e1 = p2 - p1
    e2 = p0 - p2

    v0 = [x - p0[0], y - p0[1]]
    v1 = [x - p1[0], y - p1[1]]
    v2 = [x - p2[0], y - p2[1]]
    
    div0 =  (v0[0] * e0[0] + v0[1] * e0[1]) / np.dot(e0,e0)
    div1 =  (v1[0] * e1[0] + v1[1] * e1[1]) / np.dot(e1,e1)
    div2 =  (v2[0] * e2[0] + v2[1] * e2[1]) / np.dot(e2,e2)
    
    pq0 = [v0[0] - e0[0] * div0.clip(min=0.0, max=1.0), v0[1] - e0[1] * div0.clip(min=0.0, max=1.0)]
    pq1 = [v1[0] - e1[0] * div1.clip(min=0.0, max=1.0), v1[1] - e1[1] * div1.clip(min=0.0, max=1.0)]
    pq2 = [v2[0] - e2[0] * div2.clip(min=0.0, max=1.0), v2[1] - e2[1] * div2.clip(min=0.0, max=1.0)]

    s = np.sign(e0[0] * e2[1] - e0[1] * e2[0])
    vec0 = [pq0[0] * pq0[0] + pq0[1] * pq0[1], s * (v0[0] * e0[1] - v0[1] * e0[0])]
    vec1 = [pq1[0] * pq1[0] + pq1[1] * pq1[1], s * (v1[0] * e1[1] - v1[1] * e1[0])]
    vec2 = [pq2[0] * pq2[0] + pq2[1] * pq2[1], s * (v2[0] * e2[1] - v2[1] * e2[0])]
    d = np.fmin(np.fmin(vec0, vec1), vec2)
    return -np.sqrt(d[0]) * np.sign(d[1])

def sdBezier(x, y, A, B, C):
    results = []
    for i, row in enumerate(x):
        new_row = []
        for j, cell in enumerate(row):
            pos = np.array([cell, y[i][j]])
            a = B - A
            b = A - 2.0 * B + C
            c = a * 2.0
            d = A - pos
            kk = 1.0 / np.dot(b,b)
            kx = kk * np.dot(a,b)
            ky = kk * (2.0 * np.dot(a,a) + np.dot(d,b)) / 3.0
            kz = kk * np.dot(d,a)      
            res = 0.0
            sgn = 0.0
            p = ky - kx * kx
            p3 = p * p * p
            q = kx * (2.0 * kx * kx -3.0 * ky) + kz
            h = q * q + 4.0 * p3

            if h >= 0.0:
                h = np.sqrt(h)
                x = ([h, -h] - q) / 2.0
                uv = np.sign(x) * np.power(np.abs(x), 1.0/3.0)
                t = (uv[0] + uv[1] - kx).clip(min=0.0, max=1.0)
                q = d + (c + b * t) * t
                res = np.dot(q, q)
                sgn = np.cross(c + 2.0 * b * t, q)
            else:
                z = np.sqrt(-p)
                v = np.arccos( q / (p * z * 2.0)) / 3.0
                m = np.cos(v)
                n = np.sin(v) * 1.732050808
                vec = np.array([m+m, -n-m, n-m])
                t = (vec * z - kx).clip(min=0.0, max=1.0)
                qx = d + (c + b * t[0]) * t[0]
                dx = np.dot(qx, qx)
                sx = np.cross(c + 2.0 * b * t[0], qx)
                qy = d + (c + b * t[1]) * t[1]
                dy = np.dot(qy, qy)
                sy = np.cross(c + 2.0 * b * t[1], qy)
                if dx<dy :
                    res = dx
                    sgn = sx
                else:
                    res = dy
                    sgn = sy
                # res = np.fmin(np.dot(d+(c+b*t[0])*t[0], d+(c+b*t[0])*t[0]), np.dot(d+(c+b*t[1])*t[1], d+(c+b*t[1])*t[1]))
            new_row.append(np.sqrt(res) * np.sign(sgn))
        results.append(new_row)
    return results

width = 10
height = 10
x = np.linspace(0, width, 300)
y = np.linspace(0, height, 300)
X, Y = np.meshgrid(x, y)
X -= width/2
Y -= height/2
# Z = sdCircle(X, Y, 4.0)
# Z = sdBox(X, Y, 3.0, 3.0)
Z = sdTriangle(X, Y, np.array([-2.0, -2.0]), np.array([0.0, 3.0]), np.array([2.0, -2.0]))
# Z = sdBezier(X, Y, np.array([-2.0, -2.0]), np.array([0.0, 3.0]), np.array([2.0, 2.0]))

plt.contour(X,Y,Z, cmap="Spectral", levels=20)
plt.colorbar()
plt.show()