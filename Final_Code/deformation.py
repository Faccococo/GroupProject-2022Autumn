import math
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

def deformation(n, r, central_angle, N, image1):
    theta = central_angle / 2
    A = (-r * math.sin(theta), -r + r * math.cos(theta))
    B = (r * math.sin(theta), -r + r * math.cos(theta))
    NA = (A[0] - N[0], A[1] - N[1])
    NB = (B[0] - N[0], B[1] - N[1])
    a = N[0]
    b = N[1]

    alpha = math.acos((NA[0] * NB[0] + NA[1] * NB[1]) / (np.linalg.norm(NA) * np.linalg.norm(NB)))

    K = np.zeros(n + 1)

    K_NA = NA[1] / NA[0]
    K_NA_atan = math.atan(K_NA)

    for i in range(n + 1):
        K[i] = math.tan(-i * alpha / n + K_NA_atan)

    a_point = np.zeros((n + 1, 2))

    for i in range(n + 1):
        k = K[i]
        inter = a ** 2 * k ** 2 - 2 * a * b * k - 2 * a * k * r + b ** 2 + 2 * b * r - k ** 2 * r ** 2
        sqrt = math.sqrt(-k ** 2 * inter)
        y = (sqrt - a * k + b - k ** 2 * r) / (k ** 2 + 1)
        x = a + (y - b) / k
        #     print(x**2+(y+r)**2)
        a_point[i] = (x, y)

    arc_len = np.zeros(n)
    angle = np.zeros(n)
    rate = np.zeros(n)
    for i in range(n):
        angle[i] = math.atan((a_point[i][1] + r) / a_point[i][0]) - math.atan(
            (a_point[i + 1][1] + r) / a_point[i + 1][0])
        if (angle[i] < 0):
            angle[i] = math.pi + angle[i]
        arc_len[i] = r * angle[i]
        rate[i] = arc_len[i] / (r * central_angle / n)

    high = image1.shape[0]
    width = image1.shape[1]
    depth = image1.shape[2]

    image = []
    for i in range(n):
        n_image = np.zeros([high, math.floor(width / n), depth])
        n_image = image1[0:high, math.floor(i * width / n):math.floor((i + 1) * width / n), 0:depth]
        image.append(n_image)

    new_image = []
    for i in range(n):
        n_image = cv2.resize(image[i], (0, 0), fx=rate[i], fy=1)
        new_image.append(n_image)

    result = np.zeros(image1.shape)
    x = 0
    for i in range(n):
        result[0:high, x:x + new_image[i].shape[1], 0:depth] = new_image[i]
        x += new_image[i].shape[1]

    return result