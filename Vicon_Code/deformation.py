import math
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

def get_K(n, A, B, N):
    NA = (A[0] - N[0], A[1] - N[1])
    NB = (B[0] - N[0], B[1] - N[1])
    alpha = math.acos((NA[0] * NB[0] + NA[1] * NB[1]) / (np.linalg.norm(NA) * np.linalg.norm(NB)))
    K = np.zeros(n + 1)
    K_NA = NA[1] / NA[0]
    K_NA_atan = math.atan(K_NA)
    for i in range(n + 1):
        K[i] = math.tan(-i * alpha / n + K_NA_atan)
    K_mid = math.tan(-1 * alpha / 2 + K_NA_atan)
    return K, K_mid

def get_rate_N(n, N, K, K_mid):
    a = N[0]
    b = N[1]
    K_N = -1 / K_mid
    x = np.zeros(n + 1)
    for i in range(n + 1):
        x[i] = (K[i]*a-b) / (K[i]-K_N)
    rate_N = np.zeros(n)
    for i in range(n):
        rate_N[i] = (x[i+1]-x[i]) / (x[n]-x[0])
    return rate_N

def deformation(image1, n, rate, rate_N):
    high = image1.shape[0]
    width = image1.shape[1]
    depth = image1.shape[2]

    image = []
    x = 0
    for i in range(n):
        n_image = np.zeros([high, math.floor(width * rate_N[i]), depth])
        n_image = image1[0:high, x:x+math.floor(width * rate_N[i]), 0:depth]
        x += math.floor(width * rate_N[i])
        image.append(n_image)

    new_image = []
    for i in range(n):
        n_image = cv2.resize(image[i], (0, 0), fx=rate[i]/rate_N[i], fy=1, interpolation=cv2.INTER_CUBIC)
        new_image.append(n_image)

    result = np.zeros(image1.shape)
    x = 0
    for i in range(n):
        result[0:high, x:x+new_image[i].shape[1], 0:depth] = new_image[i]
        x += new_image[i].shape[1]

    # result = [r.astype(int) for r in result]
    result = np.array(result, dtype=float) / float(255)
    return result

"""
n是将屏幕分为n等分
r是屏幕半径(米)
central_angle是屏幕圆心角
N是人物位置，比如N = (-8,-5)
image1是输入的矩阵

整个函数返回一个与image1大小相同的矩阵
deformation_curve(n, r, central_angle, N, image1)适用于弧形屏幕
"""
def deformation_curve(n, r, central_angle, N, image1):
    theta = central_angle / 2
    A = (-r * math.sin(theta), -r + r * math.cos(theta))
    B = (r * math.sin(theta), -r + r * math.cos(theta))
    a = N[0]
    b = N[1]

    """求n个角平分线的斜率与‘中线’斜率"""
    K, K_mid = get_K(n, A, B, N)

    """求角平分线与弧形屏幕的交点坐标"""
    a_point = np.zeros((n + 1, 2))###角平分线与弧形屏幕的交点
    for i in range(n + 1):
        k = K[i]
        inter = a ** 2 * k ** 2 - 2 * a * b * k - 2 * a * k * r + b ** 2 + 2 * b * r - k ** 2 * r ** 2
        sqrt = math.sqrt(-k ** 2 * inter)
        y = (sqrt - a * k + b - k ** 2 * r) / (k ** 2 + 1)
        x = a + (y - b) / k
        a_point[i] = (x, y)

    """求每个交点之间弧长的占比"""
    arc_len = np.zeros(n)
    angle = np.zeros(n)
    rate = np.zeros(n)###每一段弧长占全长的比例
    for i in range(n):
        angle[i] = math.atan((a_point[i][1] + r) / a_point[i][0]) - math.atan((a_point[i + 1][1] + r) / a_point[i + 1][0])
        if (angle[i] < 0):
            angle[i] = math.pi + angle[i]
        arc_len[i] = r * angle[i]
        rate[i] = arc_len[i] / (r * central_angle)

    """求假象屏幕各段占比"""
    rate_N = get_rate_N(n, N, K, K_mid)#假象平面屏每段占全长的比例

    """做缩放并返回"""
    return deformation(image1, n, rate, rate_N)

"""
n是将屏幕分为n等分
width是屏幕的宽度(米)
N是人物位置，比如N = (-8,-5)
image1是输入的矩阵

整个函数返回一个与image1大小相同的矩阵
deformation_flat(n, width, N, image1)适用于平面屏幕
"""
def deformation_flat(n, width, N, image1):
    A = (-1 * width / 2, 0)
    B = (width / 2, 0)
    a = N[0]
    b = N[1]

    """求n个角平分线的斜率与‘中线’斜率"""
    K, K_mid = get_K(n, A, B, N)

    """求角平分线与弧形屏幕的交点坐标"""
    a_point = np.zeros((n + 1, 2))###角平分线与平面屏幕的交点
    for i in range(n + 1):
        k = K[i]
        y = 0
        x = (k*a-b) / k
        a_point[i] = (x, y)

    """求每个交点之间弧长的占比"""
    rate = np.zeros(n)###每一段长占全长的比例
    for i in range(n):
        rate[i] = (a_point[i+1][0] - a_point[i][0]) / (a_point[n][0] - a_point[0][0])

    """求假象屏幕各段占比"""
    rate_N = get_rate_N(n, N, K, K_mid)

    """做缩放并返回"""
    return deformation(image1, n, rate, rate_N)

"""
这里是调试函数 d
可删
"""
if __name__ == "__main__":
    n = 20  # 分为n份
    r = 8  # 屏幕半径
    central_angle = math.pi / 2  # 屏幕圆心角
    N = (-9, -8)  # 人物位置
    image1 = cv2.imread("2.png")

    result1 = deformation_curve(n,r,central_angle,N,image1)

    cv2.imshow('result', result1)
    cv2.waitKey(0)

    # result2 = deformation_flat(n,16,N,image1)
    # cv2.imwrite('2-.png', result2)