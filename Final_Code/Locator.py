import math

import numpy as np


def Locator(position_x, position_y, position_depth):
    def getThreeDimensionalCoordinate(xc, yc, zc, xz, yz, zz, xup, yup, w, h, d, seita):
        # 先忽略了所有绝对值
        t = xz - xc
        u = yz - yc
        v = zz - zc
        rou = 2 * np.tan(seita / 2) / w
        f = np.sqrt(u ** 2 + t ** 2)
        xishu_g = rou * (w / 2 - xup) / f
        xl_g = (xishu_g * u, xishu_g * -t, 0)  # 没改全
        i = np.sqrt(v ** 2 * t ** 2 + u ** 2 * v ** 2 + (u ** 2 + t ** 2) ** 2)
        xishu_s = (h / 2 - yup) * rou / i
        xl_s = (-1 * xishu_s * t * v, -1 * xishu_s * u * v, xishu_s * (u ** 2 + t ** 2))
        j = np.sqrt((t + xl_s[0] + xl_g[0]) ** 2 + (u + xl_s[1] + xl_g[1]) ** 2 + (v + xl_s[2] + xl_g[2]) ** 2)  # 少了个平方
        xu = (d / j) * (t + xl_s[0] + xl_g[0]) + xc
        yu = (d / j) * (u + xl_s[1] + xl_g[1]) + yc
        zu = (d / j) * (v + xl_s[2] + xl_g[2]) + zc
        return [xu, yu, zu]

    answer = getThreeDimensionalCoordinate(0, 0, 0, np.sqrt(3) / 3, np.sqrt(3) / 3, np.sqrt(3) / 3, 2, 4, 4, 8,
                                           np.sqrt(5) / 2, 0.5 * math.pi);
    return 0, 0, 0
